"""
RestNet model classes.
Classes
-------
    Block      : one layer in iRestNet
    myModel    : iRestNet model
    Cnn_bar    : predicts the barrier parameter

@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import math
from math import ceil
import os
# Local import
from FBResNet.myfunc import MyMatmul
from FBResNet.proxop.hypercube import cardan
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# One layer
# Parameters of the network
class Block(torch.nn.Module):
    """
    One layer in myRestNet.
    Attributes
    ----------
        cnn_bar                           (Cnn_bar): computes the barrier parameter
        soft                    (torch.nn.Softplus): Softplus activation function
        gamma                (torch.nn.FloatTensor): stepsize, size 1 
        reg_mul,reg_constant (torch.nn.FloatTensor): parameters for estimating the regularization parameter, size 1
        DtD  (MyConv1d): 1-D convolution operators for the derivation
        TtT  (MyConv1d): 1-D convolution operator corresponding to TtT
        mass (scalar):  maximal integral value
    """
    def __init__(self,exp,noisy=False):
        """
        Parameters
        ----------
        exp         (Physic object) : contains the experimental parameters   
        """
        super(Block, self).__init__()
        #
        self.eigmax   = exp.eigm[-1]
        self.nx       = exp.nx
        self.m        = exp.m
        # if m is big enough, we cut high frequencies that correspond to noise
        # else m is inferior to nx the projection in cos basis functions as a regularisation
        self.cond     = noisy
        if self.cond: 
            self.reg  = nn.Parameter(torch.FloatTensor([0.001]))
        else:
            self.reg  = nn.Parameter(torch.FloatTensor([0.0]),requires_grad=False)
        self.gamma    = nn.Parameter(torch.FloatTensor([0.1]))
        self.mu       = nn.Parameter(torch.FloatTensor([0.0]))
        self.soft     = nn.Softplus()
        #
        liste_op      = exp.Operators()
        self.tDD      = MyMatmul(liste_op[0])
        self.tTT      = MyMatmul(liste_op[1])
        self.Peig     = MyMatmul(liste_op[2]) # eltTocos
        self.Pelt     = MyMatmul(liste_op[3]) # cosToelt

    def Grad(self,reg,x,x_b):
        """
        Computes the gradient of the smooth term in the objective function (data fidelity + regularization).
        Parameters
        ----------
      	    reg        (torch.FloatTensor): regularization parameter, size batch*1
            x            (torch.nn.Tensor): images, size batch*c*nx
            x_b          (torch.nn.Tensor):result of Ht applied to the degraded images, size batch*c*nx
        Returns
        -------
       	    (torch.FloatTensor): gradient of the smooth term in the cost function, size batch*c*nx
        """
        tDDx      = self.tDD(x)
        tTTx      = self.tTT(x)
        return tTTx - x_b + reg * tDDx

    def forward(self,x,x_b):
        """
        Computes the next iterate, output of the layer.
        Parameters
        ----------
      	    x     (torch.nn.FloatTensor): previous iterate, size n*c*h*w
            x_b   (torch.nn.FloatTensor): size n*c*nx
            mode_training         (bool): True if training mode, False else
            save_theta             (str): indicates if the user wants to save the values of the lipschitz cste
                                                 path to the folder to save the hyperparameters values or 'no' 
        Returns
        -------
       	    (torch.FloatTensor): next iterate, output of the layer, n*c*h*w
        """
        # set parameters
        # Barrier parameter
        mu       = self.soft(self.mu)/10**6
        # Gradient descent parameter 
        gamma    = self.soft(self.gamma)
        # Regularisation parameter
        if self.cond: 
            reg = self.soft(self.reg)/self.eigmax**2
            # gamma    = gamma/(reg*self.eigmax) # rescaling the gradient descent step
        else :
            reg = self.reg
        # compute x_tilde
        x_tilde = x - gamma*self.Grad(reg, x, x_b)
        # project in finite element basis
        x_tilde = self.Pelt(x_tilde)
        # proximal operator
        x_tilde = cardan.apply(gamma*mu,x_tilde,self.training)
        # back to eigenvector cos basis
        x_tilde = self.Peig(x_tilde)
        return x_tilde 

    
class MyModel(torch.nn.Module):
    """
    iRestNet model.
    Attributes
    ----------
        Layers (torch.nn.ModuleList object): list of iRestNet layers
        nL                            (int): number of layers
        param               (Physic object): contains the experimental parameters
    """
    def __init__(self,exp,noisy=False,nL=20):
        super(MyModel, self).__init__()
        self.Layers   = nn.ModuleList()
        self.nL       = nL
        self.noisy    = noisy
        self.param    = exp
        #
        for _ in range(nL):
            self.Layers.append(Block(self.param,self.noisy))
        

    def forward(self,x,x_b):
        """
        Computes the output of the layer.
        Parameters
        ----------
      	    x            (torch.nn.FloatTensor): previous iterate, size n*nx
            x_b          (torch.nn.FloatTensor): initial signal, size n*nx
        Returns
        -------
       	    (torch.FloatTensor): the output of the network, size n*c*h*w
        """
        for i in range(0,len(self.Layers)):
                x = self.Layers[i](x,x_b)
        return x
    
    # Lifschitz : Lifschitz constant of the network
    def Lipschitz(self,opt1="semi",opt2="entree01"):
        """
        Given a ill-posed problem of order a and a regularization of order p
        for a 1D signal of nx points,
        the fonction Physics compute the tensor of the linear transformation Trsf
        and the tensor used in the algorithm.
        Parameters
        ----------
            model (MyModel): model of the neural network
            opt      (str) : "semi" to consider the semi-norm on the output
                             (or "total" to include the norm on the bias)
        Returns
        -------
            (float): Lifschitz constant theta of the neural network
    
        """
        # Step 0.0 : initialisation
        nL       = len(self.Layers) # number of layers
        m        = self.param.m  # dimension of the eigenvector space
        #
        eig_ref  = np.zeros((nL,m))
        eig_ip   = np.zeros((nL,m))
        eig_t_ip = np.zeros((nL,m))
        ai       = np.zeros(nL)
        theta    = 1.0
        # Step 1 : vectors of eigenvals of T^*T and D^*D
        eig_T = 1/self.param.eigm**(2*self.param.a)
        eig_D = self.param.eigm**(2*self.param.p)
        # Step 2 : on parcourt les layers du reseau
        with torch.no_grad():
            for i in range(nL-1,-1,-1):
                # Acces to the parameters of each layers
                gamma = self.Layers[i].gamma.detach().numpy()
                reg   = self.Layers[i].reg.detach().numpy()
                mu    = 1
                # Computes the ref eigenvals
                eig_ref[i,:] = 1 - gamma*(eig_T+reg*eig_D)
                # Step 2.0 Computes beta_i,p
                for p in range(0,m):
                    if i==nL-1:
                        eig_ip[i,p]   = eig_ref[i,p]
                        eig_t_ip[i,p] = gamma
                    else:
                        eig_ip[i,p]   = eig_ip[i+1,p]*eig_ref[i,p]
                        eig_t_ip[i,p] = eig_t_ip[i+1,p]+gamma*np.prod(eig_ref[i+1:,p])
                # Step 2.1 : compute ai
                if opt1=="semi":
                     aip = eig_ip[i,:]**2 + eig_t_ip[i,:]**2
                     # Step 2.2 : consider i=1 first layer U =(0,1) ou (1,1)
                     if i==0:
                         if opt2=="entree01":
                             aip = 2*eig_t_ip[0,:]**2
                         if opt2 == "entree11":
                             aip = 2*(eig_ip[0,:]+eig_t_ip[0,:])**2
                if opt1=="total":
                     aip  = eig_ip[i,:]**2 + eig_t_ip[i,:]**2 +1 \
                         + np.sqrt(( eig_ip[i,:]**2 + eig_t_ip[i,:]**2+1)**2 \
                         -  4* eig_ip[i,:]**2)                  
                ai[i] = 1/2*np.amax(aip)
            # Step 3.0 : compute theta
            theta     = np.zeros(nL+1)
            theta[0]  = 1
            for i in range(1,nL+1):
                if i==1:
                    theta[i] = theta[i-1]*np.sqrt(ai[i-1])
                else:
                    theta[i] += theta[i-1]*(1+np.sqrt(ai[i-1]))
            #
            Lip = theta[-1]/(2**(nL-1))
            # Step 3.1 : if total we go back to the x space
            if opt1=="total":
                if opt2=="entree01":
                    Lip = np.sqrt(Lip**2-1)
                if opt2=="entree11":
                    Lip = np.sqrt(2*Lip**2-1)
        # Step 3 : return
        return Lip
        