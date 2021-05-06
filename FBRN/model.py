"""
RestNet model classes.

Classes
-------
    Block      : one layer in iRestNet
    myModel    : FBRestNet model
    Cnn_bar    : predicts the barrier parameter
    Cnn_reg    : predicts the regularisation parameter

@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from math import ceil
import os
# Local import
from FBRN.myfunc import MyMatmul
from FBRN.proxop.hypercube import cardan
from FBRN.proxop.hyperslab import cardan_slab
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# One layer
# Parameters of the network
class Block(torch.nn.Module):
    """
    One layer in myRestNet.
    Attributes
    ----------
        eigv      (torch.nn.FloatTensor): eigenbasis of the Abel integral operator              
        nx,m,a,p                 (float): physical parameters of the Abel integral operator    
        cond                      (bool): True if regularization parameter is learned
        constr                     (str): constraint, 'cube' or 'slab'
        gamma,reg (torch.nn.FloatTensor): stepsize, regularisation parameter, size 1 
        cnn_reg                (Cnn_reg):
        cnn_mu                 (Cnn_bar): computes the barrier parameter
        soft         (torch.nn.Softplus): Softplus activation function
        TtT,DtD,Peig,Pelt     (MyMatmul): 1-D convolution operators 
        gamma_reg,mu          (np.array): saved value to compute lispchitz constant
    """
    def __init__(self,exp,noisy=True,constr='cube'):
        """
        Parameters
        ----------
               exp         (Physic object): contains the experimental parameters  
               noisy                (bool): True if regularization parameter is learned
               constr                (str): constraint, 'cube' or 'slab'
        """
        super(Block, self).__init__()
        #
        self.eigv     = torch.FloatTensor(exp.eigm)
        self.nx       = exp.nx
        self.m        = exp.m
        self.a        = exp.a
        self.p        = exp.p
        # if m is big enough, we cut high frequencies that correspond to noise
        # else m is inferior to nx the projection in cos basis functions as a regularisation
        self.cond     = noisy
        self.constr   = constr
        if self.cond: 
            self.cnn_reg = Cnn_reg(exp)
        else:
            self.reg  = nn.Parameter(torch.FloatTensor([0.0]),requires_grad=False)
        self.gamma    = nn.Parameter(torch.FloatTensor([0.0]))
        self.cnn_mu   = Cnn_bar(self.nx)
        self.soft     = nn.Softplus()
        #
        liste_op      = exp.Operators()
        self.tDD      = MyMatmul(liste_op[0])
        self.tTT      = MyMatmul(liste_op[1])
        self.Peig     = MyMatmul(liste_op[2]) # eltTocos
        self.Pelt     = MyMatmul(liste_op[3]) # cosToelt
        # save gamma and reg for Lipschitz computation
        self.gamma_reg = [0,0]
        self.mu        = [0]

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
      	    x     (torch.nn.FloatTensor): previous iterate, size n*c*nx
            x_b   (torch.nn.FloatTensor): bias, size n*c*nx
        Returns
        -------
       	    (torch.FloatTensor): next iterate, output of the layer, n*c*nx
        """
        # set parameters
        # Barrier parameter
        mu       = self.cnn_mu(self.Pelt(x))
        # Regularisation parameter
        if self.cond:
            reg  = self.cnn_reg(x_b)
        else :
            reg  = self.reg
        # Gradient descent parameter 
        gamma    = 1.0*self.soft(self.gamma)/torch.amax(self.eigv**(-2*self.a)+reg*self.eigv**(2*self.p))
        # compute x_tilde
        x_tilde = x - gamma*self.Grad(reg, x, x_b)
        # project in finite element basis
        x_tilde = self.Pelt(x_tilde)
        # proximity operator
        if self.constr == 'cube':
            x_tilde = cardan.apply(gamma*mu/10**6,x_tilde,self.training)
        if self.constr == 'slab':
            x_tilde = cardan_slab.apply(gamma*mu,x_tilde,self.training)
        # back to eigenvector cos basis
        x_tilde = self.Peig(x_tilde)
        #
        # save
        self.gamma_reg = [gamma.detach().numpy(),reg.detach().numpy()]
        self.mu        = mu.detach().numpy()
        if np.isnan(self.gamma_reg[0]):
            print("sos")
        return x_tilde 

    
class MyModel(torch.nn.Module):
    """
    FBRestNet model.
    Attributes
    ----------
        Layers (torch.nn.ModuleList object): list of iRestNet layers
        nL                            (int): number of layers
        param               (Physic object): contains the experimental parameters
        constr                        (str): constraint, 'cube' or 'slab'
    """
    def __init__(self,exp,noisy=True,nL=20,constr='cube'):
        super(MyModel, self).__init__()
        self.Layers   = nn.ModuleList()
        self.nL       = nL
        self.noisy    = noisy
        self.param    = exp
        self.constr   = constr
        #
        for _ in range(nL):
            self.Layers.append(Block(self.param,self.noisy,self.constr))
        
    # Module Forward
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
            opt1   (str) : "semi" to consider the semi-norm on the output
                             (or "total" to include the norm on the bias)
            opt2   (str) : "entree11" if x_b =T*yd is fed for the input x0 
                             and the bias xb
                            (or "entree01" if x0 is random)
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
        # Step 1 : vectors of eigenvals of T^*T and D^*D
        eig_T = 1/self.param.eigm**(2*self.param.a)
        eig_D = self.param.eigm**(2*self.param.p)
        # Step 2 : on parcourt les layers du reseau
        with torch.no_grad():
            for i in range(nL-1,-1,-1):
                # Acces to the parameters of each layers
                gamma = self.Layers[i].gamma_reg[0]
                reg   = self.Layers[i].gamma_reg[1]
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
                if opt1 == "semi":
                     aip = eig_ip[i,:]**2 + eig_t_ip[i,:]**2
                     # Step 2.2 : consider i=1 first layer U =(0,1) ou (1,1)
                     if i==0:
                         if opt2=="entree01":
                             aip = 2*eig_t_ip[0,:]**2
                         if opt2 == "entree11":
                             aip = 2*(eig_ip[0,:]+eig_t_ip[0,:])**2
                if opt1 == "total":
                     aip  = eig_ip[i,:]**2 + eig_t_ip[i,:]**2 +1 \
                         + np.sqrt(( eig_ip[i,:]**2 + eig_t_ip[i,:]**2+1)**2 \
                         -  4* eig_ip[i,:]**2)                  
                ai[i] = 1/2*np.amax(aip)
            # Step 3.0 : compute theta
            theta     = np.zeros(nL+1)
            theta[0]  = 1
            for i in range(1,nL+1):
                if i ==1:
                    theta[i] = theta[i-1]*np.sqrt(ai[i-1])
                else:
                    theta[i] += theta[i-1]*(1+np.sqrt(ai[i-1]))
            Lip       = theta[-1]/(2**(nL-1))
            # Step 3.1 : if total we go back to the x space
            if opt1 == "total":
                if opt2 == "entree01":
                    Lip = np.sqrt(Lip**2-1)
                if opt2 == "entree11":
                    Lip = np.sqrt(2*Lip**2-1)
        # Step 3 : return
        return Lip
        

# Cnn_reg: to compute the regularisation parameter
class Cnn_reg(nn.Module):
    """
    Predicts the regularisation parameter.
    Attributes
    ----------
        a    (torch.FloatTensor): order of ill-posedness
        p    (torch.FloatTensor): order of a priori
        eig  (torch.FloatTensor): eigenvalues of Abel operator
        m                  (int): size of eigenbasis
        soft (torch.nn.Softplus): Softplus activation function
        inv           (MyMatmul): 1-D convolution operators 
        
    """
    def __init__(self,exp):
        super(Cnn_reg, self).__init__()
        self.a    = nn.Parameter(torch.FloatTensor([exp.a]),requires_grad=False)
        self.p    = nn.Parameter(torch.FloatTensor([exp.p]),requires_grad=False)
        self.eig  = nn.Parameter(torch.FloatTensor(np.diag(exp.eigm)),requires_grad=False)
        #
        self.soft = nn.Softplus()
        self.inv  = MyMatmul(self.eig**self.a)
        # numpy
        self.m    = exp.m
         
    def forward(self,x_in):
        """
        Computes the barrier parameter.
        Parameters
        ----------
      	    x_in (torch.FloatTensor): images, size n*c*nx 
        Returns
        -------
       	    (torch.FloatTensor): barrier parameter, size n*1*1
        """
        size             = x_in.size()
        x_out            = x_in.clone().detach() 
        x_out            = self.inv(x_out)
        x_fil            = x_out.clone().detach()
        nflt             = 4*self.m//5
        x_fil[:,:,nflt:] = torch.zeros((1,1,self.m-nflt))
        #
        delta            = torch.sqrt(torch.sum((x_out-x_fil)**2,2))# estimation de l' erreur
        delta            = delta.view(delta.size(0), -1)
        rho              = torch.ones(1) # torch.sqrt(torch.sum((x_fil)**2,2))# estimation de la norme
        rho              = rho.view(rho.size(0), -1)
        #
        x                = (delta/rho)**(2*(self.a+self.p)/(self.a+2))
        x                = 0.01*self.soft(x)
        x                = x.view(x.size(0),1,1)
        return x
    
# Cnn_bar: to compute the barrier parameter
class Cnn_bar(nn.Module):
    """
    Predicts the barrier parameter.
    Attributes
    ----------
        conv2, conv3  (torch.nn.Conv1d): 1-D convolution layer
        lin2, lin3    (torch.nn.Linear): fully connected layer
        avg        (torch.nn.AVgPool2d): average layer
        soft        (torch.nn.Softplus): Softplus activation function
    """
    def __init__(self,nx):
        super(Cnn_bar, self).__init__()
        self.lin2   = nn.Linear(nx, 256)
        self.conv2  = nn.Conv1d(1, 1, 5,padding=2)
        self.conv3  = nn.Conv1d(1, 1, 5,padding=2)
        self.lin3   = nn.Linear(16*1, 1)
        self.avg    = nn.AvgPool1d(4, 4)
        self.soft   = nn.Softplus()

    def forward(self, x_in):
        """
        Computes the barrier parameter.
        Parameters
        ----------
      	    x_in (torch.FloatTensor): images, size n*c*nx 
        Returns
        -------
       	    (torch.FloatTensor): barrier parameter, size n*1*1
        """
        x = self.lin2(x_in)
        x = self.soft(self.avg(self.conv2(x)))
        x = self.soft(self.avg(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.soft(self.lin3(x))
        x = x.view(x.size(0),1,1)
        return x