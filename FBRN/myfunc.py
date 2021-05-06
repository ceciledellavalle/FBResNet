"""
Classes and Functions used in the FBResNet model.
Classes
-------
    Physics  :
    MyMatmul : Multiplication with a kernel (for single or batch)
     
@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

#

#
class Physics:
    """
    Define the physical parameters of the ill-posed problem.
    Alert : nx must be >> than m.
    Attributes
    ----------
        nx         (int): size of initial signal 
        m          (int): size of eigenvectors span
        a          (int): oder of ill-posedness 
        p          (int): order of a priori smoothness
        eigm  (np.array): transformation between signal and eigenvectors basis
        Top   (np.array): Abel operator from the finite element basis to the cos basis
    """
    def __init__(self,nx=2000,m=50,a=1,p=1):
        """
        Alert : nx must be >> than m.
        """
        # Physical parameters
        self.nx   = nx
        self.m    = m
        self.a    = a
        self.p    = p
        # Eigenvalues
        self.eigm = (np.linspace(0,m-1,m)+1/2)*np.pi
        # Basis transformation
        base       = np.zeros((self.m,self.nx))        
        h          = 1/(self.nx-1)
        eig_m      = self.eigm.reshape(-1,1)
        v1         = ((2*np.linspace(0,self.nx-1,self.nx)+1)*h/2).reshape(1,-1)
        v2         = (np.ones(self.nx)/2*h).reshape(1,-1)
        base       = 2*np.sqrt(2)/eig_m*np.cos(v1*eig_m)*np.sin(v2*eig_m)
        self.basis = base
        # # Operator T
        # step 0 : Abel operator integral
        # the image of the cos(t) basis is projected in a sin(t) basis
        Tdiag      = np.diag(1/self.eigm**self.a)
        # step 1 : From sin(t) basis to cos(t) basis
        eig_m      = self.eigm.reshape(-1,1)
        base_sin   = np.zeros((self.m,self.nx))
        base_sin   = 2*np.sqrt(2)/eig_m*np.sin(v1*eig_m)*np.sin(v2*eig_m)
        # step 2 : Combinaison of Top and base change
        self.Top = np.matmul(base_sin.T,Tdiag)
        
    def BasisChange(self,x):
        """
        Change basis from signal to eigenvectors span.
        Parameters
        ----------
            x (np.array): signal of size n*c*nx
        Returns
        -------
            (np.array): of size n*c*m
        """
        return np.matmul(x,(self.basis).T)
    
    def BasisChangeInv(self,x):
        """
        Change basis from eigenvectors span to signal.
        Parameters
        ----------
            x (np.array): signal of size nxcxm
        Returns
        -------
            (np.array): of size nxcxnx
        """
        return np.matmul(x,self.basis*self.nx)
    
    def Operators(self):
       """
       Given a ill-posed problem of order a and an a priori of order p
       for a 1D signal of nx points,
       the fonction computes the array of the linear transformation T
       and arrays used in the algorithm.
       Returns
       -------
           (list): four numpy array, the regularisation a priori, the Abel operator,
                   the ortogonal matrix from element to eigenvector basis
       """
       # T  = 1/nx*np.tri(nx, nx, 0, dtype=int).T # matrice de convolution
       Top      = np.diag(1/self.eigm**(self.a))
       # D  = 2*np.diag(np.ones(nx)) - np.diag(np.ones(nx-1),-1) - np.diag(np.ones(nx-1),1)# matrice de dérivation
       Dop      = np.diag(self.eigm**(self.p))
       # matrix P of basis change from cos -> elt
       eltTocos = self.basis
       cosToelt = self.basis.T*self.nx
       # Convert to o Tensor
       tDD      = Dop*Dop
       tTT      = Top*Top
       #
       return [tDD,tTT,eltTocos,cosToelt]
      
    def Compute(self,x):
        """
        Compute the transformation by the Abel integral operator
        in the basis of finite element.
        Parameters
        ----------
            x (np.array): signal of size n*c*nx
        Returns
        -------
            (np.array): of size n*c*nx
        """
        # Change to eig basis
        xeig = self.BasisChange(x)
        # Operator T : Abel operator integral
        return np.matmul(xeig,self.nx*self.Top.T)
    
    def ComputeAdjoint(self,y):
        """
        Compute the transformation by the adjoint operator of Abel integral
        from the basis of finite element to eigenvectors.
        Parameters
        ----------
            x (np.array): signal of size n*c*nx
        Returns
        -------
            (np.array): of size n*c*m
        """
        # T*= tT
        # < en , T^* phi_m > = < T en , phi_m > 
        return np.matmul(y,self.Top)

#
class MyMatmul(nn.Module):
    """
    Performs 1D convolution with numpy array kernel
    Attributes
    ----------
        kernel (torch.FloatTensor): size nx*nx filter
    """
    def __init__(self, kernel):
        """
        Parameters
        ----------
            kernel (numpy array): convolution filter
        """
        super(MyMatmul, self).__init__()
        kernel_nn     = torch.FloatTensor(kernel)
        self.kernel   = nn.Parameter(kernel_nn.T,requires_grad=False)   
            
    def forward(self, x): 
        """
        Performs convolution.
        Parameters
        ----------
            x (torch.FloatTensor): 1D-signal, size n*c*nx
        Returns
        -------
            (torch.FloatTensor): result of the convolution, size n*c*nx
        """
        x_tilde = torch.matmul(x,self.kernel)
        return x_tilde

