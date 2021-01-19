# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from torch.autograd import Variable
import numpy as np
import pandas as pd
import cv2 as cv
import os
from PIL import Image
import matplotlib.pyplot as plt
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# Local import
from FBResNet.myfunc import Physics
from FBResNet.myfunc import MyMatmul
from FBResNet.model import MyModel
from FBResNet.bartlett import Test_cuda

        
class FBRestNet(nn.Module):
    """
    Includes the main training and testing methods of iRestNet.
    Attributes
    ----------
        im_size        (numpy array): image size
        path_test              (str): path to the folder containing the test sets
        path_train             (str): path to the training set folder 
    """
    def __init__(self, experimentation=Physics(2000,20,1,1), nb_blocks=20, noise = 0.1,\
                 folder = './', im_set="Set1",batch_size=[50,5],\
                 lr=1e-3, nb_epochs=[10,1]):
        """
        Parameters
        ----------
           
        """
        super(FBRestNet, self).__init__()   
        # physical information
        self.physics    = experimentation
        self.noise      = noise
        # training information
        self.lr_i       = lr
        self.nb_epochs  = nb_epochs[0]
        self.freq_val   = nb_epochs[1]
        self.nb_blocks  = nb_blocks
        self.nsamples   = batch_size[0]
        self.train_size = batch_size[1] # training set 
        self.val_size   = 1            # and validation set/test set 
        self.im_set     = im_set
        self.loss_fn    = torch.nn.MSELoss(reduction='mean')
        # saving info
        self.path       = folder
        # requires regularisation
        self.regul      = (noise>0)&(self.physics.m>20)
        # model creation
        self.model      = MyModel(self.physics,noisy=self.regul,nL=self.nb_blocks)

    def LoadParam(self):
        """
        Load the parameters of a trained model (in Trainings)
        """
        path_model = self.path + 'Trainings/param{}_{}_{}_{}.pt'.format(\
            self.physics.nx,self.physics.m,self.physics.a,self.physics.p)
        self.model.load_state_dict(torch.load(path_model))
    
    def CreateDataSet(self,save='yes'):
        """
        Creates the dataset from an image basis, rescale, compute transformation and noise.
        Construct the appropriate loader for the training and validation sets.
        Parameters
        ----------
            save       (str) : 'yes' if the data are saved for reloading.
        Returns
        -------
            train_loader
            val_loader
        """
        #
        # Test_cuda()
        # self.device   =
        # self.dtype    =
        # Recuperation des donnees
        nx            = self.physics.nx
        m             = self.physics.m
        noise         = self.noise
        nsample       = self.nsamples
        im_set        = self.im_set
        # Initialisation
        color         = ('b','g','r')
        #
        liste_l_trsf  = []
        liste_tT_trsf = []
        #
        save_lisse    = []
        save_l_trsf   = []
        save_blurred  = []
        save_tT_trsf  = []
        # Upload Data
        # path : './MyResNet/Datasets/Images/'
        for folder, subfolders, filenames in os.walk(self.path+'Datasets/Images/'+im_set+'/'): 
            for img in filenames: 
                item       = folder+img
                img_cv     = cv.imread(item,cv.IMREAD_COLOR)
                for i,col in enumerate(color):
                    # Etape 1 : obtenir l'histogramme lisse des couleurs images
                    histr  = cv.calcHist([img_cv],[i],None,[256],[0,256]).squeeze()
                    # Savitzky-Golay
                    y      = savgol_filter(histr, 21, 5)
                    # interpolation pour nx points
                    x      = np.linspace(0,1,256, endpoint=True)
                    xp     = np.linspace(0,1,nx,endpoint=True)
                    f      = interp1d(x,y)
                    yp     = f(xp)
                    # normalisation
                    ncrop         = nx//20
                    yp[:ncrop]    = 0
                    yp[nx-ncrop:] = 0
                    yp[yp<0]      = 0
                    x_true        = yp/np.amax(yp)
                    # filtering high frequencies
                    filtre = Physics(nx,25)
                    x_true = filtre.BasisChangeInv(filtre.BasisChange(x_true))
                    # reshaping in channelxm
                    x_true        = x_true.reshape(1,-1)
                    # save
                    save_lisse.append(x_true.squeeze())
                    # Etape 2 : passage dans la base de T^*T
                    x_true_trsf = self.physics.BasisChange(x_true)
                    # save
                    liste_l_trsf.append(x_true_trsf)
                    save_l_trsf.append( x_true_trsf.squeeze())
                    #  Etape 3 : obtenir les images bruitees par l' operateur d' ordre a
                    # transform and add noise
                    x_blurred   = self.physics.Compute(x_true) 
                    # Etape 4 : compute adjoint in the cos basis
                    tTx_blurred = self.physics.ComputeAdjoint(x_blurred)
                    # Etape 5 : Bruitage 
                    noise_vect      = np.zeros(m)
                    noise_vect[26:] = np.random.randn(m-26)
                    tTx_blurred    += noise*noise_vect/np.linalg.norm(noise_vect)
                    # save
                    save_blurred.append(x_blurred.squeeze())
                    
                    # and save
                    liste_tT_trsf.append(tTx_blurred)
                    save_tT_trsf.append(tTx_blurred.squeeze())
        # Export data in .csv
        if save =='yes':
            np.savetxt(self.path+'Datasets/Signals/data_lisse.csv',      save_lisse,   delimiter=', ', fmt='%12.8f')
            np.savetxt(self.path+'Datasets/Signals/data_lisse_trsf.csv', save_l_trsf,  delimiter=', ', fmt='%12.8f')
            np.savetxt(self.path+'Datasets/Signals/data_blurred.csv',    save_blurred, delimiter=', ', fmt='%12.8f')
            np.savetxt(self.path+'Datasets/Signals/data_tTblurred.csv',  save_tT_trsf, delimiter=', ', fmt='%12.8f')
        # Tensor completion
        x_tensor = torch.FloatTensor(liste_l_trsf) # signal in cos basis
        y_tensor = torch.FloatTensor(liste_tT_trsf)# blurred and noisy signal in element basis
        #
        dataset = TensorDataset(y_tensor[:nsample], x_tensor[:nsample])
        l       = len(dataset)
        ratio   = 2*l//3
        train_dataset, val_dataset = random_split(dataset, [ratio, l-ratio])
        #
        train_loader = DataLoader(train_dataset, batch_size=self.train_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
        #
        return train_loader, val_loader

    def LoadDataSet():
        """
        Dreates the appropriate loader for the training and validation sets
        when the dataset is already created.
        """
        #
        nsample = self.nsample
        #
        dfl     = pd.read_csv(self.path+'Datasets/Signals/data_lisse_trsf.csv', sep=',',header=None)
        dfb    = pd.read_csv(folder+'Datasets/Signals/data_tTblurred.csv', sep=',',header=None)
        _,m     = dfl.shape
        _,nx    = dfb.shape
        #
        x_tensor = torch.FloatTensor(dfl.values[:nsample]).view(-1,1,m)
        y_tensor = torch.FloatTensor(dfb.values[:nsample]).view(-1,1,nx)
        #
        dataset = TensorDataset(y_tensor, x_tensor)
        l = len(dataset)
        ratio = 2*l//3
        train_dataset, val_dataset = random_split(dataset, [ratio, l-ratio])
        #
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
        #
        return train_loader, val_loader
    
    def train(self,train_set,val_set,test_lipschitz=True,save_model=False):
        """
        Trains iRestNet.
        """      
        # to store results
        nb_epochs    = self.nb_epochs
        nb_val       = self.nb_epochs//self.freq_val
        loss_train   =  np.zeros(nb_epochs)
        loss_val     =  np.zeros(nb_val)
        lip_cste     =  np.zeros(nb_val)
        # defines the optimizer
        lr_i        = self.lr_i
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()),lr=self.lr_i)     #========================================================================================================
        # trains for several epochs
        for epoch in range(0,self.nb_epochs): 
            # sets training mode
            self.model.train()
            # modifies learning rate
            if epoch>0:
                lr_i      = self.lr_i*0.9 
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()), lr=lr_i)
            # TRAINING
            # goes through all minibatches
            for i,minibatch in enumerate(train_set):
                [y, x] = minibatch    # get the minibatch
                x_bias    = Variable(y,requires_grad=False)
                x_true    = Variable(x,requires_grad=False) 
                # definition of the initialisation tensor
                x_init   = torch.zeros(x_bias.size())
                tDD       = MyMatmul(self.physics.Operators()[0])
                x_init[:,:,:25] = tDD(y)[:,:,:25]
                x_init   = Variable(x_init,requires_grad=False)
                # prediction
                x_pred    = self.model(x_init,x_bias) 
                # Computes and prints loss
                loss               = self.loss_fn(x_pred, x_true)
                loss_train[epoch] += torch.Tensor.item(loss)
                    
                # sets the gradients to zero, performs a backward pass, and updates the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # normalisation
            loss_train[epoch] = loss_train[epoch]/i
            #
            # VALIDATION AND STATS
            if epoch%self.freq_val==0:
                # Validation
                with torch.no_grad():
                # tests on validation set
                    self.model.eval()      # evaluation mode
                    for i,minibatch in enumerate(val_set):
                        [y, x] = minibatch            # gets the minibatch
                        x_true  = Variable(x,requires_grad=False)
                        x_bias  = Variable(y,requires_grad=False)
                        # definition of the initialisation tensor
                        x_init   = torch.zeros(x_bias.size())
                        tDD       = MyMatmul(self.physics.Operators()[0])
                        x_init[:,:,:25] = tDD(y)[:,:,:25]
                        x_init   = Variable(x_init,requires_grad=False)
                        # prediction
                        x_pred  = self.model(x_init,x_bias).detach()
                        # computes loss on validation set
                        loss_val[epoch//self.freq_val] += torch.Tensor.item(self.loss_fn(x_pred, x_true))
                    # normalisation
                    loss_val[epoch//self.freq_val] = loss_val[epoch//self.freq_val]/i
                # print stat
                print("epoch : ", epoch," ----- ","validation : ",loss_val[epoch//self.freq_val])
                # Test Lipschitz
                lip_cste[epoch//self.freq_val] = self.model.Lipschitz()
                
            
        #==========================================================================================================
        # training is finished
        print('-----------------------------------------------------------------')
        print('Training is done.')
        print('-----------------------------------------------------------------')
        
        # Plots
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(np.linspace(0,1,nb_epochs),loss_train,label = 'train')
        ax1.plot(np.linspace(0,1,nb_val),loss_val,label = 'val')
        ax1.legend()
        ax2.plot(np.linspace(0,1,nb_val),lip_cste,'r-')
        ax2.set_title("Lipschitz constant")
        plt.show()
        #
        print("Final Lipschitz constant = ",lip_cste[-1])
        
        # Save model
        if save_model:
            torch.save(self.model.state_dict(), self.path+'Trainings/param{}_{}_{}_{}.pt'.format(\
            self.physics.nx,self.physics.m,self.physics.a,self.physics.p))
    
    def test(self,data_set):    
        """
        Parameters
        ----------
            dataset        (Dataloader): the test set
        """
        # initial
        l_x_true = []
        l_x_pred = []
        l_loss   = []
        torch_zeros = Variable(torch.zeros(1,1,self.physics.m),requires_grad=False)
        # gies through the minibatch
        with torch.no_grad():
            self.model.eval()
            for i,minibatch in enumerate(data_set):
                [y, x] = minibatch            # gets the minibatch
                x_true = Variable(x,requires_grad=False)
                x_bias = Variable(y,requires_grad=False)
                # definition of the initialisation tensor
                x_init   = torch.zeros(x_bias.size())
                tDD       = MyMatmul(self.physics.Operators()[0])
                x_init[:,:,:25] = tDD(y)[:,:,:25]
                x_init   = Variable(x_init,requires_grad=False)
                # prediction
                x_pred    = self.model(x_init,x_bias)
                # compute loss
                loss   = torch.Tensor.item(self.loss_fn(x_pred, x_true))
                norm   = torch.Tensor.item(self.loss_fn(torch_zeros, x_true))
                #
                l_x_true.append(x_true.numpy())
                l_x_pred.append(x_pred.numpy())
                l_loss.append(loss/norm)
        # Plots
        xtc = l_x_true[0][0,0]
        xpc = l_x_pred[0][0,0]
        xt  = self.physics.BasisChangeInv(xtc)
        xp  = self.physics.BasisChangeInv(xpc)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(xtc,label = 'true')
        ax1.plot(xpc,label = 'pred')
        ax1.legend()
        ax2.plot(np.linspace(0,1,self.physics.nx),xt,label = 'true')
        ax2.plot(np.linspace(0,1,self.physics.nx),xp,label = 'pred')
        ax2.set_title("Comparaison")
        ax2.legend()
        plt.show()
        
        print("Erreur relative : ",l_loss[0])
