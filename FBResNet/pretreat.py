"""
RestNet model classes.
Modules
-------
    CreateDataSet  : ...

@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import cv2 as cv
import os
import sys
from PIL import Image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
from FBResNet.myfunc import Physics


def CreateDataSet(test,path,noise=0.1,nsample=50,save='yes'):
    # Recuperation des donnees
    m             = test.m
    nx            = test.nx
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
    for folder, subfolders, filenames in os.walk(path+'/'+'Images/Set1/'): 
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
                ncrop        = nx//20
                yp[:ncrop]   = yp[ncrop]
                yp[nx-ncrop:]= 0
                yp[yp<0]     = 0
                x_true       = yp/np.amax(yp)
                # filtering high frequencies
                filtre = Physics(test.nx)
                x_true = filtre.BasisChangeInv(filtre.BasisChange(x_true))
                # reshaping in channelxm
                x_true       = x_true.reshape(1,-1)
                # save
                save_lisse.append(x_true.squeeze())
                # Etape 2 : passage dans la base de T^*T
                x_true_trsf = test.BasisChange(x_true)
                # save
                liste_l_trsf.append(x_true_trsf)
                save_l_trsf.append( x_true_trsf.squeeze())
                #  Etape 3 : obtenir les images bruitees par l' operateur d' ordre a
                # transform and add noise
                x_blurred   = test.Compute(x_true) 
                # Etape 5 : Bruitage 
                noise_vect      = np.zeros(m)
                noise_vect[25:] = np.random.randn(m-25)
                noise_vect      = test.BasisChangeInv(noise_vect.reshape(1,-1))
                noise_vect      = np.sqrt(nx)*noise_vect\
                                  *np.linalg.norm(x_blurred)/np.linalg.norm(noise_vect)
                x_blurred       += noise*noise_vect
                # save
                save_blurred.append(x_blurred.squeeze())
                # Etape 6 : compute adjoint in the cos basis
                tTx_blurred = test.ComputeAdjoint(x_blurred)
                # and save
                liste_tT_trsf.append(tTx_blurred)
                save_tT_trsf.append( tTx_blurred.squeeze())
    # Export data in .csv
    if save =='yes':
        np.savetxt(path+'/Signals/data_lisse.csv',      save_lisse,   delimiter=', ', fmt='%12.8f')
        np.savetxt(path+'/Signals/data_lisse_trsf.csv', save_l_trsf,  delimiter=', ', fmt='%12.8f')
        np.savetxt(path+'/Signals/data_blurred.csv',    save_blurred, delimiter=', ', fmt='%12.8f')
        np.savetxt(path+'/Signals/data_tTblurred.csv',  save_tT_trsf, delimiter=', ', fmt='%12.8f')
    # Tensor completion
    x_tensor = torch.FloatTensor(liste_l_trsf) # signal in cos basis
    y_tensor = torch.FloatTensor(liste_tT_trsf)# blurred and noisy signal in element basis
    #
    dataset = TensorDataset(y_tensor[:nsample], x_tensor[:nsample])
    l       = len(dataset)
    ratio   = 2*l//3
    train_dataset, val_dataset = random_split(dataset, [ratio, l-ratio])
    #
    #train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    #val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
    #
    return save_lisse, save_l_trsf, save_blurred, save_tT_trsf
    # return train_loader, val_loader

def LoadDataSet(folder,nsample=50):
    """
    According to the mode, creates the appropriate loader 
    for the training and validation sets.
    To reuse a data set.
    """
    dfl     = pd.read_csv(folder+'/'+'data_lisse_trsf.csv', sep=',',header=None)
    dfb    = pd.read_csv(folder+'/'+'data_tTblurred.csv', sep=',',header=None)
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
    return train_loader, val_loader
    
            
