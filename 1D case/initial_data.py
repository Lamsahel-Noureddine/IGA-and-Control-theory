# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 07:15:36 2021

@author: nour
"""

from function_bspline import *
import numpy as np

############## test of wave solver #############
def v_wave(x):
    #return 1.*x
    #return np.sin(np.pi*x)+x
    return np.sin(np.pi*x)

def w_wave(x):
    #return 1.*x
    #return 1.*x
    return 0.*x
    

############# initial data of the original  wave system  ########## 

def y_0(x):
    #return -(np.pi**2)*np.sin(np.pi*x)
    return np.ones(np.size(x))
    #return np.exp(-(5*(x-0.35))**6)
    '''
    XX=x.copy()
    i=0
    for ind in x:
        if ind<0.5:
            XX[i]=0
            i=i+1
        else:
            XX[i]=1
            i=i+1
    return XX 
    '''
    '''
    XX=x.copy()
    i=0
    for ind in x:
        if ind<1./3:
            XX[i]=0
            i=i+1
        elif 1./3<=ind <2./3:
            XX[i]=1
            i=i+1
        else:
            XX[i]=0
            i=i+1  
    return XX 
    '''
def y_1(x):
    return 0.*x
    
