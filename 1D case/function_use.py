# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 10:08:27 2021

@author: nour
"""
import numpy as np

##########exact solution of nonhommogenuous wave system ########
def exact_wave_homo(x,M,k):
    exact_wave_x=np.zeros(M+2)
    for i in range(0,M+2):
        ti=i*k
        #exact_wave_i=ti*x+x
        #exact_wave_i=np.sin(np.pi*x)*np.cos(np.pi*ti)+(ti+1)*x
        exact_wave_i=np.sin(np.pi*x)*np.cos(np.pi*ti)
        exact_wave_x[i]=exact_wave_i
    
    return exact_wave_x




####################### exact solution of poisson's equation ##########
def exact_laplacain(x):
    return np.sin(np.pi*x)