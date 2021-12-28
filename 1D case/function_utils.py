# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 03:45:54 2021

@author: nour
"""


from function_bspline import basis_funs,find_span
import numpy as np
import matplotlib.pyplot as plt
from function_use import *


def point_on_bspline_curve(knots, P, x):
    degree = len(knots) - len(P) - 1
    d = P.shape[-1]

    span = find_span( knots, degree, x )
    b    = basis_funs( knots, degree, x, span )

    c = np.zeros(d)
    for k in range(0, degree+1):
        c[:] += b[k]*P[span-degree+k,:]
    return c


def plot_field_1d(knots, degree, u, nx=101, color='b'):
    n = len(knots) - degree - 1

    xmin = knots[degree]
    xmax = knots[-degree-1]

    xs = np.linspace(xmin, xmax, nx)

    P = np.zeros((len(u), 1))
    P[:,0] = u[:]
    Q = np.zeros((nx, 1))
    for i,x in enumerate(xs):
        Q[i,:] = point_on_bspline_curve(knots, P, x)

    plt.plot(xs, Q[:,0], '-'+color)
    plt.xlabel("x")
    plt.ylabel("$\u03C6_{appr}$")
    plt.show()
    
def plot_field_1d_diff(knots, degree, sol_app_coff,ind_i,M,k, nx=101, color='b'):
    n = len(knots) - degree - 1
    xmin = knots[degree]
    xmax = knots[-degree-1]

    xs = np.linspace(xmin, xmax, nx)

    P = np.zeros((len(sol_app_coff), 1))
    P[:,0] = sol_app_coff[:,ind_i]
    Q = np.zeros((nx, 1))
    for i,x in enumerate(xs):
        Q[i,:] = point_on_bspline_curve(knots, P, x)-exact_wave_homo(x,M,k)[ind_i]

    plt.plot(xs, Q[:,0], '-'+color)
    plt.xlabel("x")
    plt.ylabel("$\u03C6_{exact}(%d)-\u03C6_{appr}(%d)$"%(ind_i,ind_i))
    plt.show()

def norm_2_bsplin(ne,knots,points, weights,sol_app_coff,M,k):
    I_T=0.
    k1 = weights.shape[1]
    for ie in range(0,ne):
        for g1 in range(0,k1):
            diff_exact_app=exact_wave_homo(points[ie,g1],M,k) - point_on_bspline_curve(knots, sol_app_coff,points[ie,g1] ) 
            sum_n=np.sum( (diff_exact_app[:-1] )**2  )
            I_T=I_T+weights[ie,g1]*sum_n
            
            
    return np.sqrt(k*I_T)

        
           
            
def norm_2_exac_app(ne,knots,points, weights,sol_app_coff_lapl):
    I_T=0.
    k1 = weights.shape[1]
    P = np.zeros((len(sol_app_coff_lapl), 1))
    P[:,0] = sol_app_coff_lapl[:]
    for ie in range(0,ne):
        for g1 in range(0,k1):
            diff_exact_app=exact_laplacain(points[ie,g1]) - point_on_bspline_curve(knots, P,points[ie,g1] )[0] 
            I_T=I_T+weights[ie,g1]*(diff_exact_app**2)
            
            
    return np.sqrt(I_T)           
            
        
    
    
    
    
def integralphi_psi(ph_i,psi_j,nbasis, matrix_mass,p):
    I_psi_phi=0.0
    for i in range(0,p+1):
        for j in range(0,i+p+1):
            I_psi_phi=I_psi_phi+ph_i[i]*psi_j[j]*matrix_mass[i,j]
            
    for i in range(p+1,nbasis-p):
        for j in range(i-p,i+p+1):
            I_psi_phi=I_psi_phi+ph_i[i]*psi_j[j]*matrix_mass[i,j]
            
    for i in range(nbasis-p,nbasis):
        for j in range(i-p,nbasis):
            I_psi_phi=I_psi_phi+ph_i[i]*psi_j[j]*matrix_mass[i,j]
            
    return I_psi_phi
    
    
def integral_DeltaPhi_psi(ph_i,psi_j,nbasis, matrix_stiffness,p):
    I_psi_phi=0.0
    for i in range(0,p+1):
        for j in range(0,i+p+1):
            I_psi_phi=I_psi_phi+ph_i[i]*psi_j[j]*matrix_stiffness[i,j]
            
    for i in range(p+1,nbasis-p):
        for j in range(i-p,i+p+1):
            I_psi_phi=I_psi_phi+ph_i[i]*psi_j[j]*matrix_stiffness[i,j]
            
    for i in range(nbasis-p,nbasis):
        for j in range(i-p,nbasis):
            I_psi_phi=I_psi_phi+ph_i[i]*psi_j[j]*matrix_stiffness[i,j]
            
    return I_psi_phi          
            
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    