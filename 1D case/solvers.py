# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 07:07:23 2021

@author: nour
"""

from function_bspline import *
from initial_data import *
from scipy import linalg as lg
from scipy.sparse import linalg , coo_matrix
import numpy as np


def solverlinear(matrA,vecb):
    cmatrA=matrA.copy()
    cvecb=vecb.copy()
    
    cmatrA=coo_matrix(cmatrA)
    cmatrA=cmatrA.tocsc()
    cmatrA=cmatrA.astype(np.float64)
    cvecb=cvecb.astype(np.float64)
    #C=dsolve.spsolve(A,b,use_umfpack=False)
    lu=linalg.splu(cmatrA)
    C_sol=lu.solve(cvecb)
    return C_sol


def intepol_functions(func,knots, degree):
    Grivis=greville( knots, degree, periodic=False )
    M_Grivis=collocation_matrix( knots, degree, Grivis, periodic=False )
    y_Grivis=func(Grivis)
    C1_Grivis=solverlinear(M_Grivis,y_Grivis)
    return C1_Grivis
    
    

def product_m(A_p,nbasis,phi_v,p):
    N_shape=nbasis-2
    result_pr=np.zeros(N_shape)
    for i in range(0,p+1):
        sum1_i=0.0
        for j in range(0,i+p+1):
            sum1_i  =sum1_i + A_p[i,j]*phi_v[j]
            
        result_pr[i]=sum1_i
    
    for i in range(p+1,nbasis-2-p):
        sum2_i=0.0
        for j in range(i-p,i+p+1):
            sum2_i  =sum2_i + A_p[i,j]*phi_v[j]
            
        result_pr[i]=sum2_i
        
    for i in range(nbasis-2-p,nbasis-2):
        sum3_i=0.0
        for j in range(i-p,nbasis-2):
            sum3_i  =sum3_i + A_p[i,j]*phi_v[j]
            
        result_pr[i]=sum3_i        
      
    return result_pr


def solver_1d_wave(AB,ABf,u_b,A_I,matrix_mass_f,v_wave_inter,w_wave_inter,k,M,nbasis,degree):
    
    PH_app=np.zeros((nbasis-2,M+2))
    phi_0=v_wave_inter
    phi_1=phi_0+k*w_wave_inter
    PH_app[:,0]=phi_0
    PH_app[:,1]=phi_1
    for n in range(1,M+1):
        #temp_sc=np.dot(AB,phi_1)-np.dot(A_I,phi_0)+u_b[n]*ABf-(u_b[n+1]+u_b[n-1])*matrix_mass_f
        temp_sc=product_m(AB,nbasis,phi_1,degree)-product_m(A_I,nbasis,phi_0,degree)+u_b[n]*ABf-(u_b[n+1]+u_b[n-1])*matrix_mass_f
        phi_next=solverlinear(A_I,temp_sc)
        PH_app[:,n+1]=phi_next
        phi_0=phi_1
        phi_1=phi_next
     
    PH_app=np.vstack((PH_app,u_b))
    PH_app=np.vstack((np.zeros(M+2),PH_app))    
    return PH_app


def solver_1d_wave_new(A_I,B_I,u_b,matrix_mass_f,matrix_stiffness_f,v_wave_inter,w_wave_inter,degree,M,k,nbasis):
    
    #approximation of the second derivative of u_b
    C_n=np.zeros((nbasis-2,M+2))
    C_0=-(     (1./(k**3))*(2*u_b[0]-5*u_b[1]+4*u_b[2]-u_b[3] )*matrix_mass_f  + u_b[0]*matrix_stiffness_f    )
    
    #at t=0
    C_n[:,0]=C_0
    
    #between t_1 and t_{M}
    for n in range(1,M+1):
        C_nplus=-(     (1./(k**2))*(u_b[n+1]-2*u_b[n]+u_b[n-1] )*matrix_mass_f  + u_b[n]*matrix_stiffness_f    )
        
        C_n[:,n]=C_nplus
        
    #at t=1 \T
    C_f=-(     (1./(k**3))*(2*u_b[M+1]-5*u_b[M]+4*u_b[M-1]-u_b[M-2] )*matrix_mass_f  + u_b[M+1]*matrix_stiffness_f    )
    C_n[:,M+1]=C_f
    
    
    ## stability conditions 
    betha=0.25
    ghama=0.5
    # all states
    PH_app=np.zeros((nbasis-2,M+2))
    
    ## state at t=0
    phi_0=v_wave_inter
    PH_app[:,0]=phi_0
    
    ##velocity at t=0
    phiV_0=w_wave_inter
    
    ##acceleration at t=0
    
    rhs_0=-product_m(B_I,nbasis,phi_0,degree)+C_n[:,0]
    phiA_0=solverlinear(A_I, rhs_0)
    
    ### loop to calculate state, velocity and acceleration at n+1
    AB_I=A_I+betha*(k**2)*B_I
    for n in range(1,M+2):
        phi_teld_n=phi_0+k*phiV_0+(k**2)*(0.5-betha)*phiA_0
        
        C_nplus1=C_n[:,n]
        
        rhs_0=-product_m(B_I,nbasis,phi_teld_n,degree)+C_nplus1
        
        phiA_n=phiA_0
        #new acceleration 
        phiA_0=solverlinear(AB_I, rhs_0)
        #new state
        phi_0=phi_teld_n+(k**2)*betha*phiA_0
        PH_app[:,n]=phi_0
        
        #new velocity
        phiV_0=phiV_0+k*(  (1-ghama)*phiA_n+ghama*phiA_0  )
        
    
    PH_app=np.vstack((np.zeros(M+2),PH_app)) 
    PH_app=np.vstack((PH_app,u_b))
    return PH_app
        
  
    

        
def solver_1d_laplac(A_I,matrix_mass_f,B_I,PSI_0,PSI_1,k,rhs_lapl,degree,nbasis):
    psi_diff_int=(1./k)*(PSI_1[1:-1]-PSI_0[1:-1])
    psi_diff_f=(1./k)*(PSI_1[-1]-PSI_0[-1])
      
    rhs_lapl_solve=product_m(A_I,nbasis,psi_diff_int,degree)+psi_diff_f*matrix_mass_f -rhs_lapl
    PH_lapl=solverlinear(B_I,rhs_lapl_solve)
    
    PH_lapl=np.append(PH_lapl,0)
    PH_lapl=np.append(0,PH_lapl)
    return PH_lapl

    
    
    
    
    
    
    
    
    
    
    
    
    