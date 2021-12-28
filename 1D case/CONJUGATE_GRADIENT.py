# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 05:29:31 2021

@author: nour
"""

from solvers import *
from function_bspline import *
from function_utils import *
from initial_data import y_0,y_1
from assembly import get_rhs_laplacian


## initialization of CG
def ph0(x):
    return 0.*x

def ph1(x):
    return 0.*x


#normal derivative
def normal_derv(matphi,x,knots):
    degree = len(knots) - len(matphi) - 1
    d = matphi.shape[-1]
    
    span=find_span( knots, degree, x )
    b_der=basis_funs_1st_der( knots, degree, x, span )
    
    resnormal_at_x=np.zeros(d)
    
    for k in range(0, degree+1):
        resnormal_at_x[:] += b_der[k]*matphi[span-degree+k,:]
    ''' 
    for n in range(0,M+2):
        sum_span=0.0
        for k in range(0, degree+1):
            sum_span =sum_span+ b_der[k]*matphi[span-degree+k,n]
        resnormal_at_x[n]=sum_span
    '''
    return resnormal_at_x



def CG_alg(degree,k,M,knots,matrix_mass,matrix_stiffness,ne, spans, basis, weights, points):
    
    nbasis    = len(knots) - degree - 1
    
    matrix_mass_f=matrix_mass[-1,:][1:-1]
    matrix_stiffness_f=matrix_stiffness[-1,:][1:-1]
    
    A_I=matrix_mass[1:-1, 1:-1]
    B_I=matrix_stiffness[1:-1, 1:-1]
    
    
    ###Number of maximum iterations
    iterMAx=2000
    
    ### the stopping criteria
    eps=1e-10
    
    
    #initialization
    vecphi0=intepol_functions(ph0,knots, degree)
    vecphi1=intepol_functions(ph1,knots, degree)

  # # # #  #  iteration 0  # # # # # # # #####
  
    ###solve ph_0 system 
    matphi=solver_1d_wave_new(A_I,B_I,np.zeros(M+2),matrix_mass_f,matrix_stiffness_f,vecphi0[1:-1],vecphi1[1:-1],degree,M,k,nbasis)
    
    ###solve psi_0 system 
    
    #get normal at 1
    normalphi_1=normal_derv(matphi,1,knots)
    
    #solve the system with the change t = T-t
    matpsi_int=solver_1d_wave_new(A_I,B_I,normalphi_1[::-1],matrix_mass_f,matrix_stiffness_f,np.zeros(nbasis-2),np.zeros(nbasis-2),degree,M,k,nbasis)
   
    #get the solution of original system
    matpsi=matpsi_int[:,::-1] 
    
    ###solve laplacian system
    
    #get right-hand side
    rhs_lapl_n=np.zeros(nbasis)
    rhs_lapl_n=get_rhs_laplacian(y_1,ne, degree, spans, basis, weights, points,rhs_lapl_n)
    
    #solve the Dirichlet system
    vecphi0_tilde=solver_1d_laplac(A_I,matrix_mass_f,B_I,matpsi[:,0],matpsi[:,1],k,rhs_lapl_n[1:-1],degree,nbasis)
    
    #compte vteld
    vecphi1_tilde=intepol_functions(y_0,knots, degree)-matpsi[:,0]
    
    ### this term we will use it for the calculation of the stopping criteria
    cond_t2=np.sqrt(integral_DeltaPhi_psi(vecphi0_tilde,vecphi0_tilde,nbasis, matrix_stiffness,degree)+integralphi_psi(vecphi1_tilde,vecphi1_tilde,nbasis, matrix_mass,degree))
    
    ###stopping criteria at instant t=0
    t1=np.sqrt(integral_DeltaPhi_psi(vecphi0_tilde,vecphi0_tilde,nbasis, matrix_stiffness,degree)+integralphi_psi(vecphi1_tilde,vecphi1_tilde,nbasis, matrix_mass,degree))
    t2=np.sqrt(integral_DeltaPhi_psi(vecphi0,vecphi0,nbasis, matrix_stiffness,degree)+integralphi_psi(vecphi1,vecphi1,nbasis, matrix_mass,degree))
    if t2==0.:
        t2=1.
    testcondition=t1/t2
    
    vecphi0_check=vecphi0_tilde.copy()
    vecphi1_check=vecphi1_tilde.copy()
    
    ###  Descent
    itern=1
    while itern<=iterMAx and  testcondition > eps:
        print(itern)
        
        ###solve ph_0_check system 
        matphi_check=solver_1d_wave_new(A_I,B_I,np.zeros(M+2),matrix_mass_f,matrix_stiffness_f,vecphi0_check[1:-1],vecphi1_check[1:-1],degree,M,k,nbasis)
        
        ###solve psi_0_check system 
    
        #get normal at 1
        normalphi_1_check=normal_derv(matphi_check,1,knots)

        #solve the system with the change t = T-t
        matpsi_check_int=solver_1d_wave_new(A_I,B_I,normalphi_1_check[::-1],matrix_mass_f,matrix_stiffness_f,np.zeros(nbasis-2),np.zeros(nbasis-2),degree,M,k,nbasis)
        
        #get the solution of original system
        matpsi_check=matpsi_check_int[:,::-1] 
        
        ###solve laplacian system
        
        #solve the Dirichlet system
        vecphi0_line=solver_1d_laplac(A_I,matrix_mass_f,B_I,matpsi_check[:,0],matpsi_check[:,1],k,np.zeros(nbasis-2),degree,nbasis)
       
        #compte phi_line
        vecphi1_line=-matpsi_check[:,0]
        
        
        ### calcul of rho_n
        term1=integral_DeltaPhi_psi(vecphi0_tilde,vecphi0_tilde,nbasis, matrix_stiffness,degree)+integralphi_psi(vecphi1_tilde,vecphi1_tilde,nbasis, matrix_mass,degree)
        term2=integral_DeltaPhi_psi(vecphi0_line,vecphi0_check,nbasis, matrix_stiffness,degree)+integralphi_psi(vecphi1_line,vecphi1_check,nbasis, matrix_mass,degree)
        
        rho=term1/term2
        
        #  #  #  #  go to n+1  #  #  #  # 
        vecphi0=vecphi0-rho*vecphi0_check
        vecphi1=vecphi1-rho*vecphi1_check
        matphi=matphi-rho*matphi_check
        matpsi=matpsi-rho*matpsi_check
        
             # for the calcul of gamma 
        gamma_vecphi0_tilde=vecphi0_tilde.copy()         #
        gamma_vecphi1_tilde=vecphi1_tilde.copy()
    
        vecphi0_tilde=vecphi0_tilde-rho*vecphi0_line
        vecphi1_tilde=vecphi1_tilde-rho*vecphi1_line
        
        ###stopping criteria
        t1=np.sqrt(integral_DeltaPhi_psi(vecphi0_tilde,vecphi0_tilde,nbasis, matrix_stiffness,degree)+integralphi_psi(vecphi1_tilde,vecphi1_tilde,nbasis, matrix_mass,degree))
        print(testcondition)
        testcondition=t1/cond_t2
        
        
        
        
        ### new descent direction
        ter1=integral_DeltaPhi_psi(vecphi0_tilde,vecphi0_tilde,nbasis, matrix_stiffness,degree)+integralphi_psi(vecphi1_tilde,vecphi1_tilde,nbasis, matrix_mass,degree)
        ter2=integral_DeltaPhi_psi(gamma_vecphi0_tilde,gamma_vecphi0_tilde,nbasis, matrix_stiffness,degree)+integralphi_psi(gamma_vecphi1_tilde,gamma_vecphi1_tilde,nbasis, matrix_mass,degree)
        gamma=ter1/ter2
    

        vecphi0_check=vecphi0_tilde+gamma*vecphi0_check
        vecphi1_check=vecphi1_tilde+gamma*vecphi1_check
        
         #  go to n+1
        
        itern=itern+1
        
    
        
    return vecphi0,vecphi1