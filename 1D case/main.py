# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:37:07 2021

@author: nour
"""

import numpy as np
from function_bspline import *
from quadratures import gauss_legendre
from assembly import *
from solvers import *
from function_utils import *
from boundary_conditions import *
from initial_data import *
from function_use import *
from initial_data import *
from numpy import linalg as LA
from CONJUGATE_GRADIENT import *

#/////######## the parameters to be modified for the tests ##########################\\\\\\\\\\\

# degree of interpolation 
p=5

# the number of elements in [0,1]
ne=99
                                                                         #!!for the wave equation p=3,4 and ne=100 give us good results
#Maximum derivative of interest
nders=1

#spatial discretization
gridx=np.linspace(0,1,ne+1)
#print(gridx)
 
#### Time discretization

#CFL condition
CFL=0.1                        #CFL=k/h 
          #approximation of the second derivative by centered finite difference: CFL <= 0.5
          #approximation of the second derivative by the newmark-beta method: unconditionally stable          

# time interval [0,T]
T=2.2
# time step
k=CFL/ne
# the number of elements in [0,T] minus 1
M=int(T/k)-1


#//////######################################################################################\\\\\\\\\\
    
    

#/////###################        About B_splines functions ##################################"\\\\

# see function_bspline
knots=make_knots( gridx, p, periodic=False )
#print(knots)

# Number of b_splines
nbasis    = len(knots) - p - 1

# see function_bspline and note that breaks=gridx
breaks=breakpoints( knots, p )
#print(breaks)

#see function_bspline
spans=elements_spans( knots, p )
#print(spans)

#see quadratures and note that with this function we can approximate the integral of any function over [-1,1]
u, w = gauss_legendre( p )

#see function_bspline 
points, weights=quadrature_grid( breaks, u, w )
#print(points)

#see function bspline
span_x=find_span( knots, p, 0.1 )
#print(span_x)
#//////################################################################################\\\\\\\\\\
    
    
    
#/////############################  Assembly of  matrix_stiffness and matrix_mass ############\\\\\\\\\

#see function bspline
basis=basis_ders_on_quad_grid( knots, p, points, nders, normalize=False )

#the total mass matrix and stiffness matrix
matrix_mass=np.zeros((nbasis, nbasis ))
matrix_stiffness=np.zeros((nbasis, nbasis ))

#see assembly: we assembles the total matrices by  approximate the integrals with Gauss legend on each element
matrix_mass,matrix_stiffness=assemble_matrix(ne, p, spans, basis, weights, points, matrix_mass,matrix_stiffness)

#we need these two vectors in the case of a non-homogeneous wave equation
matrix_mass_f=matrix_mass[-1,:][1:-1]
matrix_stiffness_f=matrix_stiffness[-1,:][1:-1]

#//////################################################################################\\\\\\\\\\
    
    
'''
#//////////////////////###################### solve the  wave equation ####################\\\\\\\\\\\\\\\\\\\\

#interpolate the initial data: see initia_data and solvers 
v_wave_inter=intepol_functions(v_wave,knots, p)
w_wave_inter=intepol_functions(w_wave,knots, p)
v_wave_inter=v_wave_inter[1:-1]
w_wave_inter=w_wave_inter[1:-1]

#the boundary condition of our  wave equation at 1 see boundary_conditions
u_b=u_wave(np.linspace(0,T,M+2))

#solver of the non-homogeneous wave equation using newmark-beta method
sol_app_coff=solver_1d_wave_new(matrix_mass[1:-1, 1:-1],matrix_stiffness[1:-1, 1:-1],u_b,matrix_mass_f,matrix_stiffness_f,v_wave_inter,w_wave_inter,p,M,k,nbasis)

'''
'''
# solver of the non-homogeneous wave equation using centered finite difference method
AB=2*matrix_mass[1:-1, 1:-1]-(k**2)*matrix_stiffness[1:-1, 1:-1]
ABf=2*matrix_mass_f-(k**2)*matrix_stiffness_f
sol_app_coff=solver_1d_wave(AB,ABf,u_b,matrix_mass[1:-1, 1:-1],matrix_mass_f,v_wave_inter,w_wave_inter,k,M,nbasis,p)
'''

'''
#in the case of a homogeneous wave equation, when we interpolate the exact solution,
#we obtain an equation of time, the projection of the exact solution in this equation gives us testAB=0:
'''
'''
testAB=-(np.pi**2)*matrix_mass[1:-1, 1:-1]+matrix_stiffness[1:-1, 1:-1]
sin_interpol=intepol_functions(v_wave,knots, p)
print(LA.norm(np.dot(testAB,sin_interpol[1:-1]),2))
'''
'''
## plots
# instant of difference between exact and approximate solutions
ind_i=M+1

#see function_utils and function_use
plot_field_1d_diff(knots, p, sol_app_coff,ind_i,M,k, nx=401, color='b')

# see function_utils:  Erreur L^2 total between the exact and the approximate solutions
erreur_L2=norm_2_bsplin(ne,knots,points, weights,sol_app_coff,M,k)
print("Erreur total for the wave equation : %.6e"% erreur_L2)

#//////################################################################################\\\\\\\\\\\\\\\\\\\
    
    
    
#/////////################################# solve laplacian equation ####################\\\\\\\\\\\\\\\\\\\

#  see assembly :assemble the right-hand side of the Dirichlet problem
rhs_lapl=np.zeros(nbasis)
rhs_lapl=get_rhs_laplacian(y_0,ne, p, spans, basis, weights, points,rhs_lapl)

# PSI is an solution of a nonhomogeneous wave equation
PSI_0=np.zeros(nbasis)
PSI_1=np.zeros(nbasis)

#see solvers
sol_app_coff_lapl=solver_1d_laplac(matrix_mass[1:-1, 1:-1],matrix_mass_f,matrix_stiffness[1:-1, 1:-1],PSI_0,PSI_1,k,rhs_lapl[1:-1],p,nbasis)


#see function_utils and exact solution in function_use
plot_field_1d(knots, p, sol_app_coff_lapl, nx=401)

#see function_utils:  Erreur L^2 total between the exact and the approximate solutions for the poisson's equation
erreur_lapl=norm_2_exac_app(ne,knots,points, weights,sol_app_coff_lapl)
print("Erreur L^2 for the Dirichlet system : %.6e"%erreur_lapl)

#//////################################################################################\\\\\\\\\\\\\\\\\\\
    
    
    
#//////################### integrals with bspline ##################################################\\\\\\\\\\
v_wave_inter1=intepol_functions(v_wave,knots, p)

#see function_utils: 
#integral of two functions 
int_app=integralphi_psi(v_wave_inter1,v_wave_inter1,nbasis, matrix_mass,p)
#print(int_app)
#integral of two grad_functions
int_app_Delta=integral_DeltaPhi_psi(v_wave_inter1,v_wave_inter1,nbasis, matrix_stiffness,p)
#print(int_app_Delta)



#//////////##############################################################################################\\\\\\\\
    
'''   
    
#//////////################################# CONJUGATE GRADIENT ALGORITHM ###################################\\\\\\\\   
    
vecphi0_hat,vecphi1_hat=CG_alg(p,k,M,knots,matrix_mass,matrix_stiffness,ne, spans, basis, weights, points)
    
    



