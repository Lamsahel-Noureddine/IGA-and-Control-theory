# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 04:44:31 2021

@author: nour
"""
from initial_data import *

def assemble_matrix(nelements, degree, spans, basis, weights, points, matrix1,matrix2):

    # ... sizes
    ne1       = nelements
    p1        = degree
    spans_1   = spans
    basis_1   = basis
    weights_1 = weights
    points_1  = points
    
    k1 = weights.shape[1]
    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for il_1 in range(0, p1+1):
            for jl_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                j1 = i_span_1 - p1 + jl_1

                v1 = 0.0
                v2=0.0
                for g1 in range(0, k1):
                    bi_0 = basis_1[ie1, il_1, 0, g1]
                    bi_x = basis_1[ie1, il_1, 1, g1]                    

                    bj_0 = basis_1[ie1, jl_1, 0, g1]
                    bj_x = basis_1[ie1, jl_1, 1, g1]                    

                    wvol1 = weights_1[ie1, g1]
                

                    v1 += (bi_0 * bj_0) * wvol1
                    v2 += (bi_x * bj_x) * wvol1
                    

                matrix1[i1, j1]  += v1
                matrix2[i1, j1]  += v2
    # ...

    return matrix1,matrix2 


def get_rhs_laplacian(y_1,nelements, degree, spans, basis, weights, points,rhs_lapl):

    # ... sizes
    ne1       = nelements
    p1        = degree
    spans_1   = spans
    basis_1   = basis
    weights_1 = weights
    points_1  = points
    
    k1 = weights.shape[1]
    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for il_1 in range(0, p1+1):
            i1 = i_span_1 - p1 + il_1
            v1 = 0.0
            for g1 in range(0, k1):
                bi_0 = basis_1[ie1, il_1, 0, g1]
                wvol1 = weights_1[ie1, g1]
                x1    = points_1[ie1, g1]
            
                v1 += (bi_0 * y_1(x1)) * wvol1
                
                
            rhs_lapl[i1]+=   v1

    # ...

    return  rhs_lapl



