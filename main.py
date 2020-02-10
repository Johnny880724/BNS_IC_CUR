# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 00:01:43 2020

@author: Johnny Tsao

file for test run
"""
import mesh_helper_functions as mhf
import test_cases as tc
import jacobi_newtonian_solver as jns
import numpy as np
import matplotlib.pyplot as plt

if(__name__ == "__main__"):
    plt.close("all")
    
    ## iteration number parameters
    # maximum iteration number for the source term method
    maxIter = 200
    # the total iteration number N for Jacobi solver = it_multi * N_grid**2
    it_multi = 10
    

    result_array = []
    
    # the array of grid sizes to be tested
#    grid_size_array = [100]
    grid_size_array = [32,50,64,80,102,126,140]
    for grid_size in grid_size_array:
        u_init, (xmesh,ymesh), h = jns.setup_grid(grid_size)
        print("grid size %d" % grid_size)
        
        # 1. create a test case object
        test_case = tc.test_cases()
        
        # 2. set up required parameters using functions from the test case
        test_case.setup_equations_5(max(0.01*grid_size*h,h))
        phi = test_case.lvl_func(xmesh, ymesh)
        sol = test_case.sol_func(xmesh, ymesh)
        rho = test_case.rho_func(xmesh, ymesh)
        rhs = test_case.rhs_func(xmesh, ymesh)
        boundary = test_case.boundary_func(xmesh, ymesh)
#        plt.matshow((boundary - test_case.desired_func_n(xmesh,ymesh)) * get_N1(phi))
        
        # 3. run the coefficient poisson solver
        u_result = jns.coef_poisson_jacobi_source_term_Neumann_hVary(u_init, it_multi, (xmesh,ymesh), phi, rhs, rho, sol, boundary, maxIter)
        
        # 4. calculate the error with theoretical value
        theory = test_case.theory_func(xmesh, ymesh)
        maxDif,L2Dif = mhf.get_error_N(u_result, theory , mhf.get_frame(rho))
        result_array.append(maxDif)

    # for plotting
    plt.figure()
    plt.plot(grid_size_array, result_array)
