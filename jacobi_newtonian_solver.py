# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:09:48 2020

@author: Johnny Tsao
"""
import sys
import mesh_helper_functions as mhf
import test_cases as tc
import numpy as np
import matplotlib.pyplot as plt

#adding to denominator to avoid 0/0 error
singular_null = mhf.singular_null

def setup_grid(N_grid_val = 100):
    # grid dimension
    grid_min = -1.
    grid_max = 1.
    
    global N_grid
    N_grid = N_grid_val
    # grid spacing
    global h
    h = (grid_max - grid_min) / (N_grid) 
    
    # define arrays to hold the x and y coordinates
    xy = np.linspace(grid_min,grid_max,N_grid + 1)
    global xmesh, ymesh
    xmesh, ymesh = np.meshgrid(xy,xy)
    
    # solution
    global u_init
    u_init = np.zeros_like(xmesh)
    return u_init, (xmesh,ymesh), h

#discretization of Step, Delta functions
def I(phi):
    return mhf.D(phi)*phi

def J(phi):
    return 1./2 * mhf.D(phi)*phi**2

def K(phi):
    return 1./6 * mhf.D(phi)*phi**3

#Characteristic function of N1
##Chi is 1 if in N1
##       0 if not in N1
def Chi(phi):
    ret = np.zeros_like(phi)
    ret[:-1,:] += mhf.D(-phi[:-1,:]*phi[ 1:,:])
    ret[ 1:,:] += mhf.D(-phi[:-1,:]*phi[ 1:,:])
    ret[:,:-1] += mhf.D(-phi[:,:-1]*phi[ :,1:])
    ret[:, 1:] += mhf.D(-phi[:,:-1]*phi[ :,1:])
    return np.heaviside(ret,0)

# return the neighbor and the domain
def get_neighbor(domain):
    ret = np.zeros_like(domain)
    ret += domain
    ret[ 1:, :] += domain[:-1, :]
    ret[:-1, :] += domain[ 1:, :]
    ret[ :, 1:] += domain[ :,:-1]
    ret[ :,:-1] += domain[ :, 1:]
    return np.heaviside(ret,0)

# return N1
def get_N1(phi):
    return Chi(phi)

# return N2
def get_N2(phi):
    N1 = get_N1(phi)    
    return get_neighbor(N1)

#return N3
def get_N3(phi):
    N2 = get_N2(phi)
    return get_neighbor(N2)

# return the heaviside function discretized in the paper
def H(phi_inv,h):
    J_mat = J(phi_inv)
    K_mat = K(phi_inv)
    first_term_1 = mhf.laplace(J_mat,h,h) / (mhf.abs_grad(phi_inv,h,h)**2 + singular_null)
    first_term_2 = -(mhf.laplace(K_mat,h,h) - J_mat*mhf.laplace(phi_inv,h,h))*mhf.laplace(phi_inv,h,h) / (mhf.abs_grad(phi_inv,h,h)**4 + singular_null)
    first_term = first_term_1 + first_term_2
    second_term = mhf.D(phi_inv)
    return Chi(phi_inv) * first_term + (1-Chi(phi_inv)) * second_term

# return the delta function discretized in the paper
def delta(phi_inv,h):
    I_mat = I(phi_inv)
    J_mat = J(phi_inv)
    first_term = mhf.laplace(I_mat,h,h) / (mhf.abs_grad(phi_inv,h,h)**2 + singular_null)
    first_term -= (mhf.laplace(J_mat,h,h) - I_mat*mhf.laplace(phi_inv,h,h))*mhf.laplace(phi_inv,h,h) / (mhf.abs_grad(phi_inv,h,h)**4 + singular_null)
    return Chi(phi_inv) * first_term
    
# return the source term discretized in the paper
def get_source(a, b, phi_,f_mat_, h_):
    #in the soruce term paper, the phi they use are inverted
    phi_inv = - phi_
    
    #Discretization of the source term - formula (7)
    H_h_mat = H(phi_inv,h_)
    H_mat = mhf.D(phi_inv)
    term1 = mhf.laplace(a * H_mat,h_,h_)
    term2 = - H_h_mat * mhf.laplace(a, h_, h_)
    term3 = - (b - mhf.grad_n(a,phi_,h_,h_)) * delta(phi_inv, h_) * mhf.abs_grad(phi_inv,h_,h_)
    term4 = H_h_mat * f_mat_
    S_mat = term1 + term2 + term3 + term4
    return S_mat

#projection algorithm
def projection(mesh_, phi_inv):
    xmesh, ymesh = mesh_
    h = xmesh[0,1]-xmesh[0,0]
    phi_abs_grad = mhf.abs_grad(phi_inv,h,h)
    grad_tup = mhf.grad(phi_inv,h,h)
    nx = -grad_tup[0] / (phi_abs_grad + singular_null)
    ny = -grad_tup[1] / (phi_abs_grad + singular_null)
    xp = xmesh + nx * phi_inv / (phi_abs_grad + singular_null)
    yp = ymesh + ny * phi_inv / (phi_abs_grad + singular_null)
    return xp, yp

# quadrature extrapolation algorithm
# (extrapolation may not work if the grid size is too small)
def extrapolation(val_, target_, eligible_):
    val_extpl = val_ * eligible_
    tau_0 = np.copy(target_)
    eps_0 = np.copy(eligible_)
    tau = np.copy(tau_0)
    eps = np.copy(eps_0)
    tau_cur = np.copy(tau)
    eps_cur = np.copy(eps)
    while(np.sum(tau) > 0):
        val_extpl_temp = np.copy(val_extpl)
        for i in range(len(val_)):
            for j in range(len(val_[i])):
                if(tau[i,j] == 1):
                    triplet_count = 0
                    triplet_sum = 0
                    #2.9 is used to check if every element in the length-3 array is 1
                    if(np.sum(eps[i+1:i+4,j]) > 2.9):
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i+1,j] - 3*val_extpl[i+2,j] + val_extpl[i+3,j]
                    if(np.sum(eps[i-3:i,j]) > 2.9):
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i-1,j] - 3*val_extpl[i-2,j] + val_extpl[i-3,j]
                    if(np.sum(eps[i,j+1:j+4]) > 2.9):
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i,j+1] - 3*val_extpl[i,j+2] + val_extpl[i,j+3]
                    if(np.sum(eps[i,j-3:j]) > 2.9):
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i,j-1] - 3*val_extpl[i,j-2] + val_extpl[i,j-3]
                        
                    if(triplet_count > 0):
                        val_extpl_temp[i,j] = triplet_sum / triplet_count
                        tau_cur[i,j] = 0
                        eps_cur[i,j] = 1
                        
        tau = np.copy(tau_cur)
        eps = np.copy(eps_cur)
        val_extpl = np.copy(val_extpl_temp)
        
    return val_extpl

# Interpolation method for 2d grid using interpolate from scipy
# mesh (xmesh, ymesh) must be equal-distanced mesh grid
from scipy import interpolate
def interpolation(mesh, mesh_p, zmesh):
    xmesh, ymesh = mesh
    xmesh_p, ymesh_p = mesh_p
    x = xmesh[0, :]
    y = ymesh[:, 0]
    f = interpolate.interp2d(x,y,zmesh, kind = 'cubic')
    zmesh_p = np.zeros_like(zmesh)
    for i in range(len(xmesh)):
        for j in range(len(xmesh[i])):
            zmesh_p[i,j] = f(xmesh_p[i,j],ymesh_p[i,j])[0]
    return zmesh_p

## poisson solver function
## the result solution is zero (making the mean of the solution 0) at every iteration
# u_init_          : (N*N np array) initial data
# maxIterNum_      : (scalar)       maximum iteration for Jacobi method
# mesh_            : (duple)        (xmesh, ymesh)
# phi_             : (N*N np array) level set
# source_          : (N*N np array) right hand side 
# print_option     : (bool)         switch to print the iteration progress
def poisson_jacobi_solver_zero(u_init_, maxIterNum_, mesh_, source_, phi_,print_option = True):
    u_prev = np.copy(u_init_)
    u      = np.copy(u_init_)
    isIn   = mhf.get_frame(phi_)
    numIn  = np.sum(isIn)
    for i in range(maxIterNum_):
        # enforce boundary condition
        u[ 0, :] = np.zeros_like(u[ 0, :])
        u[-1, :] = np.zeros_like(u[-1, :])
        u[ :, 0] = np.zeros_like(u[ :, 0])
        u[ :,-1] = np.zeros_like(u[ :,-1])
    
        u_new = np.copy(u)
    
        # update u according to Jacobi method formula
        # https://en.wikipedia.org/wiki/Jacobi_method
        
        del_u = u[1:-1,2:] + u[1:-1,0:-2] + u[2:,1:-1] + u[0:-2,1:-1]
        u_new[1:-1,1:-1] = -h**2/4 * (source_[1:-1,1:-1] - del_u/h**2)
        u = u_new
        
        # for Neumann condition: normalize the inside to mean = 0
        u -= (np.sum(u*isIn) / numIn)*isIn
        
        # check convergence and print process
        check_convergence_rate = 10**-5
        
        if(i % int(maxIterNum_*0.1) < 0.1):
            u_cur = np.copy(u)
            L2Dif = mhf.L_n_norm(np.abs(u_cur - u_prev)) / mhf.L_n_norm(u_cur)
            
            if(L2Dif < check_convergence_rate):
                break;
            else:
                u_prev = np.copy(u_cur)
            if(print_option):
                sys.stdout.write("\rProgress: %4d out of %4d" % (i,maxIterNum_))
                sys.stdout.flush()
    if(print_option):
        print("")
    
    
    return u

## main coefficient poisson solver function
# u_init_          : (N*N np array) initial data
# maxMultiple_     : (scalar)       maximum iteration multiple for Jacobi method
# mesh_            : (duple)        (xmesh, ymesh)
# phi_             : (N*N np array) level set
# rhs_             : (N*N np array) right hand side 
# rho_             : (N*N np array) density
# sol_             : (N*N np array) theoretical solution
# boundary_        : (N*N np array) Neumann boundary condition
# iteration_total  : (scalar)       maximum iteration for source term method
def coef_poisson_jacobi_source_term_Neumann_hVary(u_init_, maxMultiple_, mesh_,phi_,rhs_,rho_,\
                                       sol_, boundary_, iteration_total):
    
    # making copies of the variables
    phi = np.copy(phi_)
    rho = np.copy(rho_)
    u_cur_result = np.copy(u_init_)
    phi_inv = -phi
    
    #mesh variables
    xmesh, ymesh = mesh_
    h = xmesh[0,1] - xmesh[0,0]
    N = len(xmesh)
    
    # Level variables
    N1 = get_N1(phi)
    N2 = get_N2(phi)
    Omega_m = mhf.D(-phi)
    isOut = mhf.D(phi)
    isIn = mhf.get_frame(phi)
    
    #1. Extend g(x,y) off of Gamma, define b throughout N2
    xmesh_p, ymesh_p = projection((xmesh,ymesh), phi_inv)
    g_ext = interpolation((xmesh, ymesh), (xmesh_p, ymesh_p), boundary_)
    b_mesh = - g_ext * N2
    
    #2. extrapolate f throughout N1 U Omega^+
    f_org = np.copy(rhs_)
    eligible_0 = Omega_m * (1-N1)
    target_0 = N1 * (1 - eligible_0)
    f_extpl = extrapolation(f_org, target_0, eligible_0)
    
    #3. initialize a based on initial u throughout N2
    u_extpl = extrapolation(u_cur_result, target_0, eligible_0)  
    a_mesh = - np.copy(u_extpl)    
    
    #4. Find the source term for coefficient
    ux, uy = mhf.grad(u_cur_result, h, h)
    ux_extpl = extrapolation(ux, target_0, eligible_0)
    uy_extpl = extrapolation(uy, target_0, eligible_0)
    rhox, rhoy = mhf.grad(rho,h,h)
    extra = rhox * ux_extpl + rhoy * uy_extpl
    f_use = (f_extpl - extra) / (rho + singular_null)
    
    # termination array
    Q_array = np.zeros(iteration_total)
    
    # parameters for the iteration
    maxIterNum = maxMultiple_ * N**2
    print_it = True
    
    
    for it in range(iteration_total):
        # print iteration process
        if(print_it):
            print("This is iteration %d :" % (it + 1))
        
        #A1-1 compute the source term
        source = get_source(-a_mesh, -b_mesh, phi, f_use, h)
        
        #A1-2 compute the source term with the addition of convergence term
        q = -0.75 * min(1, it*0.1)
        source += (q / h * u_cur_result) * (1-Omega_m) * N2
        
        #A2 call a Poisson solver resulting in u throughout Omega
        u_result = poisson_jacobi_solver_zero(u_cur_result, maxIterNum, (xmesh,ymesh), source, phi,print_it)
        maxDif,L2Dif = mhf.get_error(u_result, (xmesh, ymesh), isIn, sol_, print_it)
        change = np.abs(u_result - u_cur_result)
        maxChange = np.max(change * isIn)
        u_cur_result = np.copy(u_result)
    
        #A3-1 Extrapolate u throughout N2
        eligible_0 = Omega_m * (1-N1)
        target_0 = N2 * (1-eligible_0)
        u_extpl = extrapolation(u_result, target_0, eligible_0)
        
        #A3-2 compute the new a throughout N2
        a_mesh = - np.copy(u_extpl)
        
        #A3-3 compute the new source term f_use
        ux, uy = mhf.grad(u_cur_result, h, h)
        ux_extpl = extrapolation(ux, target_0, eligible_0)
        uy_extpl = extrapolation(uy, target_0, eligible_0)
        rhox, rhoy = mhf.grad(rho,h,h)
        extra = rhox * ux_extpl + rhoy * uy_extpl
        f_use = (f_extpl - extra) / (rho + singular_null)
        
        #A4 check for termination
        Q_array[it] = np.max(u_result * isOut * N2)
        
        if(it > 5):
            hard_conergence_rate = 10**-2
            hard_convergence = maxChange < hard_conergence_rate
            if(hard_convergence):
#                maxDif,L2Dif = mhf.get_error(u_result, (xmesh, ymesh), isIn, sol_)
                break
        
    u_result_org = np.copy(u_result)
    
    # Quadruple lagrange extrapolation to the full grid
    isIn_full = mhf.get_frame(rho)
    eligible_0 = Omega_m * (1-N1)
    target_0 = isIn_full * (1-eligible_0)  
    u_extpl_lagrange = extrapolation(u_result_org, target_0, eligible_0)
    
    return u_extpl_lagrange

if(__name__ == "__main__"):
    plt.close("all")
    
    ## iteration number parameters
    # maximum iteration number for the source term method
    maxIter = 200
    # the total iteration number N for Jacobi solver = it_multi * N_grid**2
    it_multi = 10
    

    extpl_array = []
    
    # the array of grid sizes to be tested
    grid_size_array = [64]
#    grid_size_array = [32,50,64,80,100,128,140,156,178]
    for grid_size in grid_size_array:
        setup_grid(grid_size)
        print("grid size %d" % grid_size)
        
        # 1. create a test case object
        test_case = tc.test_cases()
        
        # 2. set up required parameters using functions from the test case
        test_case.setup_equations_1(max(0.01*grid_size*h,h))
        phi = test_case.lvl_func(xmesh, ymesh)
        sol = test_case.sol_func(xmesh, ymesh)
        rho = test_case.rho_func(xmesh, ymesh)
        rhs = test_case.rhs_func(xmesh, ymesh)
        boundary = test_case.boundary_func(xmesh, ymesh)
#        plt.matshow((boundary - test_case.desired_func_n(xmesh,ymesh)) * get_N1(phi))
        
        # 3. run the coefficient poisson solver
        u_extpl_result = coef_poisson_jacobi_source_term_Neumann_hVary(u_init, it_multi, (xmesh,ymesh), phi, rhs, rho, sol, boundary, maxIter)
        
        # 4. calculate the error with theoretical value
        theory = test_case.theory_func(xmesh, ymesh)
        maxDif,L2Dif = mhf.get_error_N(u_extpl_result, theory , mhf.get_frame(rho))
        extpl_array.append(maxDif)

        
    plt.figure()
    plt.plot(grid_size_array, extpl_array)
