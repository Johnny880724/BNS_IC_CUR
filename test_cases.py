# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:26:27 2020

@author: Johnny Tsao

This file contains 5 test cases for the Newtonian equation
"""
import mesh_helper_functions as mhf
import numpy as np
import matplotlib.pyplot as plt

#################### Newtonian equation ##########################
# coefficient Poisson equation:   div ( rho * grad(u)) = f * |grad(rho)|
# Neumann boundary condition:     u_n = f
# Level set:                      phi = rho + dh 
# (the level set can be any function satisfying phi = 0 + dh on the boundary)

class test_cases:
    # desired_func:    u
    # desired_func_n:  u_n
    # rhs_func:        f * |grad(rho)|
    # lvl_func:        phi
    # rho_func:        rho
    # boundary_func:   f (on phi = 0, u_n = f)
    
    # sol_func:        u (for phi < 0)
    # theory_func:     u (for rho < 0)
        
    ########################## index ###########################
    # circular symmetric test cases (r = np.sqrt(x**2 + y**2))
        # test case 1  :     u = r**2     phi = r-r0
        # test case 2  :     u = r**4     phi = r-r0
        # test case 3  :     u = r**6     phi = r-r0
        # test case 4  :     u = r**4     phi = (r**2 - r0**2)/(2*r0)
    
    # deformed test case (r = np.sqrt(k**2 * x**2 + y**2))
        # test case 5  :     u = r**4     phi = r-r0
    ############################################################
        
    
    #### test case 1 ##########################################
    # r               = np.sqrt(x**2 + y**2)
    # u               = r**2
    # u_n             = 2*r
    # phi             = r - r0
    # abs_grad(phi)   = 1
    # f               = 6*r - 4*r0
    # rhs             = 6*r - 4*r0
    def setup_equations_1(self, delta_phi):
        # theoretical result u
        def desired_result(x, y):
            return x**2 + y**2
        self.desired_func = desired_result
        
        # theoretical derivative u_n
        def desired_result_n(x,y):
            return 2*mhf.XYtoR(x,y)
        self.desired_func_n = desired_result_n
        
        #the radius of the interior circle
        r0 = 0.8
        
        # right hand side function
        def rhs(x,y):
            return 6*mhf.XYtoR(x,y) - 4*r0
        self.rhs_func = rhs
        
        # level set function
        def level(x,y):
            return coef(x,y) + delta_phi
        self.lvl_func = level
        
        # coefficient (rho) of the Poisson equation
        def coef(x,y):
            return np.array(mhf.XYtoR(x,y) - r0)
        self.rho_func = coef
        
        # boundary condition
        def Neumann_boundary(x,y):
            return rhs(x,y)
        self.boundary_func = Neumann_boundary
        
        #set up solution for checking error
        self.setup_solution()
    
    #### test case 2 ##########################################
    # r               = np.sqrt(x**2 + y**2)
    # u               = r**4
    # u_n             = 4 * r**3
    # phi             = r - r0
    # abs_grad(phi)   = 1
    # f               = 4 * r**2 * (-4*r0 + 5*r)
    # rhs             = 4 * r**2 * (-4*r0 + 5*r)
    def setup_equations_2(self, delta_phi):
        # theoretical result u
        def desired_result(x, y):
            return (x**2 + y**2)**2
        self.desired_func = desired_result
        
        # theoretical derivative u_n
        def desired_result_n(x, y):
            return 4*mhf.XYtoR(x,y)**3
        self.desired_func_n = desired_result_n
        
        #the radius of the interior circle
        r0 = 0.8
        
        # right hand side function
        def rhs(x,y):
            return 4*mhf.XYtoR(x,y)**2*(- 4*r0 + 5*mhf.XYtoR(x,y))
        self.rhs_func = rhs
        
        # level set function
        def level(x,y):
            return coef(x,y) + delta_phi
        self.lvl_func = level
        
        # coefficient (rho) of the Poisson equation
        def coef(x,y):
            return np.array(mhf.XYtoR(x,y) - r0)
        self.rho_func = coef
        
        # boundary condition
        def Neumann_boundary(x,y):
            return rhs(x,y)
        self.boundary_func = Neumann_boundary
        
        #set up solution for checking error
        self.setup_solution()
        
    #### test case 3 ##########################################
    # r               = np.sqrt(x**2 + y**2)
    # u               = r**6
    # u_n             = 6 * r**5
    # phi             = r - r0
    # abs_grad(phi)   = 1
    # f               = 6 * r**4 * (-6*r0 + 7*r)
    # rhs             = 6 * r**4 * (-6*r0 + 7*r)
    def setup_equations_3(self,delta_phi):
        # theoretical result u
        def desired_result(x, y):
            return (x**2 + y**2)**3
        self.desired_func = desired_result
        
        # theoretical derivative u_n
        def desired_result_n(x, y):
            r = mhf.XYtoR(x,y)
            return 6*r**5
        self.desired_func_n = desired_result_n
        
        #the radius of the interior circle
        r0 = 0.8
        
        # right hand side function
        def rhs(x,y):
            r = mhf.XYtoR(x,y)
            return 6*r**4*(-6*r0 + 7*r)
        self.rhs_func = rhs
        
        # level set function
        def level(x,y):
            return coef(x,y) + delta_phi
        self.lvl_func = level
        
        # coefficient (rho) of the Poisson equation
        def coef(x,y):
            return np.array(mhf.XYtoR(x,y) - r0)
        self.rho_func = coef
        
        # boundary condition
        def Neumann_boundary(x,y):
            return rhs(x,y)
        self.boundary_func = Neumann_boundary
        
        #set up solution for checking error
        self.setup_solution()
    
    #### test case 4 ##########################################
    # r               = np.sqrt(x**2 + y**2)
    # u               = r**4
    # u_n             = 4 * r**3
    # phi             = (r**2 - r0**2)/(2*r0)
    # abs_grad(phi)   = r / r0
    # f               = 8*r**2*(-2*r0**2 + 3*r**2) / (2*r0) / (r/r0)
    # rhs             = 8*r**2*(-2*r0**2 + 3*r**2) / (2*r0) (assume r = r0 on the boundary)
    def setup_equations_4(self,delta_phi):
        # theoretical result u
        def desired_result(x, y):
            return (x**2 + y**2)**2
        self.desired_func = desired_result
        
        # theoretical derivative u_n
        def desired_result_n(x, y):
            r = mhf.XYtoR(x,y)
            return 4*r**3
        self.desired_func_n = desired_result_n
        
        #the radius of the interior circle
        r0 = 0.8
        
        # right hand side function
        def rhs(x,y):
            r = mhf.XYtoR(x,y)
            return 8*r**2*(-2*r0**2 + 3*r**2) / (2*r0)
        self.rhs_func = rhs
        
        # level set function
        def level(x,y):
#            return coef(x,y) + delta_phi
            return coef(x,y) + delta_phi
        self.lvl_func = level
        
        # coefficient (rho) of the Poisson equation
        def coef(x,y):
            return (mhf.XYtoR(x,y)**2 - r0**2) / (2*r0)
        self.rho_func = coef
        
        # inverse of abs_grad(phi) 1/|grad(phi)|
        def inv_abs_grad_phi(x,y):
            return r0/(np.sqrt(y**2 + x**2) + mhf.singular_null)
        
        # boundary condition
        def Neumann_boundary(x,y):
            return rhs(x,y) * inv_abs_grad_phi(x,y)
        self.boundary_func = Neumann_boundary
        
        #set up solution for checking error
        self.setup_solution()
        
    
    #### test case 5 ##########################################
    # r               = np.sqrt(k**2 * x**2 + y**2)
    # rp              = np.sqrt(k**4 * x**2 + y**2)
    # u               = r**4
    # u_n             = 4 * rp * r**2
    # phi             = r - r0
    # abs_grad(phi)   = rp / r
    # f               = (4*(r-r0)*(y**2*(3 + k**2) + x**2*k**2*(1 + 3*k**2)) + 4*r*rp**2) * r / rp
    # rhs             = 4*(r-r0)*(y**2*(3 + k**2) + x**2*k**2*(1 + 3*k**2)) + 4*r*rp**2
    def setup_equations_5(self, delta_phi, k=1.0):
        # theoretical result u
        def desired_result(x, y):
            return (k**2*x**2 + y**2)**2
        self.desired_func = desired_result
        
        # theoretical derivative u_n
        def desired_result_n(x, y):
            return 4*np.sqrt(x**2*k**4 + y**2)*(x**2*k**2 + y**2)
        self.desired_func_n = desired_result_n
        
        #the half major axis of the interior ellipse
        r0 = 0.8
        
        # right hand side function
        def rhs(x,y):
            r = np.sqrt(y**2 + x**2*k**2)
            return 4*(r-r0)*(y**2*(3 + k**2) + x**2*k**2*(1 + 3*k**2)) + 4*r*(y**2+x**2*k**4)
        self.rhs_func = rhs
        
        # level set function
        def level(x,y):
            return coef(x,y) + delta_phi * k
        self.lvl_func = level
        
        # coefficient (rho) of the Poisson equation
        def coef(x,y):
            return (np.sqrt(y**2 + x**2*k**2) - r0)
        self.rho_func = coef
        
        # inverse of abs_grad(phi) 1/|grad(phi)|
        def inv_abs_grad_phi(x,y):
            return np.sqrt(y**2 + x**2*k**2)/(np.sqrt(y**2+ x**2*k**4)+mhf.singular_null)
        
        # boundary condition
        def Neumann_boundary(x,y):
            return rhs(x,y) * inv_abs_grad_phi(x,y)
#            return (4*(r-r0)*(y**2*(3 + k**2) + x**2*k**2*(1 + 3*k**2))*r/(rp+mhf.singular_null) + 4*r**2*rp)
        self.boundary_func = Neumann_boundary
        
        #set up solution for checking error
        self.setup_solution()
        
    # set up solution functions for checking error
    def setup_solution(self):
        # the solution u for phi < 0
        def solution(x,y):
            isIn = mhf.get_frame(self.lvl_func(x,y))
            sol = isIn*(self.desired_func(x,y))
            sol_zero = sol - np.sum(sol) / np.sum(isIn) *isIn
            return sol_zero
        self.sol_func = solution
        
        # the solution u for rho < 0 (requires extrapolation)
        def theory(x,y):
            isIn = mhf.get_frame(self.rho_func(x,y))
            sol = isIn*(self.desired_func(x,y))
            sol_zero = sol - np.sum(sol) / np.sum(isIn) *isIn
            return sol_zero
        self.theory_func = theory
        