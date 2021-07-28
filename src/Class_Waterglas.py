#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:17:09 2020

@author: zech0001
"""

import numpy as np
import copy

DEF_DATA = dict(
        domain={'x_0':0,'x_L':50,'z_0':0,'z_L':10,'z_1':1,'z_2':2,'nx':50,'nz':50},
        k_sand = 25., #m/d
        k_til_0 = 8.64e-3, #m/d = 1e-7m/s
        gradient = -3.8e-4, # m/m # hydraulic gradient in aquifer   
        head_abs = 0.287,
        n_modes = 10,
        dim_less=True,
        c1_total = 0.92,
        c1_til = 0.94,
        )

class WaterGlassTransport():
    
    def __init__(self,**settings):

        self.settings=copy.copy(DEF_DATA) 
        self.settings.update(settings)   
            
        self.set_coordinates()
         
        self.kf=self.settings['k_sand']
        self.kt0=self.settings['k_til_0']
        
        self.dh=self.settings['gradient']*self.L # head difference between left and right boundary

    def set_coordinates(self):

        ### dimensional coordinates
        self.x = np.linspace(self.settings['domain']['x_0'],self.settings['domain']['x_L'],self.settings['domain']['nx']+1,endpoint=True)
        self.z = np.linspace(self.settings['domain']['z_0'],self.settings['domain']['z_L'],self.settings['domain']['nz']+1,endpoint=True)
        self.xx,self.zz=np.meshgrid(self.x,self.z)

        ### dimensionless coordinates
        self.xL = np.linspace(0,1,self.settings['domain']['nx']+1,endpoint=True)
        self.zT = np.linspace(0,1,self.settings['domain']['nz']+1,endpoint=True)
        self.xxl,self.zzl=np.meshgrid(self.xL,self.zT)

        ### length of domains and dimensionless ratios
        self.L = self.settings['domain']['x_L']-self.settings['domain']['x_0'] # length of domain
        self.T = self.settings['domain']['z_L']-self.settings['domain']['z_0'] # total depth of domain
        self.H = self.settings['domain']['z_1']-self.settings['domain']['z_0'] # depth of injection layer
        self.D = self.settings['domain']['z_2']-self.settings['domain']['z_1'] # thickness of injection layer

        self.ratio_TL = self.T/self.L
        self.ratio_HL = self.H/self.L
        self.ratio_HT = self.H/self.T
       
    def head_homogeneous(self,head_abs = False,**settings):
        
        """ head solution in x and y as sum over n modes """

        self.settings.update(settings)   

#        xxl,nn,zzl=np.meshgrid(self.xL,np.arange(1,self.settings['n_modes']+1,1),self.zT)
#        
#        an=1./(2.*nn-1)**2              ### fourier coefficients from initial condition 
#
#        cosnx = np.cos((2.*nn-1)*np.pi*xxl)        ###  solution in x-direction
#        coshnz = np.cosh((2.*nn-1)*np.pi*(zzl-1.)*self.ratio_TL)
#        coshnlz = np.cosh((2.*nn-1)*np.pi*self.ratio_TL)   
#        self.head=(4./(np.pi*np.pi) * np.sum(an*cosnx*coshnz/coshnlz,axis=0)).T ### sum over modes n

        self.head=4./(np.pi*np.pi) * head(self.xL,self.zT,self.ratio_TL,self.settings['n_modes'])
    
        if self.settings['dim_less']:
            self.head = self.head * self.settings['gradient']
        else:
            if head_abs:                
                self.head = (-0.5+self.head) * self.dh + self.settings['head_abs']
            else:    
                self.head = self.head * self.dh 
        return self.head
       
    def q_total_homogeneous(self):

        """ Total flux into box: average flux along half of bottom boundary
        int_0^(xl/2) qz(x,z=0) dx """
        
        term = q_total(self.ratio_TL,**self.settings)
        self.q_total_hom= - 4. * self.kf * self.dh /np.pi**2 * term

        return self.q_total_hom

    def q_til_homogeneous(self):
        
        ### Exact solution        
        term = q_til(self.ratio_TL,self.ratio_HL,**self.settings)
        self.q_til_hom= - 4. * self.kf * self.dh /np.pi**2 * term

        return self.q_til_hom

    def q_vw_homogeneous(self):
        """ Total in horizontal direction below layer at z1: average flux along z axis at half of bottom boundary
        int_0^z1 qx(x=xl/2,z) dz """

        ### Exact solution for fully homogeneous case       
        term = q_total(self.ratio_TL,**self.settings) - q_til(self.ratio_TL,self.ratio_HT,**self.settings)
        self.q_vw_hom= - 4. * self.kf * self.dh /np.pi**2 * term

        return self.q_vw_hom

    def q_til_layer(self):
        
#        self.q_til_zero= self.settings['k_til_0'] * self.dh /(8.*self.D)
        self.q_til_zero= - self.settings['k_til_0'] * self.dh * self.L /(8.*self.D)
        
        return self.q_til_zero

    def q_vw_layer(self):

        ### Approximate solution to maintain constant q_vw
        term = q_total(self.ratio_HL,**self.settings)
        self.q_vw_zero= - 4. * self.kf * self.dh /np.pi**2 * term

        return self.q_vw_zero

    def q_total_layer(self):
        
#        self.q_til_layer()
#        self.q_vw_layer()
        self.q_total_zero= self.q_vw_layer() + self.q_til_layer()

        return self.q_total_zero

    def calculate_fluxes(self):
        self.q_total_homogeneous()
        self.q_til_homogeneous()
        self.q_vw_homogeneous()
        self.q_til_layer()
        self.q_vw_layer()
        self.q_total_layer()

    def q_total_trans_loglinear(self,tau):

        self.calculate_fluxes()
        self.q_total_trans = tau_transformation(tau,self.q_total_zero,self.q_total_hom)
#        self.q_total_trans = tau_transformation(tau,self.q_vw_zero + self.q_til_zero,self.q_total_hom)
#        self.q_total_trans = tau_transformation(tau,self.q_vw_hom,self.q_total_hom)

        return self.q_total_trans

    def q_til_trans_loglinear(self,tau):

        self.calculate_fluxes()
#        self.q_til_trans = tau_transformation(tau,self.q_til_zero,self.q_til_hom)
#        self.q_til_trans = tau_transformation(tau,0,self.q_total_hom - self.q_vw_zero)
        self.q_til_trans = tau_transformation(tau,self.q_til_zero,self.q_total_hom - self.q_vw_zero)

        return self.q_til_trans

    def mu_trans_scale(self,tau):
        
        self.calculate_fluxes()
        self.q_total_trans_loglinear(tau)
        self.mu_trans = 1 - self.q_vw_zero/self.q_total_trans

        ### direct calculation:
#        term1 = (self.q_vw_zero  - self.q_total_hom)*np.log(tau/tau[0])
#        term2 = (self.q_vw_zero  - self.q_total_hom)*np.log(tau) + self.q_total_hom*np.log(tau[0])

#        term1 = (self.q_vw_zero + self.q_til_zero - self.q_total_hom)*np.log(tau/tau[0]) + self.q_til_zero*np.log(tau[0])
#        term2 = (self.q_vw_zero + self.q_til_zero - self.q_total_hom)*np.log(tau) + self.q_total_hom*np.log(tau[0])

#        self.mu_trans = term1/term2
       
        return self.mu_trans

    def mu_test(self,tau):
        
#        self.calculate_fluxes()
#        self.q_total_trans_loglinear(tau)
#        self.mu_trans = 1 - self.q_vw_zero/self.q_total_trans

        ### direct calculation:
#        term1 = (self.q_vw_zero  - self.q_total_hom)*np.log(tau/tau[0])
#        term2 = (self.q_vw_zero  - self.q_total_hom)*np.log(tau) + self.q_total_hom*np.log(tau[0])
#
        term1 = (self.q_vw_zero + self.q_til_zero - self.q_total_hom)*np.log(tau/tau[0]) + self.q_til_zero*np.log(tau[0])
        term2 = (self.q_vw_zero + self.q_til_zero - self.q_total_hom)*np.log(tau) + self.q_total_hom*np.log(tau[0])
      
        return term1/term2


#    def q_vw_layer(self):
#
#        ### Approximate solution to maintain constant q_vw
#        term = q_total(self.ratio_HL,**self.settings)
#        self.q_vw_zero= 4. * self.kf * self.dh /np.pi**2 * term
#
#        return self.q_vw_zero


    def q_til_layer_sensitivity(self,length,thickness):
    
        q_til_sens= - self.kt0 * self.settings['gradient'] *length*length /(8.*thickness)

        return q_til_sens

    def q_vw_layer_sensitivity(self,length,depth):

        term = q_total(depth/length,approx = True,c1=0.89)
#        term = q_total(depth/length,approx = False,n=30, **self.settings)
        q_vw_zero_sens = -4. * self.kf * self.settings['gradient'] *length /np.pi**2 * term
    
        return q_vw_zero_sens

    def q_total_layer_sensitivity(self,length,depth,thickness):
    
        q_total_sens = self.q_til_layer_sensitivity(length,thickness) + self.q_vw_layer_sensitivity(length,depth)
 
        return q_total_sens

    def mu_layer_sensitivity(self,length,depth,thickness):
    
        mu_sens = self.q_til_layer_sensitivity(length,thickness)/self.q_total_layer_sensitivity(length,depth,thickness)
 
        return mu_sens

#    def q_total_trans(self,tau):
#
#        """ Q-Total for transition states """
#    
#        self.q_total_trans = 4. * self.kf * self.dh /np.pi**2 * q_total_trans(tau,self.ratio_TL,self.ratio_HL,**self.settings)
#        
#        return self.q_total_trans
#
#    def q_til_trans(self,tau):
#
#        self.q_til_trans =  4. * self.kf * self.dh /np.pi**2 * q_til_trans(tau,self.ratio_TL,self.ratio_HL,**self.settings)
#        
#        return self.q_til_trans
#
#    def q_vw_trans(self,tau):
#
#        self.q_vw_trans = self.q_total_trans - self.q_til_trans
#        
#        return self.q_vw_trans 
#
#    def mu_trans(self,tau):
#        
##        self.mu_trans = q_til_trans(tau,self.ratio_TL,self.ratio_HL,**self.settings)/q_total_trans(tau,self.ratio_TL,self.ratio_HL,**self.settings)
#        self.mu_trans = q_total_trans(tau,self.ratio_TL,self.ratio_HL,**self.settings)/q_til_trans(tau,self.ratio_TL,self.ratio_HL,**self.settings)
#        
#        return self.mu_trans

###############################################################################
### Auxiliary functions: Heads
###############################################################################

def head(xL,zT,ratio_TL,n_modes = 10,approx = False,c1=0.9):

    
    if approx:
        xxl,zzl=np.meshgrid(xL,zT)
        cosnx = np.cos(np.pi*xxl)        ###  solution in x-direction
        coshnz = np.cosh(c1*np.pi*(zzl-1.)*ratio_TL)
        coshnlz = np.cosh(np.pi*ratio_TL)   
    
        term_head= (cosnx*coshnz/coshnlz) ### sum over modes n
#        term_head = c1 *np.tanh(c1*np.pi*rat_TL)
    else:
        xxl,nn,zzl=np.meshgrid(xL,np.arange(1,n_modes+1,1),zT)
        an=1./(2.*nn-1)**2              ### fourier coefficients from initial condition 
        cosnx = np.cos((2.*nn-1)*np.pi*xxl)        ###  solution in x-direction
        coshnz = np.cosh((2.*nn-1)*np.pi*(zzl-1.)*ratio_TL)
        coshnlz = np.cosh((2.*nn-1)*np.pi*ratio_TL)   
    
        term_head= (np.sum(an*cosnx*coshnz/coshnlz,axis=0)).T ### sum over modes n

    return term_head

###############################################################################
### Auxiliary functions: Fluxes
###############################################################################

def q_total(rat_TL,approx = True,c1 = 0.92, n=10,**kwargs):   

    if approx:
        term = c1 *np.tanh(c1*np.pi*rat_TL)
    else:
        nn=np.arange(1,n+1,1)
        rat_TL2,nn2=np.meshgrid(rat_TL,nn)
        term = np.sum(np.power(-1,nn2+1)/(2*nn2-1)**2*np.tanh(np.pi*(2*nn2-1)*rat_TL2),axis = 0)
    
    return term

def q_til(rat_TL,rat_HT,approx = True,c1 = 0.94, n=10,**kwargs):

    """ Total in horizontal direction below layer at z1: average flux along z axis at half of bottom boundary
        int_0^z1 qx(x=xl/2,z) nz """
    
    if approx:
        rat_HL = rat_TL * rat_HT
        term = c1*(np.tanh(c1*np.pi*rat_TL) * np.cosh(c1*np.pi*rat_HL) - np.sinh(c1*np.pi*rat_HL))
    else:
        nn=np.arange(1,n+1,1)
        rat_TL2,nn2=np.meshgrid(rat_TL,nn)
        rat_HL = rat_TL2 * rat_HT
#        term = np.sum(np.power(-1,nn2+1)/(2*nn2-1)**2*np.sinh(np.pi*(2*nn2-1)*(rat_TL2-rat_HL))/np.cosh(np.pi*(2*nn2-1)*rat_TL2),axis = 0)
        term = np.sum(np.power(-1,nn2+1)/(2*nn2-1)**2* (np.tanh(np.pi*(2*nn2-1)*rat_TL2) * np.cosh(np.pi*(2*nn2-1)*rat_HL) - np.sinh(np.pi*(2*nn2-1)*rat_HL)),axis = 0)
    
    return term

def q_total_trans(rat_TL,rat_HL,tau,tau0,c1 = 0.92):

    q_total_hom = c1 *np.tanh(c1*np.pi*rat_TL)
    q_total_zero = c1 *np.tanh(c1*np.pi*rat_HL)
    q_total_trans = (q_total_zero-q_total_hom)*np.log(tau)/np.log(tau0) +  q_total_hom
    
    return q_total_trans

def q_total_trans_test(tau,rat_TL,rat_HL,kf,k_til_0,dh,D):

    q_total_hom = 4. * kf * dh /np.pi**2 * q_total(rat_TL)
    q_vw = 4. * kf * dh /np.pi**2 * q_total(rat_HL)
    
#    q_total_trans = tau_transformation(tau,q_total_zero,q_total_hom)
#    return q_total_trans

    log_tau = np.log(tau)
    m = (q_vw-q_total_hom)/log_tau[0]
    n = q_total_hom 
    
    return m*log_tau +  n 

#def q_total_trans(tau,rat_TL,rat_HL,**kwargs):
#    arg = rat_HL + tau*(rat_TL - rat_HL)   
#    return q_total(arg,**kwargs) 

def q_til_trans(rat_TL,rat_HL,tau,tau0,c1 = 0.92):

    q_total_hom = c1 *np.tanh(c1*np.pi*rat_TL)
    q_total_zero = c1 *np.tanh(c1*np.pi*rat_HL)
    q_total_trans = (q_total_zero-q_total_hom)*np.log(tau)/np.log(tau0) +  q_total_hom - q_total_zero
    
    return q_total_trans

#def q_til_trans(tau,rat_TL,rat_HL,c1 = 0.94,**kwargs):
#
#    """ Q-til for transition states """
#
#    arg = rat_HL + tau*( rat_TL - rat_HL)
#    term = q_til(arg,rat_HL/rat_TL,approx = True)
##    term = c1*(np.tanh(c1*np.pi*arg) * np.cosh(c1*np.pi*rat_HL) - np.sinh(c1*np.pi*rat_HL))
#    
#    return term

###############################################################################
### Auxiliary functions/checking approximations
###############################################################################

def tau_transformation(tau,q0,q1):

    log_tau = np.log(tau)
    m = -(q1-q0)/log_tau[0]
    n = q1 

    return m*log_tau +  n 


def fit_sum(c1,rat_TL,n,rat_HT = 1):
    """ fitting function to determine optimal parameter for approximation of 
    infinite sum over tangens hyperbolicus function """

    if rat_HT == 1:
        term_full = q_total(rat_TL,approx = False,n = n)
        term_approx = q_total(rat_TL,approx = True,c1 = c1)
    else:
        term_full = q_til(rat_TL,rat_HT,approx = False,n = n)
        term_approx = q_til(rat_TL,rat_HT,approx = True,c1 = c1)
        
    return term_full - term_approx

def fit_head(c1,xL,zT,ratio_TL,n_modes):

    term_full = head(xL,zT,ratio_TL,approx = False,n_modes = n_modes)
    term_approx = head(xL,zT,ratio_TL,approx = True,c1 = c1)
        
    return np.sum(term_full - term_approx)

def relative_difference(v1,v2):   
#    return 100*(v1-v2)/v1
    return np.abs((v1-v2)/v1)*100

def absolute_difference(v1,v2):   
#    return 100*(v1-v2)/v1
    return np.abs((v1-v2))

#def q_til_trans_test(tau,rat_TL,rat_HL,c1 = 0.94,**kwargs):
#
#    """ Q-til for transition states """
#
#    arg = rat_HL + tau*( rat_TL - rat_HL)
#    arg2 = tau*( rat_TL - rat_HL)
#    term = c1* np.sinh(c1*np.pi*arg2) / np.cosh(c1*np.pi*arg) 
#    
#    return term
# 

#def dilution_ratio(rat_TL,rat_HT,approx = True,c1_dil = 0.93, **kwargs):
#
#    """ dilution ratio: flux through injection layer to total flux"""
#    
#    if approx:
#        rat_HL = rat_TL * rat_HT
#        dil_ratio = c1_dil*(np.cosh(c1_dil*np.pi*rat_HL) - np.sinh(c1_dil*np.pi*rat_HL)/np.tanh(c1_dil*np.pi*rat_TL))
#
#    else:
#        dil_ratio= q_til(rat_TL,rat_HT,approx = False,**kwargs)/q_total(rat_TL,approx = False,**kwargs)
# 
#   
#    return dil_ratio    
