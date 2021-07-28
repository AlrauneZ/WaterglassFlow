#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:53:06 2020

@author: zech0001
"""

import numpy as np
import copy

DEF_DATA = dict(
        domain={'x_0':0,'x_L':50,'z_0':0,'z_L':10,'z_1':1,'z_2':2,'nx':50,'nz':50},
        k_sand = 25., #m/d
        k_til_0 = 0.00864, #m/d = 1e-7m/s
        gradient = 3.8e-4, # m/m # hydraulic gradient in aquifer       
        head_abs = 0.287,  # m absolute head value
        )

class Sim_WaterGlassTransport():

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

    def set_fluxes_from_data(self,data):
 
        self.tau = data[:,0]
        self.q_total = data[:,1]
        self.q_vw = data[:,2]
        self.q_til = data[:,3]
        self.mu = self.q_til/self.q_total
        self.tau0 = self.tau[0] 