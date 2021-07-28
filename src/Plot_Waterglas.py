#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:46:20 2020

@author: zech0001
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq #curve_fit
import ufz
from ogs5py import OGS

import Class_Waterglas as cw
from Class_SimWaterglas import Sim_WaterGlassTransport

plt.close('all')


ex = 'Fig_head_contours'#Fig_q_trans'

"""
Fig_head_contours (partly Fig 2 - paper)
Fig_head_profiles (Fig 3 - paper)
Fig_q_trans (Fig 4 - paper)
Fig_dilution_ratio (Fig 5 - paper)
Fig_sensitivity (Fig 6-8 - paper)

SI_q_total_approx
SI_q_til_approx
SI_head_profiles
SI_rel_diff_L
SI_rel_diff_H
SI_rel_diff_D
SI_streamlines
SI_q_trans_error

"""

lw = 3
textsize = 10
ex_sub = ''
#ex_sub = '_c'

save=True #False #
dir_fig = '/home/zech0001/Projects/Waterglass/Figures/'
dir_data = '/home/zech0001/Projects/Waterglass/Data_Figures/'
plt.rc('text', usetex=True)
""" ---------------------------------------------------------------
Setup Flow in a Construction pit - analytical vs. numerical results
----------------------------------------------------------------"""

### Create instance of flux settings in construction pit (default parameter) 
D1 = cw.WaterGlassTransport()
D1.calculate_fluxes() ### calculate analytical solutions for fluxes within domain

### Read in Data Create instance of flux settings in construction pit (default parameter) 
file_modflow = '{}Fluxes_Modflow_standard.csv'.format(dir_data)
data = np.loadtxt(file_modflow,delimiter = ',', skiprows =1)

Sim1 = Sim_WaterGlassTransport()
Sim1.set_fluxes_from_data(data)

tau = np.logspace(np.log10(Sim1.tau0),np.log10(Sim1.tau[-1]),len(Sim1.tau))
domain={'x_0':0,'x_L':50,'z_0':0,'z_L':10,'z_1':1,'z_2':1,'nx':500,'nz':100}

### Data of relative differences in Fluxes for SI:

if ex =='':
    print('Figure not specified')


elif ex == 'Fig_head_contours':

    ex_sub = '_b'
    textsize = 20

    if ex_sub == '_a':
        text = r'b) $t = 0$'
        file_head = '{}modlfow_head_t0.csv'.format(dir_data)
    elif ex_sub == '_b':
        text = r'd) $t =  \infty$'
        file_head = '{}modlfow_head_tinf.csv'.format(dir_data)

    head = np.loadtxt(file_head,delimiter = ',')
    xcoords =np.linspace(domain['x_0'],domain['x_L'],head.shape[1],endpoint=True)
    nz = head.shape[0]
    zcoords =np.linspace(domain['z_L'],0,nz,endpoint=True)
#    dz  = (domain['z_L'] - domain['z_0'])/nz
#    zcoords =np.linspace(domain['z_L']+dz,dz,nz,endpoint=True)
#    zcoords = np.linspace(T,0,nlay,endpoint=True)
#    zcoords = np.linspace(T,delv,nlay)

    levels = np.linspace(0.268,0.288,40,endpoint=True)
#    levels = np.linspace(0.268,0.287,51,endpoint=True)
#    zmax = 4
#    izmax = int(zmax / T * head.shape[0])
#    plt.figure(figsize=[8.1,8.2*0.4])
    plt.figure(figsize=[10.8,10.8*0.4])
    ax = plt.subplot(111)
    cmap = plt.get_cmap('RdYlGn')#'PiYG') #
    cf = plt.contourf(xcoords[1:-1],zcoords[1:],head[1:,1:-1],cmap=cmap,levels=levels)
    cbar = plt.colorbar(cf,format='%.3f')
    cbar.set_label('head [m]',fontsize = textsize)
    cbar.set_ticks([0.268,0.272,0.276,0.28,0.284,0.288])
    cbar.ax.tick_params(labelsize=textsize) 
#    cbar.ax.tick_params(labelsize=textsize)
    plt.xlabel('$x$ [m]',fontsize = textsize)
    plt.ylabel('$y$ [m]',fontsize = textsize)
    plt.text(0.05,0.85,text,fontsize = textsize, transform=ax.transAxes,bbox=dict(boxstyle='round', facecolor='w', alpha=0.9))
    plt.plot(xcoords,domain['z_1']*np.ones_like(xcoords),c='0.1',ls='--',lw=1)
    plt.plot(xcoords,(domain['z_1']+domain['z_2'])*np.ones_like(xcoords),c='0.1',ls='--',lw=1)
    plt.ylim([0,4])
    plt.xlim([0,50])
    plt.yticks([0,1,2,3,4])
    
elif ex == 'Fig_head_profiles':

#    textsize = 12
    cc1 = ufz.get_brewer('Paired12', rgb=True,reverse=True)
    plt.figure(figsize=[7.5,3])
    for ii in [0,1]:
        if ii ==0:
            file_head = '{}data_heads_layer_t0.csv'.format(dir_data)
            s = r'a) $t = 0$'
        elif ii == 1:            
            file_head = '{}data_heads_layer_tend.csv'.format(dir_data)
            s = r'b) $t =  \infty$'

        data_head = np.loadtxt(file_head,delimiter = ',', skiprows =1)
        ax=plt.subplot(1,2,ii+1)
        ax.plot(data_head[1:-1,0],data_head[1:-1,4],c=cc1[9],ls = '-',lw =lw+0.5,label = r'$h(x)$ above til, sim')
        ax.plot(data_head[:,0],data_head[:,2],c=cc1[8],ls = '--',lw =lw,label = r'$h(x)$ above til, analyt')
        ax.plot(data_head[1:-1,0],data_head[1:-1,3],c=cc1[3],ls = '-',lw =lw+0.5,label = r'$h(x)$ below til, sim')
        ax.plot(data_head[:,0],data_head[:,1],c=cc1[2],ls = '--',lw =lw,label = r'$h(x)$ below til, analyt')
        ax.set_xlabel(r'$x$ [m]')
        ax.tick_params(axis="both",which="major",labelsize=textsize)
        ax.grid(True)
        ax.set_xlim([0,50]) 
        ax.set_ylim([0.268,0.289]) 
        ax.set_yticks([0.27,0.275,0.28,0.285])
        ax.text(0.1,0.1,s, fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
        if ii ==0:
            ax.set_ylabel('Hydraulic head $h(x)$ [m]')
        elif ii == 1:
            ax.legend(loc = 'upper right',fontsize = textsize-1)
#        ax.set_ylim([0,0.137])


elif ex == 'Fig_q_trans':

    plt.figure(figsize=[7.5,3.1])
#    plt.figure(figsize=[7.25,3.5])
#    ex_sub = 'log'    
    for ii in [1,2]:
        ax=plt.subplot(1,2,ii)

        ax.plot(Sim1.tau,Sim1.q_total,c='C0',ls = '-',lw =lw,label = '$Q_\\textup{total}$ - sim') #'$Q_\\textup{total}$ - sim')
        ax.plot(Sim1.tau,Sim1.q_til,c='C1',ls = '-',lw =lw,label = '$Q_\\textup{til}$ - sim')
        ax.plot(Sim1.tau,Sim1.q_vw,c='C5',ls = '-',lw =lw,label = '$Q_\\textup{vw}$ - sim')    
    
        ax.plot(tau,D1.q_total_trans_loglinear(tau),c='C0',ls = '--',lw =lw,label = '$Q_\\textup{total}$ - ana')
        ax.plot(tau, D1.q_til_trans_loglinear(tau),c='C1',ls = '--',lw =lw,label = '$Q_\\textup{til}$ - ana')
        ax.plot(tau,D1.q_vw_zero*np.ones(len(tau)),c='C5',ls = '--',lw =lw,label = '$Q_\\textup{vw}$ - ana ')
        print(D1.q_til_zero)

#        eps_vw = cw.relative_difference(Sim1.q_vw,D1.q_vw_zero*np.ones(len(tau)))
#        eps_til = cw.relative_difference(q_til_sim,q_til_ana)
#        eps_total = cw.relative_difference(q_total_sim,q_total_ana)
#        eps_mu = cw.relative_difference(mu_sim,mu_ana)

        ax.set_xlabel('$\\tau = K_\\textup{til}/K_\\textup{sand}$',fontsize = textsize)
        ax.set_ylabel('Flux $Q$',fontsize = textsize)
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize)

        if ii ==1:
            ax.legend(loc = 'upper center', ncol=2,fontsize = textsize)
            ax.set_ylim([0,0.145])
            ax.text(-0.1,-0.15,r'a', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
        else:
            ax.set_xscale('log')
            ax.text(-0.1,-0.15,r'b', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))

#    plt.yscale('log')   
    
#    plt.plot(Sim1.tau,Sim1.q_total,c='C0',ls = '-',label = '$Q_{total}(k_{til}/k_{sand})$ - sim')
#    plt.plot(Sim1.tau,Sim1.q_til,c='C1',ls = '-',label = '$Q_{til}(k_{til}/k_{sand})$ - sim')
#    plt.plot(Sim1.tau,Sim1.q_vw,c='C3',ls = '-',label = '$Q_{vw}(k_{til}/k_{sand})$ - sim')    
#
#    plt.plot(tau,D1.q_total_trans_loglinear(tau),c='C0',ls = '--',label = r'$Q_{total} (\tau)$ - ana')
#    plt.plot(tau, D1.q_til_trans_loglinear(tau),c='C1',ls = '--',label = r'$Q_{til} (\tau)$ - ana')
#    plt.plot(tau,D1.q_vw_zero*np.ones(len(tau)),c='C3',ls = '--',label = r'$Q_{vw}$ -ana ')
#
##    print(D1.q_til_zero,Sim1.q_til[0])
#
#    plt.xlabel(r'$\tau_k = k_{til}/k_{sand}$',fontsize = textsize)
#    plt.ylabel('Fluxes $Q$',fontsize = textsize)
#    plt.ylim([0,0.139])
##    plt.xlim([Sim1.tau[0],Sim1.tau[-1]])
##    plt.title('Fluxes in homogeneous domain')
#
#    plt.legend(loc = 'upper left', ncol=2,fontsize = textsize)
#    plt.xscale(ex_sub)
##    plt.yscale('log')   

elif ex == 'Fig_dilution_ratio':
    mu_sill = 0.1
    plt.figure(figsize=[7.5,2.8])
    for ii in [1,2]:
        ax=plt.subplot(1,2,ii)
        ax.plot(Sim1.tau,Sim1.mu,c='C2',ls = '-',lw =lw,label = r'$\mu(K_\textup{til}/K_\textup{sand})$ - sim')
        ax.plot(tau,D1.mu_trans_scale(tau),c='C2',ls = '--',lw =lw,label = r'$\mu(\tau)$ - ana')
        ax.set_xlabel(r'$\tau = K_\textup{til}/K_\textup{sand}$')
        ax.grid(True)

        if ii ==1:
            ax.set_ylabel(r'Dilution ratio $\mu = Q_\textup{til}/Q_\textup{total}$')
            ax.legend(loc = 'lower right',fontsize = textsize)
#            ax.set_ylim([0,0.137])
            ax.text(-0.07,-0.15,r'a', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
        else:
            ax.set_xscale('log')
            ax.text(-0.07,-0.15,r'b', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))

elif ex == 'Fig_sensitivity':

    ex_sub = '_c'

    tau_range = [0.001,0.01,0.1,1]
    tau0 =D1.settings['k_til_0'] /  D1.kf
    rat_TL =  np.arange(0,1,0.01)
    cc1 = ufz.get_brewer('Blues9', rgb=True,reverse=True)[::2]
    cc2 = ufz.get_brewer('YlOrRd5', rgb=True,reverse=True)
    cc3 = ufz.get_brewer('Greens9', rgb=True,reverse=True)[::2]
    abc = ['a','b','c','d']
#    cc = ufz.get_brewer('Paired12', rgb=True)
    plt.figure(figsize=[7.5,5.4])
    for ii,tau in enumerate(tau_range):
        ax=plt.subplot(2,2,ii+1)
        rat_HT_range = [0.05,0.1,0.25,0.5]
        rat_HT = 0.1 #D1.ratio_HT
        for ir,rat_HT in enumerate(rat_HT_range):        
            rat_HL = rat_TL*rat_HT #D1.ratio_HL
            q_tot_tl = cw.q_total_trans(rat_TL,rat_HL,tau,tau0)
            q_til_tl = cw.q_til_trans(rat_TL,rat_HL,tau,tau0)
            mu_tl = q_til_tl/q_tot_tl
            if ex_sub == '_a':
                ax.plot(rat_TL,q_tot_tl,c=cc1[ir],ls = '--',lw =lw,label = r'$H/T = {}$'.format(rat_HT),zorder = 5-ir)
                ax.set_ylim([0,0.95])
            elif ex_sub == '_b':
                ax.plot(rat_TL,q_til_tl,c=cc2[ir],ls = '--',lw =lw,label = r'$H/T = {}$'.format(rat_HT),zorder = 5-ir)
                ax.set_ylim([0,0.95])
            elif ex_sub == '_c':
                ax.plot(rat_TL,mu_tl,c=cc3[ir],ls = '--',lw =lw,label = r'$H/T = {}$'.format(rat_HT),zorder = 5-ir)
                ax.set_ylim([0,1.25])

        ax.text(0.05,0.85,r'{}) $\tau = {}$'.format(abc[ii],tau),transform=ax.transAxes,fontsize = textsize,bbox=dict(boxstyle='round',facecolor='w'))
        ax.set_xlim([0,1])
        ax.grid(True)

        if ii >1:
            ax.set_xlabel(r'$T/L$')
        if ii in [0,2]:
            if ex_sub == '_a':
                ax.set_ylabel(r'$\hat C \cdot Q_\textup{total} (\tau,T,H,L)$',fontsize = textsize)
                loc,ileg = 'lower right',2
            elif ex_sub == '_b':
                ax.set_ylabel(r'$\hat C \cdot Q_\textup{til} (\tau,T,H,L)$',fontsize = textsize)
                loc,ileg = 'upper right',0
            elif ex_sub == '_c':
                ax.set_ylabel(r'$\mu = Q_\textup{til}/Q_\textup{total}$',fontsize = textsize)
                loc,ileg = 'upper right',0
        if ii == ileg:
            ax.legend(loc = loc,fontsize = textsize)
##            ax.set_ylim([0,0.137])
#        else:
#            ax.set_xscale('log')

elif ex == 'SI_streamlines':

    domain={'x_0':0,'x_L':50,'z_0':0,'z_L':10,'z_1':1,'z_2':2,'nx':500,'nz':100}
    model = OGS(task_root= '{}flow_in_box_hom'.format(dir_data), task_id='flow_in_box_hom')
    output = model.readvtk()
    velocity_x = output['']['DATA'][1]['cell_data']['quad']['VELOCITY1_X'].reshape((domain['nz'],domain['nx']))*86400
    velocity_z = output['']['DATA'][1]['cell_data']['quad']['VELOCITY1_Y'].reshape((domain['nz'],domain['nx']))*86400
       
    x1D=np.linspace(domain['x_0'],domain['x_L'],domain['nx'],endpoint=True)
    z1D=np.linspace(domain['z_0'],domain['z_L'],domain['nz'],endpoint=True)
    xxs,zzs=np.meshgrid(x1D,z1D)
             
    plt.figure(1,figsize=[7.5,3])
#    cmap=plt.get_cmap('summer_r')
    cmap=plt.get_cmap('Blues')
    cf=plt.streamplot(xxs, zzs, velocity_x, velocity_z,color = velocity_x, density=1,cmap=cmap) #, color=u,linewidth=5*speed/speed.max())
#    plt.title('Simulated streamfunction')
    cbar = plt.colorbar(cf.lines,format='%.3f')
    cbar.set_label('velocity [m/d]',fontsize = textsize)
    plt.xlabel('$x$ [m]',fontsize = textsize), plt.ylabel('$y$ [m]',fontsize = textsize)
    plt.ylim([0,10])
    plt.xlim([0,50])

elif ex == 'SI_q_total_approx':
   
    rat_TL = np.arange(0,1,0.02)
    c0,n = 0.92,20
    c1, success = leastsq(cw.fit_sum,c0,args = (rat_TL,n,0))
    print(c1,success)

    #c2 = c1
    c2 = 0.9
    q1 = cw.q_total(rat_TL,approx = True, c1=c1)
    q2 = cw.q_total(rat_TL,approx = False)#,n=n)   

#    print('Absolute Difference: ',np.abs(q2-q1))
#    print('Relative Difference: ',np.abs((q2-q1)/q1))

    ### For standard values:    
    q1s,q2s = cw.q_total(0.1,approx = True, c1=c2), cw.q_total(0.1,approx = False)#,n=n)   
    print('Absolute Difference: ',np.abs(q2s-q1s))
    print('Relative Difference: ',np.abs((q2s-q1s)/q1s))

    
    plt.figure(figsize=[5,3])
    plt.plot(rat_TL,q2,label = 'exact solution',lw = lw)
    plt.plot(rat_TL,q1,'--',label = 'approximate solution',lw = lw)
    plt.legend(fontsize = textsize)
    plt.xlabel(r'$T/L$',fontsize = textsize)
    plt.ylabel(r'$\hat C \cdot Q_\textup{total}$',fontsize = textsize)
    
elif ex == 'SI_q_til_approx':

    plt.figure(figsize=[6,4])
    rat_TL = np.arange(0,2.02,0.02)

    q1 = cw.q_total(rat_TL)#,approx = True, c1=c1)    
    plt.plot(rat_TL,q1,c='k',lw = lw,label = '$Q_\textup{total}$')

    for ir,rat_HT in enumerate(np.arange(0.1,1,0.2)):
        c0,n = 0.92,20
        c1, success = leastsq(cw.fit_sum,c0,args = (rat_TL,n,rat_HT))
        print('Optimal parameter: ',c1)
        
        c1,n = 0.92,10
        
        q2 = cw.q_til(rat_TL,rat_HT,approx = True, c1=c1)
        q3 = cw.q_til(rat_TL,rat_HT,approx = False, n = n)

        ### For standard values:   
        rat_TL_standard = 0.1
        q2s,q3s = cw.q_til(rat_TL_standard,rat_HT,approx = True, c1=c1), cw.q_til(rat_TL_standard,rat_HT,approx = False)#,n=n)   

        print('Absolute Difference: ',np.abs(q3s-q2s))
        print('Relative Difference: ',np.abs((q3s-q2s)/q3s))
        
        plt.plot(rat_TL,q3,'-',lw = lw,c = 'C{}'.format(ir),label = '$Q_\textup{til}(H/T = {:.1f})$'.format(rat_HT))
        plt.plot(rat_TL[::5],q2[::5],ls = '',marker = '*',ms=8,c = 'C{}'.format(ir))#,label = 'qtil (approx)')

    plt.legend(loc = 'upper center',ncol = 3,fontsize = textsize)
    plt.grid(True)
    plt.xlim([0,2])
    plt.ylim([0,1.15])
    plt.xlabel(r'$T/L$',fontsize = textsize)
    plt.ylabel(r'$\hat C \cdot Q(T/L)$',fontsize = textsize)

elif ex == 'SI_rel_diff_L':

    """ Sensitivity Study for Supporting Information """
    file_reldiff = '{}data_reldiff_L_H1.csv'.format(dir_data)
    rel_diff = np.loadtxt(file_reldiff,delimiter = ',', skiprows =1)
    arg = rel_diff[:,0]

    plt.figure(figsize=[5,3.5])
    ax=plt.subplot(1,1,1)

    q_vw_sim = rel_diff[:,4]
    q_til_sim = rel_diff[:,7]
    q_total_sim = rel_diff[:,1] #q_vw_sim + q_til_sim
    mu_sim = q_til_sim/q_total_sim
    
    q_til_ana = D1.q_til_layer_sensitivity(length=arg,thickness=D1.D)
    q_vw_ana = D1.q_vw_layer_sensitivity(length=arg,depth = D1.H)
    q_total_ana = D1.q_total_layer_sensitivity(length=arg,depth = D1.H,thickness=D1.D)
    mu_ana = D1.mu_layer_sensitivity(length=arg,depth = D1.H,thickness=D1.D)

    eps_vw = cw.relative_difference(q_vw_sim,q_vw_ana)
    eps_til = cw.relative_difference(q_til_sim,q_til_ana)
    eps_total = cw.relative_difference(q_total_sim,q_total_ana)
    eps_mu = cw.relative_difference(mu_sim,mu_ana)


    ax.plot([D1.L,D1.L],[0,10],'k--',lw = 1)
    ax.plot(arg,eps_total,c='C0',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{total})$')
    ax.plot(arg,eps_til,c='C1',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{til})$',zorder = 5)
    ax.plot(arg,eps_vw,c='C5',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{vw})$')    
    ax.plot(arg,eps_mu,c='C2',ls = '-',lw =lw,label = r'$\epsilon(\mu)$')
    ax.set_xlabel(r'$L$ [m]')
    ax.set_xlim([10,80]) 
    ax.set_ylim([0,8]) 
       

    ax.set_ylabel('Relative difference $\epsilon$ [%]')
    ax.tick_params(axis="both",which="major",labelsize=textsize)
    ax.grid(True)
    ax.text(0.1,0.9,r'$t = 0$', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
    ax.legend(loc = 'upper right',fontsize = textsize)

elif ex == 'SI_rel_diff_H':

    """ Sensitivity Study for Supporting Information """

    file_reldiff = '{}data_reldiff_H_L50.csv'.format(dir_data)
    rel_diff = np.loadtxt(file_reldiff,delimiter = ',', skiprows =1)
  
    q_vw_sim = rel_diff[:,4]
    q_til_sim = rel_diff[:,7]
    q_total_sim = rel_diff[:,1] #q_vw_sim + q_til_sim
    mu_sim = q_til_sim/q_total_sim
    
    arg = rel_diff[:,0]*D1.T - D1.D
    
    q_til_ana = D1.q_til_layer_sensitivity(length=D1.L,thickness=np.ones_like(arg)*D1.D)
    q_vw_ana = D1.q_vw_layer_sensitivity(length=D1.L,depth = arg)
    q_total_ana = D1.q_total_layer_sensitivity(length=D1.L,depth = arg,thickness=D1.D)
    mu_ana = D1.mu_layer_sensitivity(length=D1.L,depth = arg,thickness=D1.D)

    eps_vw = cw.relative_difference(q_vw_sim,q_vw_ana)
    eps_til = cw.relative_difference(q_til_sim,q_til_ana)
    eps_total = cw.relative_difference(q_total_sim,q_total_ana)
    eps_mu = cw.relative_difference(mu_sim,mu_ana)

    plt.figure(figsize=[5,3.5])
    ax=plt.subplot(1,1,1)

    ax.plot([D1.H,D1.H],[0,10],'k--',lw = 1)
    ax.plot(arg,eps_total,c='C0',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{total})$')
    ax.plot(arg,eps_til,c='C1',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{til})$',zorder = 5)
    ax.plot(arg,eps_vw,c='C5',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{vw})$')    
    ax.plot(arg,eps_mu,c='C2',ls = '-',lw =lw,label = r'$\epsilon(\mu)$')
    ax.set_xlabel(r'$H$ [m]')
    ax.set_xlim([0.5,5]) 
    ax.set_ylim([0,8]) 
       
    ax.set_ylabel('Relative difference $\epsilon$ [%]')
    ax.tick_params(axis="both",which="major",labelsize=textsize)
    ax.grid(True)
    ax.text(0.2,0.9,r'$t = 0$', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
#    ax.legend(loc = 'center right',fontsize = textsize)
    ax.legend(fontsize = textsize,bbox_to_anchor = (0.7,0.2))

elif ex == 'SI_rel_diff_D':

    """ Sensitivity Study for Supporting Information """

    file_reldiff = '{}data_reldiff_D.csv'.format(dir_data)
    rel_diff = np.loadtxt(file_reldiff,delimiter = ',', skiprows =1)

    q_vw_sim = rel_diff[:,4]
    q_til_sim = rel_diff[:,7]
    q_total_sim = rel_diff[:,1] #q_vw_sim + q_til_sim
    mu_sim = q_til_sim/q_total_sim
    
    arg = rel_diff[:,0]
    
    q_til_ana = D1.q_til_layer_sensitivity(length=D1.L,thickness=arg)
    q_vw_ana = D1.q_vw_layer_sensitivity(length=D1.L,depth = np.ones_like(arg)*D1.H)
    q_total_ana = D1.q_total_layer_sensitivity(length=D1.L,depth = D1.H ,thickness=arg)
    mu_ana = D1.mu_layer_sensitivity(length=D1.L,depth = D1.H, thickness=arg)

    eps_vw = cw.relative_difference(q_vw_sim,q_vw_ana)
    eps_til = cw.relative_difference(q_til_sim,q_til_ana)
    eps_total = cw.relative_difference(q_total_sim,q_total_ana)
    eps_mu = cw.relative_difference(mu_sim,mu_ana)


    plt.figure(figsize=[5,3.5])
    ax=plt.subplot(1,1,1)
    
    ax.plot([D1.D,D1.D],[0,10],'k--',lw = 1)
    ax.plot(arg,eps_total,c='C0',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{total})$')
    ax.plot(arg,eps_til,c='C1',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{til})$',zorder = 5)
    ax.plot(arg,eps_vw,c='C5',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{vw})$')    
    ax.plot(arg,eps_mu,c='C2',ls = '-',lw =lw,label = r'$\epsilon(\mu)$')

    ax.set_xlabel(r'$D$ [m]')
    ax.set_ylabel('Relative difference $\epsilon$ [%]')
    ax.tick_params(axis="both",which="major",labelsize=textsize)
    ax.grid(True)
#    ax.set_xlim([0,80]) 
    ax.set_ylim([0,7]) 
    ax.set_xlim([0.25,3.5]) 
    ax.text(0.1,0.9,r'$t = 0$', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
    ax.legend(loc = 'upper right',fontsize = textsize)
     
elif ex == 'SI_head_profiles':
    ### Figure of head profiles in Supporting Information

    file_head = '{}data_headprofiles_t0_SI.csv'.format(dir_data)
    head_profiles = np.loadtxt(file_head,delimiter = ',', skiprows =1)

    cc1 = ufz.get_brewer('Paired12', rgb=True,reverse=True)
    plt.figure(figsize=[5,3.5])
    ax=plt.subplot(1,1,1)

    ax.plot(head_profiles[:,0],head_profiles[:,1],c=cc1[9],ls = '-',lw =lw+1,label = r'$h(x,y=H+D)$ (just above til)')
    ax.plot(head_profiles[:,0],head_profiles[:,2],c=cc1[3],ls = '-',lw =lw+1,label = r'$h(x,y=H)$ (just below til)')
    ax.plot(head_profiles[:,0],head_profiles[:,3],c='k',ls = '--',lw = lw,label = r'$h(x,y=0)$ (bottom of domain)')

#    ax.plot(data_head[:,0],data_head[:,2],c=cc1[8],ls = '--',lw =lw,label = r'$h(x)$ above til, analyt')
#    ax.plot(data_head[1:-1,0],data_head[1:-1,3]
#    ax.plot(data_head[:,0],data_head[:,1],c=cc1[2],ls = '--',lw =lw,label = r'$h(x)$ below til, analyt')
##        ax.plot(Sim1.tau,Sim1.mu
##        ax.plot(tau,D1.mu_trans_scale(tau),c='C3',ls = '--',lw =lw,label = r'$\mu(\tau)$ - ana')
###        ax.plot(tau,D1.mu_test(tau),c='C9',ls = ':',label = r'$\mu(\tau)$ - ana')
###        ax.plot(tau,mu_sill*np.ones_like(tau),'--k',lw=1)
    ax.set_xlabel(r'$x$ [m]')
    ax.tick_params(axis="both",which="major",labelsize=textsize)
    ax.grid(True)
    ax.set_xlim([0,50]) 
#    ax.set_ylim([0.268,0.287]) 
#    ax.set_yticks([0.27,0.275,0.28,0.285])
    ax.text(0.1,0.1,r'$t = 0$', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
    ax.set_ylabel('Hydraulic head $h(x)$ [m]')
    ax.legend(loc = 'upper right',fontsize = textsize)
#        ax.set_ylim([0,0.137])


elif ex == 'SI_q_trans_error':

    plt.figure(figsize=[7.5,3.1])
#    plt.figure(figsize=[7.25,3.5])


    ax=plt.subplot(1,1,1)
    eps_til = cw.relative_difference(Sim1.q_til, D1.q_til_trans_loglinear(tau))
    eps_vw = cw.relative_difference(Sim1.q_vw,D1.q_vw_zero*np.ones(len(tau)))
    eps_total = cw.relative_difference(Sim1.q_total,D1.q_total_trans_loglinear(tau))
    eps_mu = cw.relative_difference(Sim1.mu,D1.mu_trans_scale(tau))

#    eps_til = cw.absolute_difference(Sim1.q_til, D1.q_til_trans_loglinear(tau))
#    eps_vw = cw.absolute_difference(Sim1.q_vw,D1.q_vw_zero*np.ones(len(tau)))
#    eps_total = cw.absolute_difference(Sim1.q_total,D1.q_total_trans_loglinear(tau))
#    eps_mu = cw.absolute_difference(Sim1.mu,D1.mu_trans_scale(tau))
#
    ax = plt.subplot(111)
    ax.plot(Sim1.tau,eps_total,c='C0',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{total})$')
    ax.plot(Sim1.tau,eps_til,c='C1',ls = '-',lw =lw,label = r'$\epsilon(Q\textup_{til})$')
    ax.plot(Sim1.tau,eps_vw,c='C5',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{vw})$')
    ax.plot(Sim1.tau,eps_mu,c='C2',ls = '-',lw =lw,label = r'$\epsilon(\mu)$')
    ax.set_ylim([0,10])
    it = 0
    print(tau[it])
#    print('eps Q_total: ',np.max(eps_total))
#    print('eps Q_til: ',np.max(eps_til))
#    print('eps Q_vw: ',np.max(eps_vw))

    print('eps Q_total: ',eps_total[it])
    print('eps Q_til: ',eps_til[it])
    print('eps Q_vw: ',eps_vw[it])
    print('eps mu: ',eps_mu[it])
    
#    ax.set_xlabel(r'$\tau = K_{til}/K_{sand}$',fontsize = textsize)
#        ax.set_ylabel('Fluxes $Q$',fontsize = textsize)
#        ax.grid(True)
#        ax.tick_params(axis="both",which="major",labelsize=textsize)

    ax.legend(loc = 'upper center', ncol=2,fontsize = textsize)
#            ax.set_ylim([0,0.145])

     
else:
    print('Figure not specified')

if ex != 'Fig_head_contours':
    plt.grid(True)
plt.tick_params(axis="both",which="major",labelsize=textsize)
if save:            
    plt.tight_layout()
    plt.savefig('{}{}{}.png'.format(dir_fig,ex,ex_sub),dpi=300)   
    plt.savefig('{}{}{}.pdf'.format(dir_fig,ex,ex_sub))   
    print('Figure saved to {}'.format(dir_fig))
