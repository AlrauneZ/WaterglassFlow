#!/usr/bin/env python3
"""
Script reproducing Fig. 4 of the Paper 
>Groundwater flow below construction pits and erosion of temporary horizontal layers of silicate grouting<
by Joris M. Dekker, Thomas Sweijen, Alraune Zech; Hydrogeology Journal
https://doi.org/10.1007/s10040-020-02246-3

@author: A. Zech
"""

import numpy as np
import matplotlib.pyplot as plt
from Class_Waterglas import WaterGlassTransport,Sim_WaterGlassTransport

##############################################################################
### LOAD DATA
##############################################################################

### Read in Data Create instance of flux settings in construction pit (default parameter) 
data = np.loadtxt('../data/Fluxes_Modflow_standard.csv',delimiter = ',', skiprows =1)
Sim1 = Sim_WaterGlassTransport()
Sim1.set_fluxes_from_data(data)
tau = np.logspace(np.log10(Sim1.tau0),np.log10(Sim1.tau[-1]),len(Sim1.tau))

### Create instance of flux settings in construction pit (default parameter) 
D1 = WaterGlassTransport()
### calculate analytical solutions for fluxes within domain
D1.calculate_fluxes() 

##############################################################################
### PLOTTING RESULTS
##############################################################################

### Plot setting specifics
plt.rc('text', usetex=True)
lw = 3
textsize = 10

plt.figure(figsize=[7.5,3.1])
ax=plt.subplot(1,2,1)
ax.plot(Sim1.tau,Sim1.q_total,c='C0',ls = '-',lw =lw,label = '$Q_\\textup{total}$ - sim') #'$Q_\\textup{total}$ - sim')
ax.plot(Sim1.tau,Sim1.q_til,c='C1',ls = '-',lw =lw,label = '$Q_\\textup{til}$ - sim')
ax.plot(Sim1.tau,Sim1.q_vw,c='C5',ls = '-',lw =lw,label = '$Q_\\textup{vw}$ - sim')    

ax.plot(tau,D1.q_total_trans_loglinear(tau),c='C0',ls = '--',lw =lw,label = '$Q_\\textup{total}$ - ana')
ax.plot(tau, D1.q_til_trans_loglinear(tau),c='C1',ls = '--',lw =lw,label = '$Q_\\textup{til}$ - ana')
ax.plot(tau,D1.q_vw_zero*np.ones(len(tau)),c='C5',ls = '--',lw =lw,label = '$Q_\\textup{vw}$ - ana ')

ax.set_xlabel('$\\tau = K_\\textup{til}/K_\\textup{sand}$',fontsize = textsize)
ax.set_ylabel('Flux $Q$',fontsize = textsize)
ax.grid(True)
ax.tick_params(axis="both",which="major",labelsize=textsize)
ax.legend(loc = 'upper center', ncol=2,fontsize = textsize)
ax.set_ylim([0,0.145])
ax.text(-0.1,-0.15,r'a', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))

ax=plt.subplot(1,2,2)
ax.plot(Sim1.tau,Sim1.q_total,c='C0',ls = '-',lw =lw,label = '$Q_\\textup{total}$ - sim') #'$Q_\\textup{total}$ - sim')
ax.plot(Sim1.tau,Sim1.q_til,c='C1',ls = '-',lw =lw,label = '$Q_\\textup{til}$ - sim')
ax.plot(Sim1.tau,Sim1.q_vw,c='C5',ls = '-',lw =lw,label = '$Q_\\textup{vw}$ - sim')    

ax.plot(tau,D1.q_total_trans_loglinear(tau),c='C0',ls = '--',lw =lw,label = '$Q_\\textup{total}$ - ana')
ax.plot(tau, D1.q_til_trans_loglinear(tau),c='C1',ls = '--',lw =lw,label = '$Q_\\textup{til}$ - ana')
ax.plot(tau,D1.q_vw_zero*np.ones(len(tau)),c='C5',ls = '--',lw =lw,label = '$Q_\\textup{vw}$ - ana ')

ax.set_xlabel('$\\tau = K_\\textup{til}/K_\\textup{sand}$',fontsize = textsize)
ax.set_ylabel('Flux $Q$',fontsize = textsize)
ax.grid(True)
ax.tick_params(axis="both",which="major",labelsize=textsize)
ax.set_xscale('log')
ax.text(-0.1,-0.15,r'b', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))

plt.tight_layout()
# plt.savefig('../results/Fig04_q_trans.png',dpi=300)   
plt.savefig('../results/Fig04_q_trans.pdf')   
