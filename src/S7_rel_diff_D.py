#!/usr/bin/env python3
"""
Script reproducing Fig. S7 of the Supporting Information of Paper 
>Groundwater flow below construction pits and erosion of temporary horizontal layers of silicate grouting<
by Joris M. Dekker, Thomas Sweijen, Alraune Zech; Hydrogeology Journal
https://doi.org/10.1007/s10040-020-02246-3

@author: A. Zech
"""

import numpy as np
import matplotlib.pyplot as plt
from Class_Waterglas import WaterGlassTransport,relative_difference 

##############################################################################
### LOAD DATA AND SPECIFY SETTINGS
##############################################################################

### Create instance of flux settings in construction pit (default parameter) 
D1 = WaterGlassTransport()
D1.calculate_fluxes() ### calculate analytical solutions for fluxes within domain

rel_diff = np.loadtxt('../data/data_reldiff_D.csv',delimiter = ',', skiprows =1)
arg = rel_diff[:,0]

q_vw_sim = rel_diff[:,4]
q_til_sim = rel_diff[:,7]
q_total_sim = rel_diff[:,1] 
mu_sim = q_til_sim/q_total_sim

q_til_ana = D1.q_til_layer_sensitivity(length=D1.L,thickness=arg)
q_vw_ana = D1.q_vw_layer_sensitivity(length=D1.L,depth = np.ones_like(arg)*D1.H)
q_total_ana = D1.q_total_layer_sensitivity(length=D1.L,depth = D1.H ,thickness=arg)
mu_ana = D1.mu_layer_sensitivity(length=D1.L,depth = D1.H, thickness=arg)

eps_vw = relative_difference(q_vw_sim,q_vw_ana)
eps_til = relative_difference(q_til_sim,q_til_ana)
eps_total = relative_difference(q_total_sim,q_total_ana)
eps_mu = relative_difference(mu_sim,mu_ana)

##############################################################################
### PLOTTING RESULTS
##############################################################################

### Plot setting specifics
plt.close('all')
plt.rc('text', usetex=True)
textsize = 10
lw = 3

plt.figure(figsize=[5,3.5])
ax=plt.subplot(1,1,1)

ax.plot([D1.D,D1.D],[0,10],'k--',lw = 1)
ax.plot(arg,eps_total,c='C0',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{total})$')
ax.plot(arg,eps_til,c='C1',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{til})$',zorder = 5)
ax.plot(arg,eps_vw,c='C5',ls = '-',lw =lw,label = r'$\epsilon(Q_\textup{vw})$')    
ax.plot(arg,eps_mu,c='C2',ls = '-',lw =lw,label = r'$\epsilon(\mu)$')

ax.set_xlabel(r'$D$ [m]')
ax.set_ylabel('Relative difference $\epsilon$ [%]')
ax.set_ylim([0,7]) 
ax.set_xlim([0.25,3.5]) 
ax.text(0.1,0.9,r'$t = 0$', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
ax.legend(loc = 'upper right',fontsize = textsize)   
ax.tick_params(axis="both",which="major",labelsize=textsize)
ax.grid(True)

plt.tight_layout()
# plt.savefig('../results/Fig_S07_rel_diff_D',dpi=300)   
plt.savefig('../results/Fig_S07_rel_diff_D.pdf')   
