#!/usr/bin/env python3
"""
Script reproducing Fig. 7 of the Paper 
>Groundwater flow below construction pits and erosion of temporary horizontal layers of silicate grouting<
by Joris M. Dekker, Thomas Sweijen, Alraune Zech; Hydrogeology Journal
https://doi.org/10.1007/s10040-020-02246-3

@author: A. Zech
"""

import numpy as np
import matplotlib.pyplot as plt
from Class_Waterglas import WaterGlassTransport,q_til_trans #,q_total_trans

##############################################################################
### LOAD/Setup DATA
##############################################################################

### Create instance of flux settings in construction pit (default parameter) 
D1 = WaterGlassTransport()
### calculate analytical solutions for fluxes within domain
D1.calculate_fluxes() 

tau_range = [0.001,0.01,0.1,1]
tau0 =D1.settings['k_til_0'] /  D1.kf
rat_TL =  np.arange(0,1,0.01)
rat_HT_range = [0.05,0.1,0.25,0.5]

##############################################################################
### PLOTTING RESULTS
##############################################################################

### Plot setting specifics
plt.rc('text', usetex=True)
lw = 3
textsize = 10
abc = ['a','b','c','d']

cc2 = ['darkred','red','orange','gold']

plt.figure(figsize=[7.5,5.4])
for ii,tau in enumerate(tau_range):
    ax=plt.subplot(2,2,ii+1)
    for ir,rat_HT in enumerate(rat_HT_range):        
        rat_HL = rat_TL*rat_HT 
        # q_tot_tl = q_total_trans(rat_TL,rat_HL,tau,tau0)
        q_til_tl = q_til_trans(rat_TL,rat_HL,tau,tau0)
        # mu_tl = q_til_tl/q_tot_tl

        ax.plot(rat_TL,q_til_tl,c=cc2[ir],ls = '--',lw =lw,label = r'$H/T = {}$'.format(rat_HT),zorder = 5-ir)
        ax.set_ylim([0,0.95])

    ax.text(0.05,0.85,r'{}) $\tau = {}$'.format(abc[ii],tau),transform=ax.transAxes,fontsize = textsize,bbox=dict(boxstyle='round',facecolor='w'))
    ax.set_xlim([0,1])
    ax.tick_params(axis="both",which="major",labelsize=textsize)
    ax.grid(True)

    if ii >1:
        ax.set_xlabel(r'$T/L$')
    if ii in [0,2]:
        ax.set_ylabel(r'$\hat C \cdot Q_\textup{til} (\tau,T,H,L)$',fontsize = textsize)
    if ii == 0:
        ax.legend(loc = 'upper right',fontsize = textsize)

plt.tight_layout()
# plt.savefig('../results/Fig07_sensitivity_Qtil.png',dpi=300)   
plt.savefig('../results/Fig07_sensitivity_Qtil.pdf')   
