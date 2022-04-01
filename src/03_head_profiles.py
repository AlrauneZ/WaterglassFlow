#!/usr/bin/env python3
"""
Script reproducing Fig. 3 of the Paper 
>Groundwater flow below construction pits and erosion of temporary horizontal layers of silicate grouting<
by Joris M. Dekker, Thomas Sweijen, Alraune Zech; Hydrogeology Journal
https://doi.org/10.1007/s10040-020-02246-3

@author: A. Zech
"""

import numpy as np
import matplotlib.pyplot as plt

##############################################################################
### LOAD DATA
##############################################################################

data_head_t0 = np.loadtxt('../data/data_heads_layer_t0.csv',delimiter = ',', skiprows =1)
data_head_tend = np.loadtxt('../data/data_heads_layer_tend.csv',delimiter = ',', skiprows =1)

##############################################################################
### PLOTTING RESULTS
##############################################################################

### Plot setting specifics
plt.rc('text', usetex=True)
textsize = 10
lw = 3
# import ufz
# cc1 = ufz.get_brewer('Paired12', rgb=True,reverse=True)
# c1,c2,c3,c4  = cc1[9],cc1[8],cc1[3],cc1[2]

c1,c2,c3,c4  = 'lightgreen','darkgreen','thistle','purple'

plt.figure(figsize=[7.5,3])

ax=plt.subplot(1,2,1)
ax.plot(data_head_t0[1:-1,0],data_head_t0[1:-1,4],c=c1,ls = '-',lw =lw+0.5,label = r'$h(x)$ above til, sim')
ax.plot(data_head_t0[:,0],data_head_t0[:,2],c=c2,ls = '--',lw =lw,label = r'$h(x)$ above til, analyt')
ax.plot(data_head_t0[1:-1,0],data_head_t0[1:-1,3],c=c3,ls = '-',lw =lw+0.5,label = r'$h(x)$ below til, sim')
ax.plot(data_head_t0[:,0],data_head_t0[:,1],c=c4,ls = '--',lw =lw,label = r'$h(x)$ below til, analyt')
ax.set_xlabel(r'$x$ [m]')
ax.set_ylabel('Hydraulic head $h(x)$ [m]')
ax.grid(True)
ax.set_xlim([0,50]) 
ax.set_ylim([0.268,0.289]) 
ax.tick_params(axis="both",which="major",labelsize=textsize)
ax.set_yticks([0.27,0.275,0.28,0.285])
ax.text(0.1,0.1,r'a) $t = 0$', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
ax.tick_params(axis="both",which="major",labelsize=textsize)

ax=plt.subplot(1,2,2)
ax.plot(data_head_tend[1:-1,0],data_head_tend[1:-1,4],c=c1,ls = '-',lw =lw+0.5,label = r'$h(x)$ above til, sim')
ax.plot(data_head_tend[:,0],data_head_tend[:,2],c=c2,ls = '--',lw =lw,label = r'$h(x)$ above til, analyt')
ax.plot(data_head_tend[1:-1,0],data_head_tend[1:-1,3],c=c3,ls = '-',lw =lw+0.5,label = r'$h(x)$ below til, sim')
ax.plot(data_head_tend[:,0],data_head_tend[:,1],c=c4,ls = '--',lw =lw,label = r'$h(x)$ below til, analyt')
ax.set_xlabel(r'$x$ [m]')
ax.tick_params(axis="both",which="major",labelsize=textsize)
ax.grid(True)
ax.set_xlim([0,50]) 
ax.set_ylim([0.268,0.289]) 
ax.set_yticks([0.27,0.275,0.28,0.285])
ax.text(0.1,0.1,r'b) $t =  \infty$', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
ax.legend(loc = 'upper right',fontsize = textsize-1)
ax.tick_params(axis="both",which="major",labelsize=textsize)

plt.tight_layout()
# plt.savefig('../results/Fig03_head_profiles.png',dpi=300)   
plt.savefig('../results/Fig03_head_profiles.pdf')   
