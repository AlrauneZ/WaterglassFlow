#!/usr/bin/env python3
"""
Script reproducing Fig. S1 of the Supporting Information of Paper 
>Groundwater flow below construction pits and erosion of temporary horizontal layers of silicate grouting<
by Joris M. Dekker, Thomas Sweijen, Alraune Zech; Hydrogeology Journal
https://doi.org/10.1007/s10040-020-02246-3

@author: A. Zech
"""

import numpy as np
import matplotlib.pyplot as plt

##############################################################################
### LOAD DATA AND SPECIFY SETTINGS
##############################################################################

domain={'x_0':0,'x_L':50,'z_0':0,'z_L':10,'z_1':1,'z_2':2,'nx':500,'nz':100}
velocity_x = np.loadtxt('../data/velocity_x_tinf.csv',delimiter = ',')
velocity_z = np.loadtxt('../data/velocity_z_tinf.csv',delimiter = ',')

x1D=np.linspace(domain['x_0'],domain['x_L'],domain['nx'],endpoint=True)
z1D=np.linspace(domain['z_0'],domain['z_L'],domain['nz'],endpoint=True)
xxs,zzs=np.meshgrid(x1D,z1D)

##############################################################################
### PLOTTING RESULTS
##############################################################################

### Plot setting specifics
plt.close('all')
plt.rc('text', usetex=True)
textsize = 10
        
plt.figure(1,figsize=[7.5,3])
cf=plt.streamplot(xxs, zzs, velocity_x, velocity_z,color = velocity_x, density=1,cmap='Blues')
cbar = plt.colorbar(cf.lines,format='%.3f')
cbar.set_label('velocity [m/d]',fontsize = textsize)
plt.xlabel('$x$ [m]',fontsize = textsize), plt.ylabel('$y$ [m]',fontsize = textsize)
plt.ylim([0,10])
plt.xlim([0,50])
plt.tick_params(axis="both",which="major",labelsize=textsize)

plt.tight_layout()
# plt.savefig('../results/Fig_S01_streamlines.png',dpi=300)   
plt.savefig('../results/Fig_S01_streamlines.pdf')   
