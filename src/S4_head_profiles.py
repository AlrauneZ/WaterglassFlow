import numpy as np
import matplotlib.pyplot as plt

##############################################################################
### LOAD DATA AND SPECIFY SETTINGS
##############################################################################

head_profiles = np.loadtxt( '../data/data_headprofiles_t0_SI.csv',delimiter = ',', skiprows =1)

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

ax.plot(head_profiles[:,0],head_profiles[:,1],c='lightgreen',ls = '-',lw =lw+1,label = r'$h(x,y=H+D)$ (just above til)')
ax.plot(head_profiles[:,0],head_profiles[:,2],c='plum',ls = '-',lw =lw+1,label = r'$h(x,y=H)$ (just below til)')
ax.plot(head_profiles[:,0],head_profiles[:,3],c='k',ls = '--',lw = lw,label = r'$h(x,y=0)$ (bottom of domain)')

ax.set_xlabel(r'$x$ [m]')
ax.tick_params(axis="both",which="major",labelsize=textsize)
ax.grid(True)
ax.set_xlim([0,50]) 
ax.text(0.1,0.1,r'$t = 0$', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
ax.set_ylabel('Hydraulic head $h(x)$ [m]')
ax.legend(loc = 'upper right',fontsize = textsize)

plt.tick_params(axis="both",which="major",labelsize=textsize)
plt.tight_layout()
# plt.savefig('../results/Fig_S4_head_profiles.png',dpi=300)   
plt.savefig('../results/Fig_S4_head_profiles.pdf')   
