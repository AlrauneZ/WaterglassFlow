import numpy as np
import matplotlib.pyplot as plt

##############################################################################
### LOAD DATA AND SPECIFY SETTINGS
##############################################################################

head_t0 = np.loadtxt('../data/modlfow_head_t0.csv',delimiter = ',')
head_tinf = np.loadtxt('../data/modlfow_head_tinf.csv',delimiter = ',')

domain={'x_0':0,'x_L':50,'z_0':0,'z_L':10,'z_1':1,'z_2':1,'nx':500,'nz':100}
xcoords =np.linspace(domain['x_0'],domain['x_L'],head_t0.shape[1],endpoint=True)
zcoords =np.linspace(domain['z_L'],0,head_t0.shape[0],endpoint=True)

##############################################################################
### PLOTTING RESULTS
##############################################################################

### Plot setting specifics
plt.rc('text', usetex=True)
textsize = 20
cmap = plt.get_cmap('RdYlGn')#'PiYG') #
levels = np.linspace(0.268,0.288,40,endpoint=True)
figsize=[10.8,4.32]

plt.figure(figsize=figsize)
ax = plt.subplot(111)
cf = plt.contourf(xcoords[1:-1],zcoords[1:],head_t0[1:,1:-1],cmap=cmap,levels=levels)
cbar = plt.colorbar(cf,format='%.3f')
cbar.set_label('head [m]',fontsize = textsize)
cbar.set_ticks([0.268,0.272,0.276,0.28,0.284,0.288])
cbar.ax.tick_params(labelsize=textsize) 
plt.xlabel('$x$ [m]',fontsize = textsize)
plt.ylabel('$y$ [m]',fontsize = textsize)
plt.text(0.05,0.85, r'b) $t = 0$' ,fontsize = textsize, transform=ax.transAxes,bbox=dict(boxstyle='round', facecolor='w', alpha=0.9))
plt.plot(xcoords,domain['z_1']*np.ones_like(xcoords),c='0.1',ls='--',lw=1)
plt.plot(xcoords,(domain['z_1']+domain['z_2'])*np.ones_like(xcoords),c='0.1',ls='--',lw=1)
plt.ylim([0,4])
plt.xlim([0,50])
plt.yticks([0,1,2,3,4])
plt.tick_params(axis="both",which="major",labelsize=textsize)
plt.tight_layout()
# plt.savefig('../results/Fig02_head_contours_b.png',dpi=300)   
plt.savefig('../results/Fig02b_head_contours.pdf')   


plt.figure(figsize=figsize)
ax = plt.subplot(111)
cf = plt.contourf(xcoords[1:-1],zcoords[1:],head_tinf[1:,1:-1],cmap=cmap,levels=levels)
cbar = plt.colorbar(cf,format='%.3f')
cbar.set_label('head [m]',fontsize = textsize)
cbar.set_ticks([0.268,0.272,0.276,0.28,0.284,0.288])
cbar.ax.tick_params(labelsize=textsize) 
plt.xlabel('$x$ [m]',fontsize = textsize)
plt.ylabel('$y$ [m]',fontsize = textsize)
plt.text(0.05,0.85,r'd) $t =  \infty$',fontsize = textsize, transform=ax.transAxes,bbox=dict(boxstyle='round', facecolor='w', alpha=0.9))
plt.plot(xcoords,domain['z_1']*np.ones_like(xcoords),c='0.1',ls='--',lw=1)
plt.plot(xcoords,(domain['z_1']+domain['z_2'])*np.ones_like(xcoords),c='0.1',ls='--',lw=1)
plt.ylim([0,4])
plt.xlim([0,50])
plt.yticks([0,1,2,3,4])
plt.tick_params(axis="both",which="major",labelsize=textsize)
plt.tight_layout()
# plt.savefig('../results/Fig02_head_contours_d.png',dpi=300)   
plt.savefig('../results/Fig02d_head_contours.pdf')   
