import numpy as np
import matplotlib.pyplot as plt
from Class_SimWaterglas import Sim_WaterGlassTransport
from Class_Waterglas import WaterGlassTransport

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
D1.calculate_fluxes() ### calculate analytical solutions for fluxes within domain

##############################################################################
### PLOTTING RESULTS
##############################################################################

### Plot setting specifics
plt.rc('text', usetex=True)
lw = 3
textsize = 10

plt.figure(figsize=[7.5,2.8])

ax=plt.subplot(1,2,1)
ax.plot(Sim1.tau,Sim1.mu,c='C2',ls = '-',lw =lw,label = r'$\mu(K_\textup{til}/K_\textup{sand})$ - sim')
ax.plot(tau,D1.mu_trans_scale(tau),c='C2',ls = '--',lw =lw,label = r'$\mu(\tau)$ - ana')
ax.set_xlabel(r'$\tau = K_\textup{til}/K_\textup{sand}$')
ax.grid(True)
ax.set_ylabel(r'Dilution ratio $\mu = Q_\textup{til}/Q_\textup{total}$')
ax.legend(loc = 'lower right',fontsize = textsize)
ax.text(-0.07,-0.15,r'a', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
ax.tick_params(axis="both",which="major",labelsize=textsize)

ax=plt.subplot(1,2,2)
ax.plot(Sim1.tau,Sim1.mu,c='C2',ls = '-',lw =lw,label = r'$\mu(K_\textup{til}/K_\textup{sand})$ - sim')
ax.plot(tau,D1.mu_trans_scale(tau),c='C2',ls = '--',lw =lw,label = r'$\mu(\tau)$ - ana')
ax.set_xlabel(r'$\tau = K_\textup{til}/K_\textup{sand}$')
ax.grid(True)
ax.set_xscale('log')
ax.text(-0.07,-0.15,r'b', fontsize=textsize,transform=ax.transAxes, bbox=dict(boxstyle='round',facecolor='w'))
ax.tick_params(axis="both",which="major",labelsize=textsize)

plt.tight_layout()
# plt.savefig('../results/Fig05_dilution_ratio.png',dpi=300)   
plt.savefig('../results/Fig05_dilution_ratio.pdf')   
