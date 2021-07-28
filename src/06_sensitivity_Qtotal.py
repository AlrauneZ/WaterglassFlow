import numpy as np
import matplotlib.pyplot as plt
from Class_Waterglas import WaterGlassTransport,q_total_trans #,q_til_trans

##############################################################################
### LOAD/Setup DATA
##############################################################################

### Create instance of flux settings in construction pit (default parameter) 
D1 = WaterGlassTransport()
D1.calculate_fluxes() ### calculate analytical solutions for fluxes within domain

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

# import ufz
# cc1 = ufz.get_brewer('Blues9', rgb=True,reverse=True)[::2]
cc1 = ['midnightblue','navy','cornflowerblue','lightblue']

plt.figure(figsize=[7.5,5.4])
for ii,tau in enumerate(tau_range):
    ax=plt.subplot(2,2,ii+1)
    for ir,rat_HT in enumerate(rat_HT_range):        
        rat_HL = rat_TL*rat_HT 
        q_tot_tl = q_total_trans(rat_TL,rat_HL,tau,tau0)
        # q_til_tl = q_til_trans(rat_TL,rat_HL,tau,tau0)
        # mu_tl = q_til_tl/q_tot_tl

        ax.plot(rat_TL,q_tot_tl,c=cc1[ir],ls = '--',lw =lw,label = r'$H/T = {}$'.format(rat_HT),zorder = 5-ir)
        ax.set_ylim([0,0.95])

    ax.text(0.05,0.85,r'{}) $\tau = {}$'.format(abc[ii],tau),transform=ax.transAxes,fontsize = textsize,bbox=dict(boxstyle='round',facecolor='w'))
    ax.set_xlim([0,1])
    ax.grid(True)
    ax.tick_params(axis="both",which="major",labelsize=textsize)

    if ii >1:
        ax.set_xlabel(r'$T/L$')
    if ii in [0,2]:
        ax.set_ylabel(r'$\hat C \cdot Q_\textup{total} (\tau,T,H,L)$',fontsize = textsize)
    if ii == 2:
        ax.legend(loc = 'lower right',fontsize = textsize)

plt.tight_layout()
# plt.savefig('../results/Fig06_sensitivity_Qtotal.png',dpi=300)   
plt.savefig('../results/Fig06_sensitivity_Qtotal.pdf')   
