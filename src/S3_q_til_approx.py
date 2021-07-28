import numpy as np
import matplotlib.pyplot as plt
from Class_Waterglas import q_total,q_til

##############################################################################
### LOAD DATA AND SPECIFY SETTINGS
##############################################################################

### Create instance of flux settings in construction pit (default parameter) 
D1 = cw.WaterGlassTransport()
D1.calculate_fluxes() ### calculate analytical solutions for fluxes within domain

rat_TL = np.arange(0,2.02,0.02)
q1 = q_total(rat_TL)#,approx = True, c1=c1)    
c1 = 0.92
rat_TL_standard = 0.1

##############################################################################
### PLOTTING RESULTS
##############################################################################

### Plot setting specifics
plt.close('all')
plt.rc('text', usetex=True)
textsize = 10
lw = 3


plt.figure(figsize=[6,4])
plt.plot(rat_TL,q1,c='k',lw = lw,label = r'$Q_\textup{total}$')

for ir,rat_HT in enumerate(np.arange(0.1,1,0.2)):
   
    q2 = q_til(rat_TL,rat_HT,approx = True, c1=c1)
    q3 = q_til(rat_TL,rat_HT,approx = False, n = 10)
    
    plt.plot(rat_TL,q3,'-',lw = lw,c = 'C{}'.format(ir),label = r'$Q_\textup{{til}}(H/T = {:.1f})$'.format(rat_HT))
    plt.plot(rat_TL[::5],q2[::5],ls = '',marker = '*',ms=8,c = 'C{}'.format(ir))

plt.legend(loc = 'upper center',ncol = 3,fontsize = textsize)
plt.grid(True)
plt.xlim([0,2])
plt.ylim([0,1.15])
plt.xlabel(r'$T/L$',fontsize = textsize)
plt.ylabel(r'$\hat C \cdot Q(T/L)$',fontsize = textsize)
plt.tick_params(axis="both",which="major",labelsize=textsize)
plt.tight_layout()
# plt.savefig('../results/S3_q_til_approx.png',dpi=300)   
plt.savefig('../results/S3_q_til_approx.pdf')   
