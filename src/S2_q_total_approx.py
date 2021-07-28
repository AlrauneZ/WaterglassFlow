import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

from Class_Waterglas import q_total,fit_sum 

##############################################################################
### LOAD DATA AND SPECIFY SETTINGS
##############################################################################

rat_TL = np.arange(0,1,0.02)
c0,n = 0.92,20
c1, success = leastsq(fit_sum,c0,args = (rat_TL,n,0))
print(c1,success)

#c2 = c1
c2 = 0.9
q1 = q_total(rat_TL,approx = True, c1=c1)
q2 = q_total(rat_TL,approx = False)#,n=n)   

### For standard values:    
q1s,q2s = q_total(0.1,approx = True, c1=c2), q_total(0.1,approx = False)#,n=n)   
print('Absolute Difference: ',np.abs(q2s-q1s))
print('Relative Difference: ',np.abs((q2s-q1s)/q1s))

##############################################################################
### PLOTTING RESULTS
##############################################################################

### Plot setting specifics
plt.close('all')
plt.rc('text', usetex=True)
textsize = 10
lw = 3

plt.figure(figsize=[5,3])
plt.plot(rat_TL,q2,label = 'exact solution',lw = lw)
plt.plot(rat_TL,q1,'--',label = 'approximate solution',lw = lw)
plt.legend(fontsize = textsize)
plt.xlabel(r'$T/L$',fontsize = textsize)
plt.ylabel(r'$\hat C \cdot Q_\textup{total}$',fontsize = textsize)
plt.grid(True)
plt.tick_params(axis="both",which="major",labelsize=textsize)
plt.tight_layout()
# plt.savefig('../results/Fig04_q_trans.png',dpi=300)   
plt.savefig('../results/Fig_S02_q_total_approx.pdf')   
