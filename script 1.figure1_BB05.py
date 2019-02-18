###recreating image in BB05 Fig 1


import os,sys

#import math

import numpy as np

from matplotlib import pyplot as pl 



# -----------------------------------------------------------------
#Define constants and variables

nu = np.linspace(0.3e9,5e9,5120) # frequency coverage

const_c = 3e8

lam = const_c/nu

lam_squared = lam**2 

phi1 = 10.0 #rad/m**2

Q_fg0 = 2 #rad/m**2

chi0 = 0.  #degrees


#------------------------------------------------------------------
#Generate data and noise

noise = np.random.normal(0, 0.001, nu.shape)

#stokesQ = np.sin(2*2*lam_squared)/(2*2*lam_squared) + 0.25*np.cos(2*10*lam_squared)

dataQ = np.sin(2*Q_fg0*lam_squared)/(2*Q_fg0*lam_squared) + 0.25*np.cos(2*phi1*lam_squared)

stokesQ = dataQ + noise

#stokesU = 0.25*np.sin(2*10*lam_squared)

dataU = 0.25*np.sin(2*phi1*lam_squared)

stokesU= dataU + noise


#.....................................................................
#calculate chi

chi = 0.5*np.arctan2(stokesU,stokesQ)

grad_chi = np.diff(chi)

imax = np.argmax(grad_chi)

chi[:imax+1]+=np.pi

chi_deg = np.degrees(chi)


#--------------------------------------------------------------------
#calculate Q_fg

Q_fg = np.sin(2*Q_fg0*lam_squared)/(2*Q_fg0*lam_squared)


#--------------------------------------------------------------------
#calculate ||p||


p_complex = stokesQ + 1j*stokesU

p_modulus = np.abs(p_complex)

#--------------------------------------------------------------------
#Plot chi, Q_fg, and ||p||


fig, ax1= pl.subplots()

ax1.set_xlabel('$\lambda^2 [m^2]$')

ax1.set_ylabel('$\chi [deg]$')

l1 =ax1.plot(lam_squared, chi_deg, 'k-')

ax1.tick_params(axis='y')

ax1.set_ylim ((-90, 350))

ax1.set_xlim ((0,1))


ax2 = ax1.twinx()

ax2.set_ylabel('Flux [Jy]')  


l2, l3 = ax2.plot(lam_squared, Q_fg, 'k--', lam_squared, p_modulus,'k-.' )

ax2.tick_params(axis='y')

fig.legend((l1, l2, l3),('$\chi$', '$Q_{fg}$','$||p||$' ),'right', bbox_to_anchor=(0.4, 0.6, 0.4, 0.4))

fig.savefig('Figure1.png')

pl.show()



