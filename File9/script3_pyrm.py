#Data P_array_test.npy from Joe
#10000 data sets
#135 frequency channels per dataset
#Each dataset has 2 columns, stokes Q and U
#Script implements a for loop to perform rm synthesis for the 10000 datasets
#With no images
#Calculates time the loop takes

#Note the size of phi significantly determines runtime



import os,sys

import numpy as np

import time

from matplotlib import pyplot as pl

import scipy.constants 

import rm_tools as R

# -----------------------------------------------------------------



def calc_phi_res(nu):



        C = scipy.constants.c
        C2 = C**2
        #c2 = 8.98755179e16

        nus = np.sort(nu)

        delta_l2 = C2 * (nus[0] ** (-2) - nus[-1] ** (-2))



        res = 2. * np.sqrt(3) / delta_l2



        return res



# -----------------------------------------------------------------



def calc_maxscale(nu,dnu):


        C = scipy.constants.c
        C2 = C**2       

        #C2 = 8.98755179e16

        nus = np.sort(nu)

        delta_l2 = C2 * (nus[0] ** (-2) - nus[-1] ** (-2))



        l2min = 0.5 * C2 * ((nus[-1] + dnu) ** (-2)

                        + (nus[-1] - dnu) ** (-2))



        maxscale = np.pi / l2min



        return maxscale



# -----------------------------------------------------------------



channels = 135 

freq_min = 8.59079e8 

freq_max = 1.69933e9

freq = np.linspace(freq_min, freq_max, channels)

lamsq = (scipy.constants.c)**2 / (freq*freq)

dataset = np.load('P_array_test.npy')

nu = freq

rm_values = []

start = time.time()

for data in dataset:

  stokes_params = data

  stokes_Q = data[:, 0]

  stokes_U = data[:, 1]

   
  #----------------------------------------------------------------------
  #stokes parameters versus frequency
  """

  pl.title('Stokes parameters versus frequency')

  pl.plot (freq, stokes_Q, color = 'blue', linestyle = 'none', marker = '.')

  pl.plot (freq, stokes_U, color = 'orange', linestyle = 'none', marker = '.')

  pl.ylabel('Q and U')

  pl.xlabel('Frequency')

  pl.legend(['stokesQ','stokesU'])

  pl.show()

  #------------------------------------------------------------------------
  #stokes parameters versus wavelength

  pl.title('Stokes parameters versus wavelength squared')

  pl.plot (lamsq, stokes_Q, color = 'blue', linestyle = 'none', marker = '.')

  pl.plot (lamsq, stokes_U, color = 'orange', linestyle = 'none', marker = '.')

  pl.ylabel('Q and U')

  pl.xlabel('$\lambda^2$')

  pl.legend(['stokesQ','stokesU'])

  pl.show()

  """
  # specify the range of fd space:

  #phi = np.linspace(-200,200,4000)
  phi = np.linspace(-1000,1000,4000) #the size significantly determines runtime

  los = stokes_Q + 1j*stokes_U
  
  dnu = nu[1]-nu[0]

  # initialise the pyrmsynth class:

  rms = R.RMSynth(nu,dnu,phi) 


  # calculate the resolution in fd space and the maximum recoverable scale:

  res = calc_phi_res(nu)

  maxscale = calc_maxscale(nu,dnu)

  print "\n"

  print "Max f.d. resolution: " +str(round(res)) + " rad/m^2"

  print "Max f.d. scale " +str(round(maxscale)) + " rad/m^2"

  print "\n"
  
  # plot the RMTF:
  """

  pl.subplot(111)

  pl.title('RMSF')

  pl.plot(rms.rmsf_phi,np.abs(rms.rmsf))

  pl.axis([-1000,1000,-0.2,1.5])

  pl.xlabel('RMSF ($\phi$)')  

  pl.ylabel('RMSF')

  pl.show()

  """ 
  

  # run the RM synthesis transform on the data:

  fd_spec = rms.compute_dirty_image(los)

  #calculate rotation measure by determining the value of phi when fd_spec is max

  max_fd_spec = phi[np.abs(fd_spec).argmax()]

  rm_values.append(max_fd_spec)

  print "RM: {} rad/m^2".format(max_fd_spec)

  
  # plot the output:

  """

  pl.subplot(111)

  pl.title('Faraday depth profile')

  pl.plot(phi,np.abs(fd_spec))

  #pl.axis([-500,500,-0.00003,0.002])

  pl.xlabel('$\phi$ [rad $m^{-2}$]')

  pl.ylabel('F ($\phi$)')

  pl.show()

  """
   

np.savetxt('rm_pyrm.txt', rm_values)  #save in .txt format
np.save('rm_pyrm', rm_values)   #save file in .npy format

end = time.time()
time1 = end - start

print ("Runtime: {} seconds".format(time1))






  

 



