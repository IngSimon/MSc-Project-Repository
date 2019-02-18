###Faraday depth spectra for a Faraday thick source

###MeerKAT


import os,sys

import numpy as np

import pylab as pl

import rm_tools as R



# -----------------------------------------------------------------



def get_1d_data(input_dir):



        files = os.listdir(input_dir)

        #if (len(files)!=3):

           # print "Wrong number of input files in directory"

            #return



        nu_file = input_dir+"/frequencies.txt"

        q_file = input_dir+"/stokesq.txt"

        u_file = input_dir+"/stokesu.txt"



        nu = np.loadtxt(nu_file, comments="#")

        stokesQ = np.loadtxt(q_file, comments="#")

        stokesU = np.loadtxt(u_file, comments="#")



        los = stokesQ + 1j*stokesU



        return nu,los



# -----------------------------------------------------------------



def calc_phi_res(nu):



        C2 = 8.98755179e16

        nus = np.sort(nu)

        delta_l2 = C2 * (nus[0] ** (-2) - nus[-1] ** (-2))



        res = 2. * np.sqrt(3) / delta_l2



        return res



# -----------------------------------------------------------------



def calc_maxscale(nu,dnu):



        C2 = 8.98755179e16

        nus = np.sort(nu)

        delta_l2 = C2 * (nus[0] ** (-2) - nus[-1] ** (-2))



        l2min = 0.5 * C2 * ((nus[-1] + dnu) ** (-2)

                        + (nus[-1] - dnu) ** (-2))



        maxscale = np.pi / l2min



        return maxscale



# -----------------------------------------------------------------

# -----------------------------------------------------------------



const_c = 3e8



p0 = 1.0

alpha = 0.7

nu0 = 1.4e9

rm = 50.

phi0 = 0.




#nu = np.linspace(120e6,180e6,1024) # LOFAR frequency coverage

nu = np.linspace(580e6,250e7,1024) # MeerKAT frequency coverage

#nu = np.linspace(0.3e9,2.5e9,1024) # frequency coverage



lam = const_c/nu

lam_squared = lam**2  ###added

noise = np.random.normal(0, 0.001, nu.shape)

dataQ = np.sin(2*2*lam_squared)/(2*2*lam_squared) + 0.25*np.cos(2*10*lam_squared)

#dataQ = np.sin(2*2*lam_squared)/(2*2*lam_squared) + 0.25*np.cos(2*10*lam_squared)

stokesQ = dataQ + noise

dataU = 0.25*np.sin(2*10*lam_squared)

#dataU = 0.25*np.sin(2*10*lam_squared)

stokesU= dataU + noise

#chi_init = np.degrees(0.5*np.arctan2(stokesU,stokesQ))

#chi =chi_init + phi0


pl.subplot(111)

pl.plot(nu,dataQ)

pl.plot(nu,dataU)

pl.xlabel('0.58 to 2.5GHz, RM=50')   # string must be enclosed with quotes '  '

pl.ylabel('Q and U ')

pl.title('Stokes parameters against Frequency')

pl.legend(['stokesQ','stokesU'])

pl.savefig('plot1.png')

pl.show()



pl.subplot(111)

#pl.plot(nu,dataQ)

pl.plot(nu,stokesQ)

#pl.plot(nu,dataU)

pl.plot(nu,stokesU)

pl.xlabel('0.58 to 2.5GHz, RM=50')   # string must be enclosed with quotes '  '

pl.ylabel('Q and U ')

pl.legend(['stokesQ','stokesU'])

pl.savefig('plot2.png')

pl.show()



np.savetxt("./data/frequencies.txt",nu)

np.savetxt("./data/stokesq.txt",stokesQ)

np.savetxt("./data/stokesu.txt",stokesU)



# specify the range of fd space:

phi = np.linspace(-1000,1000,4000)



# get the input data:

inputdir = "./data"

nu,los = get_1d_data(inputdir)

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

pl.subplot(111)

pl.plot(rms.rmsf_phi,np.abs(rms.rmsf))

pl.axis([-1000,1000,-0.1,1.1])

pl.xlabel('RMSF ($\phi$)')   # string must be enclosed with quotes '  '
pl.ylabel('RMSF')

pl.savefig('plot3.png')
pl.show()



# run the RM synthesis transform on the data:

fd_spec = rms.compute_dirty_image(los)



# plot the output:

pl.subplot(111)
 
pl.plot(phi,np.abs(fd_spec))

pl.xlabel('$\phi$ [rad $m^{-2}$]')

pl.ylabel('F ($\phi$)')

pl.axis([-1000,1000,-0.1,1.5])

pl.savefig('plot4.png')

pl.show()

