###Faraday depth spectra for a Faraday thick source 

###


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



nu = np.linspace(120e6,240e6,1024) # LOFAR frequency coverage. From the 
#specifications LOFAR has standard 30-80MHz and 110-240MHz observing frequencies

lam = const_c/nu

lam_squared = lam**2  ###added

noise = np.random.normal(0, 0.0015, nu.shape) #estimate from Lofar sensitivity calc

dataQ = np.sin(2*2*lam_squared)/(2*2*lam_squared) + 0.25*np.cos(2*10*lam_squared)

stokesQ = dataQ + noise

dataU = 0.25*np.sin(2*10*lam_squared)

stokesU= dataU + noise


pl.subplot(111)

pl.plot(nu,dataQ)

pl.plot(nu,dataU)

pl.xlabel('frequency 120MHz to 240MHz')   ###changed to lofar

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

pl.xlabel('frequency (120MHz to 240 MHz)')   

pl.ylabel('Q and U ')

#pl.legend(['dataQ','stokesQ','dataU','stokesU'])

pl.legend(['stokesQ','stokesU'])

pl.savefig('plot2.png')

pl.show()



np.savetxt("./data/frequencies.txt",nu)

np.savetxt("./data/stokesq.txt",stokesQ)

np.savetxt("./data/stokesu.txt",stokesU)



# specify the range of fd space:

phi = np.linspace(-100,100,4000) ###changed to lofar



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

pl.axis([-100,100,-0.1,1.1])   ###changed to lofar

pl.xlabel('RMSF $\phi$ [rad $m^{-2}$]')  

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

pl.axis([-20,40,-0.1,0.4]) ###changed to lofar

pl.savefig('plot4.png')

pl.show()

