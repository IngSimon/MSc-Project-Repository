###Faraday depth spectra for a Faraday thick source 

###Some rows of data have been deliberately removed to mimic RFI flagging

###create necessary files


import os,sys

import numpy as np

import pylab as pl

import rm_tools as R



# -----------------------------------------------------------------



def get_1d_data(input_dir):



        files = os.listdir(input_dir)


        params_file = input_dir+"/all_params.txt" 

        nu = np.loadtxt(params_file, usecols=(0,))

        stokesQ = np.loadtxt(params_file, usecols=(1,))

        stokesU = np.loadtxt(params_file, usecols=(2,)) 
        

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
#Retrive data from the file




nu = np.loadtxt("./data/all_params.txt", usecols=(0,))

stokesQ = np.loadtxt("./data/all_params.txt", usecols=(1,))

stokesU = np.loadtxt("./data/all_params.txt", usecols=(2,))


#--------------------------------------------------------------------------

pl.subplot(111)

pl.plot(nu,stokesQ)

pl.plot(nu,stokesU)

pl.xlabel('frequency (120MHz to 240 MHz)')   

pl.ylabel('Q and U ')

pl.legend(['stokesQ','stokesU'])

pl.savefig('plot1.png')

pl.show()

#------------------------------------------------------------------------


# specify the range of fd space:

phi = np.linspace(-100,100,4000) ###changed to lofar



# get the input data:

inputdir = "./data"

nu,los = get_1d_data(inputdir)

dnu = nu[1]-nu[0]


#---------------------------------------------------------------------
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

pl.savefig('plot2.png')

pl.show()


#------------------------------------------------------------------
# run the RM synthesis transform on the data:

fd_spec = rms.compute_dirty_image(los)



# plot the output:

pl.subplot(111)
 
pl.plot(phi,np.abs(fd_spec))

pl.xlabel('$\phi$ [rad $m^{-2}$]')

pl.ylabel('F ($\phi$)')

pl.axis([-20,40,-0.1,0.4]) ###changed to lofar

pl.savefig('plot3.png')

pl.show()

