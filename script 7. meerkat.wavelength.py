###Faraday depth spectra for a Faraday thick source 

###Create a single file with all parameters

###Plotting stokes parameters against wavelength squared 

###changed the values of phi0 and phi1



import os,sys

import numpy as np

import pylab as pl

import rm_tools as R



# -----------------------------------------------------------------



def get_1d_data(input_dir):



        files = os.listdir(input_dir)


        nu_file = input_dir+"/frequencies.txt"

        lam_file = input_dir+"/wavelengths.txt"

        q_file = input_dir+"/stokesq.txt"

        u_file = input_dir+"/stokesu.txt"
        
        params_file = input_dir+"/all_params.txt" 



        nu = np.loadtxt(params_file, usecols=(0,))

        lam_squared = np.loadtxt(params_file, usecols=(1,))

        stokesQ = np.loadtxt(params_file, usecols=(2,))

        stokesU = np.loadtxt(params_file, usecols=(3,)) 
        

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

phi1= 50 

phi_fg = 10




nu = np.linspace(580e6,250e7,1024) # MeerKAT frequency coverage

lam = const_c/nu

lam_squared = lam**2  

noise = np.random.normal(0, 0.0015, nu.shape) #estimate from Lofar sensitivity calc

dataQ = np.sin(2*phi_fg*lam_squared)/(2*phi_fg*lam_squared) + 0.25*np.cos(2*phi1*lam_squared)

stokesQ = dataQ + noise


dataU = 0.25*np.sin(2*phi1*lam_squared)

stokesU= dataU + noise


#--------------------------------------------------------------------------

pl.subplot(111)

pl.plot(lam_squared,stokesQ)

pl.plot(lam_squared,stokesU)

pl.xlabel('$\lambda^2 [m^2]$')  

pl.ylabel('Stokes parameters ')

pl.legend(['stokesQ','stokesU'])

pl.savefig('plot1.png')

pl.show()

#------------------------------------------------------------------------

np.savetxt("./data/frequencies.txt",nu)

np.savetxt("./data/wavelengths.txt",lam_squared)

np.savetxt("./data/stokesq.txt",stokesQ)

np.savetxt("./data/stokesu.txt",stokesU)


#------------------------------------------------------------------------
#combine the three parameters into one file

with open("./data/all_params.txt", 'w') as file5:

  with open("./data/frequencies.txt", 'r') as file1:
    
    with open("./data/wavelengths.txt", 'r') as file2:

       with open("./data/stokesq.txt", 'r') as file3:

          with open("./data/stokesu.txt", 'r') as file4:

           

            for line1, line2, line3, line4  in zip(file1, file2, file3, file4):

                print >>file5, line1.strip(), line2.strip(),line3.strip(), line4.strip()




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

pl.xlabel('RMSF ($\phi$)')  

pl.ylabel('RMSF')

pl.savefig('plot2.png')

pl.show()



# run the RM synthesis transform on the data:

fd_spec = rms.compute_dirty_image(los)



# plot the output:

pl.subplot(111)
 
pl.plot(phi,np.abs(fd_spec))

pl.xlabel('$\phi$ [rad $m^{-2}$]')

pl.ylabel('F ($\phi$)')

pl.axis([-1000,1000,-0.1,1.5])

pl.savefig('plot3.png')

pl.show()

