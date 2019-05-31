###Script for performaing RM synthesis after data predicting missing data

###Script optimized for stokes parameters predicted at same frequency as original data. (not at evenly space wavelength-squared points)

###Script for use with data generated using mlscript12.py




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


        nu_file = input_dir + "frequencies.txt"
        
        lam2_file = input_dir + "wavelengths.txt" 

        q_file = input_dir + "stokesQ.txt"

        u_file = input_dir + "stokesU.txt"



        nu = np.loadtxt(nu_file, comments="#")

        lam2 = np.loadtxt(lam2_file, comments="#")

        stokesQ = np.loadtxt(q_file, comments="#")

        stokesU = np.loadtxt(u_file, comments="#")



        los = stokesQ + 1j*stokesU



        return nu,lam2, stokesQ, stokesU, los



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

name_dir = input('Please enter the name of the directory containing data (use quotes): ') #

input_dir = './' + name_dir + "/complete/" #

nu,lam2, stokesQ, stokesU, los = get_1d_data(input_dir)



pl.subplot(111)

pl.plot(lam2,stokesQ,linestyle='none', marker='.')

pl.plot(lam2,stokesU, linestyle='none', marker='.')

pl.xlabel('Wavelengths squared')   # string must be enclosed with quotes '  '

pl.ylabel('Q and U ')

pl.title('Stokes parameters against Frequency')

pl.legend(['stokesQ','stokesU'])

pl.savefig('./' + name_dir + '/' + '3a.png')

pl.show()


# specify the range of fd space:

phi = np.linspace(-1000,1000,4000)



# get the input data:

#inputdir = "./data"

#nu,los = get_1d_data(inputdir)

dnu = nu[1]-nu[0]



# initialise the pyrmsynth class:

rms = R.RMSynth(nu,dnu,phi)



# calculate the resolution in fd space and the maximum recoverable scale:

res = calc_phi_res(nu)

maxscale = calc_maxscale(nu,dnu)

print ("\n")

print ("Max f.d. resolution: " +str(round(res)) + " rad/m^2")

print ("Max f.d. scale " +str(round(maxscale)) + " rad/m^2")

print ("\n")



# plot the RMTF:

pl.subplot(111)

pl.plot(rms.rmsf_phi,np.abs(rms.rmsf))

pl.axis([-1000,1000,-0.1,1.1])

pl.xlabel('RMSF ($\phi$)')   # string must be enclosed with quotes '  '
pl.ylabel('RMSF')

pl.savefig('./' + name_dir + '/' + '3b.png')

pl.show()



# run the RM synthesis transform on the data:

fd_spec = rms.compute_dirty_image(los)


fd_spec_final = fd_spec

np.save('./' + name_dir + '/' + 'fd_spec_final.npy',fd_spec_final)


# plot the output:

pl.subplot(111)
 
pl.plot(phi,np.abs(fd_spec))

pl.xlabel('$\phi$ [rad $m^{-2}$]')

pl.ylabel('F ($\phi$)')

pl.axis([-1000,1000,-0.1,1.5])

pl.savefig('./' + name_dir + '/' + '3c.png')

pl.show()





#---------------------------------------------------------------------------
#compare faraday depth spectra

fd_spec_file = name_dir + "/fd_spec_initial.npy"

fd_spec_initial = np.load(fd_spec_file)

#Get the difference between the initial and final faraday depth spectra
dif_fd_spec = fd_spec_initial - fd_spec_final

np.save('./' + name_dir + '/' + 'fd_spec_dif.npy',dif_fd_spec)

#----------------------------------------------------------------------------
#plot the difference

fig1 = pl.figure()

frame1=fig1.add_axes((.1,.3,.8,.7)) #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]

pl.plot(phi,np.abs(fd_spec_final), 'r--') #Faraday spectra after data imputation  

pl.plot(phi,np.abs(fd_spec_initial), 'k')  #Initial faraday spectra

pl.legend(['fd_spec_final','fd_spec_initial'])

pl.ylabel('Faraday depth spectra')

frame1.set_xticklabels([]) #Remove x-tic labels for the first frame

pl.grid()

#Residual plot

frame2=fig1.add_axes((.1,.1,.8,.2))  
      
pl.plot(phi,np.abs(dif_fd_spec),color = 'red')

pl.savefig('./' + name_dir + '/' + '3d.png')

pl.ylabel('Residuals')

pl.xlabel('$\phi$')

pl.show()



















