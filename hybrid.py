"""

Faraday depth spectra for a Faraday thick source 

Given an input file, the script removes user specified number of lines

The script generates a new file with a reduced number of entries

The scripts also creates a directory with the same name as files

The new generated file and the output images are stored in the new directory

Some rows of data have been deliberately removed to mimic RFI flagging


"""


import os,sys

import random

import numpy as np

import pylab as pl

import rm_tools as R


#----------------------------------------------------------------

def trim_data(num1, filename2, num2= 1024, filename1='./data/all_params.txt'):


        with open(filename1) as file:

           lines = file.read().splitlines()

        random_lines = random.sample(lines, num1) 

        with open(filename2, "w") as output_file:  

           output_file.writelines(line + '\n' for line in lines if line not in random_lines)

           output_file.close()  

        num_of_lines = num2 - num1

        return num_of_lines


# -----------------------------------------------------------------



def get_1d_data(input_dir):



        files = os.listdir(input_dir)


        params_file = './' + name2 + '/' +name2 + '.txt'

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


num1 = input('Please enter the number of lines to remove: ')

name2 = input('Please enter the name of the output file (in quotes): ')

os.makedirs(name2)

output_dir = './' + name2 + '/'

filename2 = './' + name2 + '/' +name2 + '.txt'

print 'A file {} with {} rows has been successfully generated'.format(filename2, trim_data(num1, filename2, num2= 1024, filename1='./data/all_params.txt'))

nu = np.loadtxt( filename2, usecols=(0,))

stokesQ = np.loadtxt( filename2, usecols=(1,))

stokesU = np.loadtxt(filename2, usecols=(2,))

 
#--------------------------------------------------------------------------

pl.subplot(111)

pl.plot(nu,stokesQ)

pl.plot(nu,stokesU)

pl.xlabel('frequency (0.58GHz to 2.5GHz)')  

pl.ylabel('Q and U ')

pl.legend(['stokesQ','stokesU'])

pl.savefig(output_dir +'plot1.png')

pl.show()

#------------------------------------------------------------------------
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

pl.savefig(output_dir + 'plot2.png')

pl.show()



# run the RM synthesis transform on the data:

fd_spec = rms.compute_dirty_image(los)



# plot the output:

pl.subplot(111)
 
pl.plot(phi,np.abs(fd_spec))

pl.xlabel('$\phi$ [rad $m^{-2}$]')

pl.ylabel('F ($\phi$)')

pl.axis([-1000,1000,-0.1,1.5])

pl.savefig(output_dir +'plot3.png')

pl.show()
