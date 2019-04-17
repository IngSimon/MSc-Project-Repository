"""

Faraday depth spectra for a Faraday thick source 

Given an input file, the script removes user specified number of lines

The script generates a new file with a reduced number of entries

The scripts also creates a directory with the same name as files


Some rows of data have been deliberately removed to mimic RFI flagging

The algorithm picks a random number of blocks of lines, randomly

The generated files and the output images are stored in the new directory

"""


import os,sys

import random

from random import randint

from random import sample

import numpy as np

import pylab as pl

import celerite

from celerite import terms

import rm_tools as R


def trim_data(percent, name2, filename2, filename3, filename4, num2= 1024, filename1="./data/all_params.txt"):

       maxwid = 50

       lines = np.loadtxt(filename1)

       deleted_entries = [] 
 
       clean = []

       train = []

       while True:
    
            # choose a random position and flagging width:
            pos = np.random.randint(0,len(lines))

            wid = np.random.randint(1,maxwid+1)

            # check end of bandpass:
            if (pos+wid)>num2:

               wid = (pos+wid)-num2

            print("flagging ",wid," channels starting with channel ",pos)
    
            # apply flagging

            selected_lines = lines[pos:pos+wid]

            selected_lines[:,2:4]= 0.0 

            for i in selected_lines:
                 
               deleted_entries.append(i)

            # check flagged percentage:       

            ndata = 1024 - len(deleted_entries)
           
            print (len(deleted_entries))

            perc = 1 -(float(ndata)/float(num2))
             
            print (perc) 
    
            # if flagged percentage exceeds specification: exit loop
            if perc>float(percent):

                print("Percentage flagged: ",perc)

                break

       np.savetxt(filename3, deleted_entries, delimiter=' ')   # X is an array

       for line in lines:

         clean.append(line)

       np.savetxt(filename4, clean, delimiter=' ')   # X is an array

       for line in lines:

         if (line[2]!= 0.0):

            train.append(line)
        
       np.savetxt(filename2, train, delimiter=' ')   # X is an array
       
       return perc

def create_removed(file2, file3, file1="./data/all_params.txt"):

       with open(file1) as infile:

           f1 = infile.readlines()

       with open(file2) as infile:

           f2 = infile.readlines()

       only_in_f1 = [i for i in f1 if i not in f2] 

       only_in_f2 = [i for i in f2 if i not in f1]

       with open(file3, 'w') as outfile:

           if only_in_f1:

             for line in only_in_f1:

               outfile.write(line)

           if only_in_f2:

             for line in only_in_f2:

               outfile.write(line)

       return file3



# -----------------------------------------------------------------



def get_1d_data(input_dir):



        files = os.listdir(input_dir)


        params_file = './' + name2 + '/' + "training" + '.txt'

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
#Retrive data from the file


percent = input('Please enter the percentage of entries to remove: ')

name2 = input('Please enter the name of the directory for the output files: ')

os.makedirs(name2)

input_dir = './' + name2 + '/'

filename2 = "./" + name2 + "/training.txt"

filename3 = "./" + name2 + "/deleted.txt"

filename4 = "./" + name2 + "/cleaned.txt"

print ('A file with {} percent of the original data has been successfully generated'.format(trim_data(percent, name2, filename2, filename3, filename4, num2= 1024, filename1="./data/all_params.txt")))

file2 = "./" + name2 +"/training.txt"

file3 = "./" + name2 +"/removed.txt"

print ("A file {} has been successfully created".format(create_removed(file2, file3, file1="./data/all_params.txt")))



nu = np.loadtxt( filename2, usecols=(0,))

lam_squared = np.loadtxt( filename2, usecols=(1,))

stokesQ = np.loadtxt( filename2, usecols=(2,))

stokesU = np.loadtxt(filename2, usecols=(3,))

 
#--------------------------------------------------------------------------

pl.subplot(111)

pl.plot(lam_squared,stokesQ, linestyle='none', marker='.')

pl.plot(lam_squared,stokesU, linestyle='none', marker='.')

pl.xlabel('$\lambda^2 [m^2]$')  

pl.ylabel('stokes parameters ')

pl.legend(['stokesQ','stokesU'])

pl.savefig(input_dir +'1a.png')

pl.show()

#------------------------------------------------------------------------
# specify the range of fd space:

phi = np.linspace(-1000,1000,4000)



# get the input data:

#inputdir = "./data"

nu,los = get_1d_data(input_dir)

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

pl.savefig(input_dir + '1b.png')

pl.show()



# run the RM synthesis transform on the data:

fd_spec = rms.compute_dirty_image(los)



# plot the output:

pl.subplot(111)
 
pl.plot(phi,np.abs(fd_spec))

pl.xlabel('$\phi$ [rad $m^{-2}$]')

pl.ylabel('F ($\phi$)')

pl.axis([-1000,1000,-0.1,1.5])

pl.savefig(input_dir +'1c.png')

pl.show()















