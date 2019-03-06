###Faraday depth spectra for a Faraday thick source 

###Create a single file with all parameters

###https://stackoverflow.com/questions/36455083/working-with-nan-values-in-matplotlib   

###Not using line style

import os,sys

import numpy as np

import pylab as pl

import scipy.optimize as op

import george

from george import kernels


#import rm_tools as R
#-----------------------------------------------------------------------------------------------------------------------

def read_data(input_dir, filename1):
        
       params_file = input_dir + "/" + filename1 

       nu = np.loadtxt(params_file, usecols=(0,))

       lam_squared = np.loadtxt(params_file, usecols=(1,))

       stokesQ = np.loadtxt(params_file, usecols=(2,))

       stokesU = np.loadtxt(params_file, usecols=(3,)) 
        

       los = stokesQ + 1j*stokesU

       #return nu, lam_squared, stokesQ, stokesU

       return lam_squared, stokesQ, stokesU

#-----------------------------------------------------------------------------------------------------------------------

#read original data

def original_data( input_dir, filename2="all_params.txt"):

       lam_squared, stokesQ, stokesU = read_data(filename2)

       return  lam_squared, stokesQ, stokesU

#----------------------------------------------------------------------------------------------------------------------    

#read deleted 
def removed_data(input_dir_2, filename3="removed.txt"):

       lam_squared, stokesQ, stokesU = read_data(input_dir_2, filename3)

       deleted_lam = lam_squared

       deleted_Q = stokesQ

       deleted_U = stokesU

       return deleted_lam, deleted_Q, deleted_U


#-----------------------------------------------------------------------------------------------------------------------

def predict_Q_data(input_dir_2, filename4 = "training.txt"):

       lam_squared, stokesQ, stokesU = read_data(input_dir_2, filename4)

       stokesQ_R = stokesQ
       
       lam_R = lam_squared


       # Squared exponential kernel
       k1 = 0.3**2 * kernels.ExpSquaredKernel(0.02**2)

       # periodic covariance kernel with exponential component toallow decay away from periodicity

       k2 = 0.6**2 * kernels.ExpSquaredKernel(0.5**2) * kernels.ExpSine2Kernel(gamma=2/2.5**2, log_period=0.0) 
       ###vary gamma value to widen the    funnel

       # rational quadratic kernel for medium term irregularities.

       k3 = 0.3**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.1), metric=0.1**2)

       # noise kernel: includes correlated noise & uncorrelated noise

       k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2) 


       kernel = k1+ k2 #+k3 + k4


       gp = george.GP(kernel, mean=np.mean(stokesQ), fit_mean=True, white_noise=np.log(0.02**2), fit_white_noise=True)
       #gp = george.GP(kernel)

       gp.compute(lam_R)


       # range of times for prediction:
       #x = np.linspace(0.18, 0.21998, 360)
       x = x_values(input_dir_2)
 
       # calculate expectation and variance at each point:
       mu, cov = gp.predict(stokesQ_R, x)
       #mu, cov = gp.predict(stokesQ_2Gz, x, return_var = True)

       std = np.sqrt(np.diag(cov))
       #std = np.sqrt(cov)

       return  lam_R, stokesQ_R, mu, std

#------------------------------------------------------------------------------------------------------------------------

def predict_U_data(input_dir_2, filename4 = "training.txt"):

       lam_squared, stokesQ, stokesU = read_data(input_dir_2, filename4)

       stokesU_R = stokesU
       
       lam_R = lam_squared


       # Squared exponential kernel

       k1 = 0.3**2 * kernels.ExpSquaredKernel(0.02**2)
       
       # periodic covariance kernel with exponential component to allow decay away from periodicity

       k2 = 0.6**2 * kernels.ExpSquaredKernel(0.5**2) * kernels.ExpSine2Kernel(gamma=2/2.5**2, log_period=0.0) 

       # rational quadratic kernel for medium term irregularities.
       #k3 = 0.3**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.1), metric=1.2**2)
       k3 = 0.3**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.1), metric=0.1**2)

       # noise kernel: includes correlated noise & uncorrelated noise

       k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2) 

       kernel = k1 + k2 #+k3 + k4

       gp = george.GP(kernel, mean=np.mean(stokesU), fit_mean=True, white_noise=np.log(0.02**2), fit_white_noise=True)
       #gp = george.GP(kernel)

       gp.compute(lam_R)


       # range of times for prediction:
       #x = np.linspace(0.18, 0.21998, 360)
       x = x_values(input_dir_2)
 
       # calculate expectation and variance at each point:
       mu1, cov = gp.predict(stokesU_R, x)
       #mu, cov = gp.predict(stokesQ_2Gz, x, return_var = True)

       std1 = np.sqrt(np.diag(cov))
       #std = np.sqrt(cov)

       #return  stokesU_R, lam_R, x, mu1, std1

       return  stokesU_R, mu1, std1
#--------------------------------------------------------------------------------------------------------------------------

def x_values( input_dir_2, filename5 = "removed.txt"):

       lam_squared, stokesQ, stokesU = read_data(input_dir_2, filename5)
       
       x = lam_squared  

       return x


#-------------------------------------------------------------------------------------------------------------------------
name_dir = input('Please enter the name of the directory containing training and deleted data: ')

input_dir= "./data"

input_dir_2 = './' + name_dir 

lam_squared, stokesQ, stokesU = read_data(input_dir, "all_params.txt")

pl.subplot(111)

pl.plot(lam_squared,stokesQ, color='royalblue')

pl.plot(lam_squared,stokesU, color='darkorange')

pl.xlabel('$\lambda^2 [m^2]$')   

pl.ylabel('stokes parameters')

pl.axis([0.,0.3,-1.,1.5])

#pl.legend(['stokesQ','stokesU'])

pl.savefig(input_dir_2 + '/' +'ml8a.png')

pl.show()


#------------------------------------------------------------------------------------------------------------------------

lam_R, stokesQ_R, mu, std  = predict_Q_data(input_dir_2)

stokesU_R, mu1, std1  = predict_U_data(input_dir_2)

x = x_values(input_dir_2)


ax = pl.subplot(111)
 
# plot the original values

pl.plot(lam_R, stokesQ_R, color='royalblue', linestyle='none', marker='.')

pl.plot(lam_R, stokesU_R, color='darkorange', linestyle='none', marker='.')

#pl.plot(lam_R, stokesQ_R,  linestyle='none', marker='o')

pl.plot(x, mu, ls=':',color = 'red', linestyle='none', marker='.')

pl.plot(x, mu1, ls=':', color = 'red', linestyle='none', marker='.')

# shade in the area inside a one standard deviation bound:

ax.fill_between(x,mu-std,mu+std,facecolor='lightgrey', lw=0, interpolate=True)

ax.fill_between(x,mu1-std1,mu1+std1,facecolor='lightgrey', lw=0, interpolate=True)

pl.ylabel("Stokes parameters")

pl.xlabel('$\lambda^2 [m^2]$')

pl.title("Stokes parameters - Initial Prediction")

pl.axis([0., 0.3,-0.75,1.5])

pl.grid()

pl.savefig(input_dir_2 + '/' +'ml8b.png')

pl.show()

#-------------------------------------------------------------------------------------------------------------------------

# plot the full dataset

deleted_lam, deleted_Q, deleted_U = removed_data(input_dir_2)

ax = pl.subplot(111)

pl.plot(lam_R,stokesQ_R, color='royalblue', linestyle='none', marker='.')

pl.plot(lam_R,stokesU_R, color='darkorange', linestyle='none', marker='.')

pl.plot(x, mu, ls=':', color = 'red', linestyle='none', marker='.')

pl.plot(x, mu1, ls=':', color = 'red', linestyle='none', marker='.')

ax.fill_between(x,mu-std,mu+std,facecolor='lightgrey', lw=0, interpolate=True)

ax.fill_between(x,mu1-std1,mu1+std1,facecolor='lightgrey', lw=0, interpolate=True)

pl.plot(deleted_lam, deleted_Q,color='royalblue', linestyle='none', marker='.') 

pl.plot(deleted_lam, deleted_U,color='darkorange', linestyle='none', marker='.') 
 
pl.ylabel("Stokes parameters")

pl.xlabel('$\lambda^2 [m^2]$')

pl.title("Comparison")

pl.axis([0., 0.3,-0.75,1.5])

pl.grid()

pl.savefig(input_dir_2 + '/' +'ml8c.png')

pl.show()







