###Faraday depth spectra for a Faraday thick source 

###imputing missing data

###initial prediction

###optimize parameters and do second prediction
 

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

       return nu, lam_squared, stokesQ, stokesU

#-----------------------------------------------------------------------------------------------------------------------

#read original data

def original_data( input_dir_2, filename2="all_params.txt"):

       nu, lam_squared, stokesQ, stokesU = read_data(filename2)


       return  lam_squared, stokesQ, stokesU

#----------------------------------------------------------------------------------------------------------------------    

#read deleted 
def removed_data(input_dir_2, filename3="removed.txt"):

       nu, lam_squared, stokesQ, stokesU = read_data(input_dir_2, filename3)

       deleted_nu = nu

       deleted_lam = lam_squared

       deleted_Q = stokesQ

       deleted_U = stokesU

       return deleted_nu, deleted_lam, deleted_Q, deleted_U


#-----------------------------------------------------------------------------------------------------------------------

def predict_Q_data(input_dir_2, filename4 = "training.txt"):

       nu, lam_squared, stokesQ, stokesU = read_data(input_dir_2, filename4)

       nu_R = nu

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
       mu_initial, cov = gp.predict(stokesQ_R, x)
       #mu, cov = gp.predict(stokesQ_2Gz, x, return_var = True)

       std_initial = np.sqrt(np.diag(cov))

       print ("Initial prediction of missing Q values: \n {}".format(mu_initial))


       # Define the objective function (negative log-likelihood in this case).
       def nll(p):
              gp.set_parameter_vector(p)
              ll = gp.log_likelihood(stokesQ_R, quiet=True)
              return -ll if np.isfinite(ll) else 1e25

       # And the gradient of the objective function.
       def grad_nll(p):
              gp.set_parameter_vector(p)
              return -gp.grad_log_likelihood(stokesQ_R, quiet=True)


       gp.compute(lam_R)

       p0 = gp.get_parameter_vector()

       results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")


       # run optimization:
       #results = op.minimize(nll, p0, jac=grad_nll)


       gp.set_parameter_vector(results.x)


       x = x_values(input_dir_2)

       mu_optimized, cov = gp.predict(stokesQ_R, x)

       
       std_optimized = np.sqrt(np.diag(cov))

       print ("Final prediction of missing Q values: \n {}".format(mu_optimized))

       return  nu_R, lam_R, stokesQ_R, mu_initial, std_initial, mu_optimized, std_optimized


#------------------------------------------------------------------------------------------------------------------------

def predict_U_data(input_dir_2, filename4 = "training.txt"):

       nu, lam_squared, stokesQ, stokesU = read_data(input_dir_2, filename4)

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

       gp = george.GP(kernel, mean=np.mean(stokesU_R), fit_mean=True, white_noise=np.log(0.02**2), fit_white_noise=True)
       #gp = george.GP(kernel)

       gp.compute(lam_R)

       # range of times for prediction:
       x = x_values(input_dir_2)

       # calculate expectation and variance at each point:
       mu1_initial, cov = gp.predict(stokesU_R, x)
       #mu, cov = gp.predict(stokesQ_2Gz, x, return_var = True)

       std1_initial = np.sqrt(np.diag(cov))

       #return  stokesU_R, mu1, std1

       print ("Initial prediction of missing U values: \n {}".format(mu1_initial))


       # Define the objective function (negative log-likelihood in this case).
       def nll(p):
              gp.set_parameter_vector(p)
              ll = gp.log_likelihood(stokesU_R, quiet=True)
              return -ll if np.isfinite(ll) else 1e25

       # And the gradient of the objective function.
       def grad_nll(p):
              gp.set_parameter_vector(p)
              return -gp.grad_log_likelihood(stokesU_R, quiet=True)


       gp.compute(lam_R)

       p0 = gp.get_parameter_vector()

       results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")


       # run optimization:
       #results = op.minimize(nll, p0, jac=grad_nll)


       gp.set_parameter_vector(results.x)


       x = x_values(input_dir_2)

       mu1_optimized, cov = gp.predict(stokesU_R, x)

       std1_optimized = np.sqrt(np.diag(cov))

       print ("Final prediction of missing U values: \n {}".format(mu1_optimized))

       return  stokesU_R, mu1_initial, std1_initial, mu1_optimized, std1_optimized

#--------------------------------------------------------------------------------------------------------------------------

def x_values( input_dir_2, filename5 = "removed.txt"):

       nu, lam_squared, stokesQ, stokesU = read_data(input_dir_2, filename5)

       x = lam_squared  

       return x


#-------------------------------------------------------------------------------------------------------------------------
name_dir = input('Please enter the name of the directory containing data (no quotes): ')

input_dir= "./data"

input_dir_2 = './' + name_dir 

nu, lam_squared, stokesQ, stokesU = read_data(input_dir_2, "all_params.txt")

pl.subplot(111)

pl.plot(lam_squared,stokesQ, color='royalblue')

pl.plot(lam_squared,stokesU, color='darkorange')

pl.xlabel('$\lambda^2 [m^2]$')   

pl.ylabel('stokes parameters')

pl.axis([0.,0.3,-1.,1.5])

#pl.legend(['stokesQ','stokesU'])

pl.savefig(input_dir_2 + '/' +'2a.png')

pl.show()



#------------------------------------------------------------------------------------------------------------------------


nu_R, lam_R, stokesQ_R, mu_initial, std_initial, mu_optimized, std_optimized = predict_Q_data(input_dir_2)

stokesU_R, mu1_initial, std1_initial, mu1_optimized, std1_optimized = predict_U_data(input_dir_2)

x = x_values(input_dir_2)

deleted_nu, deleted_lam, deleted_Q, deleted_U = removed_data(input_dir_2)

#create directory to store the files containing training and predicted data

os.makedirs(name_dir + '/' + "complete" )


filename6 = "./" + name_dir + "/" + "complete" + "/" + "frequencies.txt"

filename7 = "./" + name_dir + "/" + "complete" + "/" + "wavelengths.txt"

filename8 = "./" + name_dir + "/" + "complete" + "/" + "stokesQ.txt"

filename9 = "./" + name_dir + "/" + "complete" + "/" + "stokesU.txt"


nu_data = []

stokesQ_data= []

stokesU_data= []

lam2_data = []

#generate the complete files (with training and predicted data)


for i in nu_R:

    nu_data.append(i)

for i in deleted_nu:
     
     nu_data.append(i)

np.savetxt(filename6, nu_data) 


for i in lam_R:

    lam2_data.append(i)

for i in x:
     
     lam2_data.append(i)

np.savetxt(filename7, lam2_data) 

for i in stokesQ_R:

    stokesQ_data.append(i)

for i in mu_optimized:
     
     stokesQ_data.append(i)

np.savetxt(filename8, stokesQ_data) 

for i in stokesU_R:

    stokesU_data.append(i)

for i in mu1_optimized:
     
     stokesU_data.append(i)

np.savetxt(filename9, stokesU_data) 



#plot initial predicition

ax = pl.subplot(111)
 
# plot the original values

pl.plot(lam_R, stokesQ_R, color='royalblue', linestyle='none', marker='.')

pl.plot(lam_R, stokesU_R, color='darkorange', linestyle='none', marker='.')

#pl.plot(lam_R, stokesQ_R,  linestyle='none', marker='o')

pl.plot(x, mu_initial,color = 'red', linestyle='none', marker='.')

pl.plot(x, mu1_initial, color = 'red', linestyle='none', marker='.')

# shade in the area inside a one standard deviation bound:

ax.fill_between(x,mu_initial-std_initial,mu_initial+std_initial,facecolor='lightgrey', lw=0, interpolate=True)

ax.fill_between(x,mu1_initial-std1_initial,mu1_initial+std1_initial,facecolor='lightgrey', lw=0, interpolate=True)

pl.ylabel("Stokes parameters")

pl.xlabel('$\lambda^2 [m^2]$')

pl.title("Stokes parameters - Initial Prediction")

pl.axis([0., 0.3,-0.75,1.5])

pl.grid()

pl.savefig(input_dir_2 + '/' +'2b.png')

pl.show()


#------------------------------------------------------------------------------------------------------------------------

#plot the final prediction



ax = pl.subplot(111)
 
# plot the original values

pl.plot(lam_R, stokesQ_R, color='royalblue', linestyle='none', marker='.')

pl.plot(lam_R, stokesU_R, color='darkorange', linestyle='none', marker='.')

#pl.plot(lam_R, stokesQ_R,  linestyle='none', marker='o')

pl.plot(x, mu_optimized,color = 'red', linestyle='none', marker='.')

pl.plot(x, mu1_optimized, color = 'red', linestyle='none', marker='.')

# shade in the area inside a one standard deviation bound:

ax.fill_between(x,mu_optimized-std_optimized,mu_optimized+std_optimized,facecolor='lightgrey', lw=0, interpolate=True)

ax.fill_between(x,mu1_optimized-std1_optimized,mu1_optimized+std1_optimized,facecolor='lightgrey', lw=0, interpolate=True)

pl.ylabel("Stokes parameters")

pl.xlabel('$\lambda^2 [m^2]$')

pl.title("Stokes parameters - Final Prediction")

pl.axis([0., 0.3,-0.75,1.5])

pl.grid()

pl.savefig(input_dir_2 + '/' +'2c.png')

pl.show()


#-------------------------------------------------------------------------------------------------------------------------

# plot the full dataset

#deleted_lam, deleted_Q, deleted_U = removed_data(input_dir_2)

ax = pl.subplot(111)

pl.plot(lam_R,stokesQ_R, color='royalblue', linestyle='none', marker='.')

pl.plot(lam_R,stokesU_R, color='darkorange', linestyle='none', marker='.')

pl.plot(x, mu_optimized, color = 'red', linestyle='none', marker='.')

pl.plot(x, mu1_optimized, color = 'red', linestyle='none', marker='.')

ax.fill_between(x,mu_optimized-std_optimized,mu_optimized+std_optimized,facecolor='lightgrey', lw=0, interpolate=True)

ax.fill_between(x,mu1_optimized-std1_optimized,mu1_optimized+std1_optimized,facecolor='lightgrey', lw=0, interpolate=True)

pl.plot(deleted_lam, deleted_Q,color='royalblue', linestyle='none', marker='.') 

pl.plot(deleted_lam, deleted_U,color='darkorange', linestyle='none', marker='.') 
 
pl.ylabel("Stokes parameters")

pl.xlabel('$\lambda^2 [m^2]$')

pl.title("Comparison")

pl.axis([0., 0.3,-0.75,1.5])

pl.grid()

pl.savefig(input_dir_2 + '/' +'2d.png')

pl.show()

#-----------------------------------------------------------------------------------------------------------------------


