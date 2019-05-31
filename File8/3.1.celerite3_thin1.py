###works well for all datasets (faraday thin sources)

###improved for faraday thin source (parameters tuned)

###no wavelengths extension


#adapted from https://allofyourbases.com/2017/10/02/spinning-stars-ii-celerite/



import numpy as np

import pylab as pl

#import matplotlib.pyplot as plt

import os,sys

#import rm_tools as R

#import celerite

#from celerite import terms

import celerite

from celerite import terms

const_c = 3e8


def read_data(input_dir, filename1):
        
       params_file = input_dir + "/" + filename1 

       nu = np.loadtxt(params_file, usecols=(0,))   


       lam_squared = np.loadtxt(params_file, usecols=(1,))

       stokesQ = np.loadtxt(params_file, usecols=(2,))

       stokesU = np.loadtxt(params_file, usecols=(3,)) 
        

       los = stokesQ + 1j*stokesU

       #return nu, lam_squared, stokesQ, stokesU

       return nu, lam_squared, stokesQ, stokesU



def QU_data(input_dir, filename4 = "training.txt"):

       nu, lam_squared, stokesQ, stokesU = read_data(input_dir, filename4)

       nu_R = nu

       stokesQ_R = stokesQ

       stokesU_R = stokesU
       
       lam2_R = lam_squared

       return nu_R, lam2_R, stokesQ_R, stokesU_R


def x_values( input_dir, filename5 = "removed.txt"):

       nu, lam_squared, stokesQ, stokesU = read_data(input_dir, filename5)

       t2 = lam_squared

       stokesQ_D = stokesQ

       stokesU_D = stokesU  

       return t2,stokesQ_D, stokesU_D


#input_dir_1="./data/"

name_dir = input('Please enter the name of the directory containing data (use quotes): ')

input_dir = "./" + name_dir + "/"


nu, lam_squared, stokesQ, stokesU = read_data(input_dir, "all_params.txt") ###inserted


nu_R, lam2_R, stokesQ_R, stokesU_R = QU_data(input_dir, filename4 = "training.txt")

lam2_R = lam2_R[::-1]

stokesQ_R = stokesQ_R[::-1]

stokesU_R = stokesU_R[::-1]

t2,stokesQ_D, stokesU_D = x_values( input_dir, filename5 = "removed.txt")

#t = lam2_R

t1 = np.linspace(np.min(lam2_R), np.max(lam2_R), 1024)  ###no frequency extension

pl.plot(lam2_R, stokesQ_R, linestyle='none', marker='.', color='darkblue')

pl.plot(lam2_R, stokesU_R, linestyle='none', marker='.', color='darkorange')

pl.xlabel('$\lambda^2 [m^2]$')

pl.ylabel('stokes parameters ')

pl.savefig("./" + name_dir + "/"+ "2a.png")

pl.show()


import autograd.numpy as np
 
class CustomTerm(terms.Term):
    parameter_names = ("log_a", "log_b", "log_c", "log_P")
 
    def get_real_coefficients(self, params):
        log_a, log_b, log_c, log_P = params
        b = np.exp(log_b)
        return (
            np.exp(log_a) * (1.0 + b) / (2.0 + b), np.exp(log_c),
        )
 
    def get_complex_coefficients(self, params):
        log_a, log_b, log_c, log_P = params
        b = np.exp(log_b)
        return (
            np.exp(log_a) / (2.0 + b), 0.0,
            np.exp(log_c), 2*np.pi*np.exp(-log_P),
        )




log_a = 0.1;log_b = 0.1; log_c = 0.1; log_P = 0.1 #working up to 3072

#log_a = 0.1;log_b = 0.1; log_c = 0.1; log_P = 1.0   #working with 1024

kernel = CustomTerm(log_a, log_b, log_c, log_P)


gp = celerite.GP(kernel, mean=0.0)
gp.compute(lam2_R)

#gp1 = celerite.GP(kernel, mean=np.mean(stokesU_R))
#gp.compute(lam2_R)
 
print("Initial log-likelihood: {0}".format(gp.log_likelihood(stokesQ_R)))
print("Initial log-likelihood: {0}".format(gp.log_likelihood(stokesU_R)))
 
# calculate expectation and variance at each point:
mu, cov = gp.predict(stokesQ_R, t1)

std = np.sqrt(np.diag(cov))


mu1, cov1 = gp.predict(stokesU_R, t1)

std1 = np.sqrt(np.diag(cov1))

ax = pl.subplot(111)

#pl.plot(lam2_R, stokesQ_R, color='royalblue', linestyle='none', marker='.')

pl.plot(t1,mu, linestyle='none', marker='.', color = 'red') 
pl.plot(t1,mu1, linestyle='none', marker='.', color = 'red')

pl.plot (lam2_R,stokesQ_R,linestyle='none', marker='.', color = 'darkblue')
pl.plot (lam2_R,stokesU_R,linestyle='none', marker='.', color = 'darkorange')

ax.fill_between(t1, mu-std, mu+std,facecolor='lightgray')# lw=0, interpolate=True)
ax.fill_between(t1, mu1-std1, mu1+std1,facecolor='lightgray')# lw=0, interpolate=True)

pl.title('Initial prediction')

pl.ylabel('stokes parameters')

pl.xlabel('$\lambda^2 [m^2]$')  

pl.savefig("./" + name_dir + "/"+ "2b.png")

pl.show()


def nll(p, y, gp):
 
    # Update the kernel parameters:
    gp.set_parameter_vector(p)
 
    #  Compute the loglikelihood:
    ll = gp.log_likelihood(y)
 
    return -ll if np.isfinite(ll) else 1e25



def grad_nll(p, y, gp):
 
    # Update the kernel parameters:
    gp.set_parameter_vector(p)
 
    #  Compute the gradient of the loglikelihood:
    gll = gp.grad_log_likelihood(y)[1]
 
    return -gll

import scipy.optimize as op
 

p0 = gp.get_parameter_vector()
 

#bnds = ((-5.,5.),(-5.,5.),(-5.,-1.),(-5.,5.)) #working up to 3072
#bnds = ((-5.,2.),(-5.,5.),(-5.,1.),(-5.,5.))
#bnds = ((-15.,15.),(-15.,15.),(-15.,15.),(-15.,15.))
bnds = ((-15.,10.),(-15.,10.),(-15.,10.),(-15.,10.))

# run optimization:
results = op.minimize(nll, p0, method='L-BFGS-B', jac=grad_nll, bounds=bnds, args=(stokesQ_R, gp)) #

print np.exp(results.x)
print("Final log-likelihood: {0}".format(-results.fun))

# pass the parameters to the george kernel:
gp.set_parameter_vector(results.x)



results1 = op.minimize(nll, p0, method='L-BFGS-B', jac=grad_nll, bounds=bnds, args=(stokesU_R, gp))
print np.exp(results1.x)
print("Final log-likelihood: {0}".format(-results1.fun))
gp.set_parameter_vector(results1.x)



# calculate expectation and variance at each point:
mu, cov = gp.predict(stokesQ_R, t1)

std = np.sqrt(np.diag(cov))

mu1, cov1 = gp.predict(stokesU_R, t1)

std1 = np.sqrt(np.diag(cov1))


#Store new dataset in a new directory


os.makedirs(input_dir + '/' + "complete" )


filename6 = "./" + name_dir + "/" + "complete" + "/" + "frequencies.txt"

filename7 = "./" + name_dir + "/" + "complete" + "/" + "wavelengths.txt"

filename8 = "./" + name_dir + "/" + "complete" + "/" + "stokesQ.txt"

filename9 = "./" + name_dir + "/" + "complete" + "/" + "stokesU.txt"


nu_data = []

stokesQ_data= []

stokesU_data= []

lam2_data = []



#generate the complete files (with training and predicted data)


for i in nu:
     
     nu_data.append(i)

np.savetxt(filename6, nu_data) 


for i in t1:
     
     lam2_data.append(i)

np.savetxt(filename7, lam2_data) 


for i in mu:
     
     stokesQ_data.append(i)

np.savetxt(filename8, stokesQ_data) 


for i in mu1:
     
     stokesU_data.append(i)

np.savetxt(filename9, stokesU_data) 




ax = pl.subplot(111)

pl.plot(t1,mu, linestyle='none', marker='.', color = 'red')
pl.plot(t1,mu1, linestyle='none', marker='.', color = 'red')

pl. title('Maximum likelihood prediction')

pl.plot(lam2_R,stokesQ_R, linestyle='none', marker='.', color = 'darkblue')
pl.plot(lam2_R,stokesU_R, linestyle='none', marker='.', color = 'darkorange')


ax.fill_between(t1,mu-std,mu+std, facecolor='lightgray') #, lw=0, interpolate=True)
ax.fill_between(t1,mu1-std1,mu1+std1, facecolor='lightgray')

#pl.axis([0.,60.,-1.,1.])
pl.ylabel('stokes parameters ')

pl.xlabel('$\lambda^2 [m^2]$')

pl.savefig("./" + name_dir + "/"+ "2c.png")

pl.show()

#comparison
#----------------------------------------------------------------------------------------------
ax = pl.subplot(111)

pl. title('Comparison')

pl.plot(t1,mu, linestyle='none', marker='.', color = 'red')
pl.plot(t1,mu1, linestyle='none', marker='.', color = 'red')

pl.plot(lam2_R,stokesQ_R, linestyle='none', marker='.', color = 'darkblue')
pl.plot(lam2_R,stokesU_R, linestyle='none', marker='.', color = 'darkorange')

pl.plot(t2,stokesQ_D, linestyle='none', marker='.', color = 'black')
pl.plot(t2,stokesU_D, linestyle='none', marker='.', color = 'black')

ax.fill_between(t1,mu-std,mu+std, facecolor='lightgray') #, lw=0, interpolate=True)
ax.fill_between(t1,mu1-std1,mu1+std1, facecolor='lightgray') #, lw=0, interpolate=True)

#pl.axis([0.,60.,-1.,1.])
pl.ylabel('stokes parameters ')

pl.xlabel('$\lambda^2 [m^2]$')

pl.savefig("./" + name_dir + "/"+ "2d.png")

pl.show()









import emcee
 
# we need to define three functions:
# a log likelihood, a log prior & a log posterior.


# set the loglikelihood:
def lnlike(p, x, y1, y2):
 
    ln_a = p[0]
    ln_b = p[1]
    ln_c = p[2]      
    ln_p = p[3]  

    p0 = np.array([ln_a,ln_b,ln_c,ln_p])
 
    # update kernel parameters:
    gp.set_parameter_vector(p0)
 
    # calculate the likelihood:
    ll1 = gp.log_likelihood(y1)
    ll2 = gp.log_likelihood(y2)
    ll = ll1 + ll2
    
    return ll if np.isfinite(ll) else 1e25


# set the logprior
def lnprior(p):
 
    lnB = p[0]
    lnC = p[1]
    lnL = p[2]
    lnP = p[3]
 
    if (-15.<lnB<10.) and (-15.<lnC<10.) and (-15.<lnL<5.) and (-5<lnP<5.):    
        return 0.0
 
    return -np.inf


# set the logposterior:
def lnprob(p, x, y1, y2):
 
    lp = lnprior(p)
 
    return lp + lnlike(p, x, y1, y2) if np.isfinite(lp) else -np.inf


x_train = lam2_R
y_1 = stokesQ_R
y_2 = stokesU_R

data = (x_train,y_1,y_2)



p = gp.get_parameter_vector()

initial = np.array([p[0],p[1],p[2],p[3]])
print("Initial guesses: ",initial)
 
# initial log(likelihood):
init_logL = gp.log_likelihood(y_1) + gp.log_likelihood(y_2)
 
# set the dimension of the prior volume
# (i.e. how many parameters do you have?)
ndim = len(initial)
print("Number of parameters: ",ndim)
 
nwalkers = 32
 
p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim)
      for i in range(nwalkers)]

# initalise the sampler:
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)



# run a few samples as a burn-in:
print("Running burn-in")
p0, lnp, _ = sampler.run_mcmc(p0, 500)
sampler.reset()
 
print("Finished Burn-In")



# take the highest likelihood point from the burn-in as a
# starting point and now begin your production run:
print("Running production")
p = p0[np.argmax(lnp)]
p0 = [p + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
p0, lnp, _ = sampler.run_mcmc(p0, 5000)
 
print("Finished Production")






import corner
 
# Find the maximum likelihood values:
ml = p0[np.argmax(lnp)]
print("Maximum likelihood parameters: ",np.exp(ml))
print("Period: ",np.pi/np.exp(ml[3])," rad/m^2")
 
MLlnB = ml[0]
MLlnC = ml[1]
MLlnL = ml[2]
MLlnP = ml[3]
 
p = np.array([MLlnB,MLlnC,MLlnL,MLlnP])
gp.set_parameter_vector(p)
ml_logL = gp.log_likelihood(y_1) + gp.log_likelihood(y_2)
print("ML logL:", ml_logL)





ax = pl.subplot(111)

pl.plot(t1,mu, linestyle='none', marker='.', color = 'red')
pl.plot(t1,mu1, linestyle='none', marker='.', color = 'red')

pl.plot(lam2_R,stokesQ_R, linestyle='none', marker='.', color = 'darkblue')
pl.plot(lam2_R,stokesU_R, linestyle='none', marker='.', color = 'darkorange')

ax.fill_between(t1,mu-std,mu+std, facecolor='lightgray') #, lw=0, interpolate=True)
ax.fill_between(t1,mu1-std1,mu1+std1, facecolor='lightgray')

pl. title('Maximum likelihood prediction')
pl.ylabel('stokes parameters ')
pl.xlabel('$\lambda^2 [m^2]$')

pl.savefig("./" + name_dir + "/"+ "2e.png")

pl.show()



# Plot it.
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
#samples = sampler.flatchain
#ax = pl.subplot(111)


figure = corner.corner(samples, labels=[r"$lnB$", r"$lnC$", r"$lnL$", r"$lnP$"],
                         truths=ml,
                         quantiles=[0.16,0.5,0.84],
                         levels=[0.19,0.86,0.99],
                         title="Faraday Thin",
                         show_titles=True, title_args={"fontsize": 10})
                         

figure.show() 
figure.savefig("./" + name_dir + "/"+ "2f.png")

"""

import corner
fig = corner.corner(samples, labels=[r"$lnB$", r"$lnC$", r"$lnL$", r"$P$"],
                      truths=ml) #truths=[m_true, b_true, np.log(f_true)])
fig.savefig("triangle.png")
"""
