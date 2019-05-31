#Data P_array_test.npy from Joe
#10000 data sets
#135 frequency channels per dataset
#Each dataset has 2 columns, stokes Q and U
#Script uses for loop to fit the 10000 datasets (using celerite)



import os,sys

import numpy as np

from matplotlib import pyplot as pl

import scipy.constants 

import celerite

from celerite import terms

import autograd.numpy as np

import scipy.optimize as op

import emcee

# -----------------------------------------------------------------


channels = 135 

freq_min = 8.59079e8 

freq_max = 1.69933e9

freq = np.linspace(freq_min, freq_max, channels)

lamsq = (scipy.constants.c)**2 / (freq*freq)

dataset = np.load('P_array_test.npy')

nu = freq

rm_celerite = []

count = 0

for data in dataset:

 #while count < 2:
  
  stokes_params = data

  stokes_Q = data[:, 0]

  stokes_U = data[:, 1]


  #sort data

  freq = freq[::-1]

  stokesQ_R = stokes_Q[::-1]

  stokesU_R = stokes_U[::-1]

  lam2_R = lamsq[::-1]

  t1 = np.linspace(np.min(lam2_R), np.max(lam2_R), 135)

   
  #----------------------------------------------------------------------
  #stokes parameters versus frequency
  """

  pl.title('Stokes parameters versus frequency')

  pl.plot (freq, stokesQ_R, color = 'blue', linestyle = 'none', marker = '.')  

  #pl.plot (freq, stokes_Q, color = 'green', linestyle = 'none', marker = '.')

  pl.plot (freq, stokesU_R, color = 'orange', linestyle = 'none', marker = '.')

  #pl.plot (freq, stokes_U, color = 'purple', linestyle = 'none', marker = '.')

  pl.ylabel('Q and U')

  pl.xlabel('Frequency')

  pl.legend(['stokesQ1','stokesQ2','stokesU1','stokes2'])

  pl.show()


  #------------------------------------------------------------------------
  #stokes parameters versus wavelength

  pl.title('Stokes parameters versus wavelength squared')

  pl.plot (lam2_R, stokesQ_R, color = 'blue', linestyle = 'none', marker = '.')

  #pl.plot (lamsq, stokes_Q, color = 'green', linestyle = 'none', marker = '.')

  pl.plot (lam2_R, stokesU_R, color = 'orange', linestyle = 'none', marker = '.')

  #pl.plot (lamsq, stokes_U, color = 'purple', linestyle = 'none', marker = '.')

  pl.ylabel('Q and U')

  pl.xlabel('$\lambda^2$')

  pl.legend(['stokesQ','stokesU'])

  pl.show()
  """
 
  #import autograd.numpy as np
 
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



  log_a = 0.1;log_b = 0.1; log_c = 0.1; log_P = 0.1 


  kernel = CustomTerm(log_a, log_b, log_c, log_P)

  gp = celerite.GP(kernel, mean=0.0)

  gp.compute(lam2_R)

  print("Initial log-likelihood: {0}".format(gp.log_likelihood(stokesQ_R)))

  print("Initial log-likelihood: {0}".format(gp.log_likelihood(stokesU_R)))


  # calculate expectation and variance at each point:
  mu, cov = gp.predict(stokesQ_R, t1)

  std = np.sqrt(np.diag(cov))


  mu1, cov1 = gp.predict(stokesU_R, t1)

  std1 = np.sqrt(np.diag(cov1))
  
  """ 
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

  pl.show()

  """


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





  #import scipy.optimize as op
 

  p0 = gp.get_parameter_vector()


  #bnds = ((-20.,0.),(-15.,10.),(-15.,10.),(-15.,10.)) works for most of them

  bnds = ((-25.,5.),(-15.,10.),(-15.,10.),(-15.,10.))


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

  """

  ax = pl.subplot(111)

  pl.plot(t1,mu, linestyle='none', marker='.', color = 'red')
  pl.plot(t1,mu1, linestyle='none', marker='.', color = 'red')

  pl. title('Maximum likelihood prediction-Scipy')

  pl.plot(lam2_R,stokesQ_R, linestyle='none', marker='.', color = 'darkblue')
  pl.plot(lam2_R,stokesU_R, linestyle='none', marker='.', color = 'darkorange')


  ax.fill_between(t1,mu-std,mu+std, facecolor='lightgray') #, lw=0, interpolate=True)
  ax.fill_between(t1,mu1-std1,mu1+std1, facecolor='lightgray')

  #pl.axis([0.,60.,-1.,1.])
  pl.ylabel('stokes parameters ')

  pl.xlabel('$\lambda^2 [m^2]$')

  pl.show()

  """






  #import emcee
 
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
 
    #if (-5.<lnB<15.) and (-5.<lnC<10.) and (-15.<lnL<5.) and (-5<lnP<5.):    
        #return 0.0

    #if (-55.<lnB<10.) and (-15.<lnC<10.) and (-15.<lnL<5.) and (-15<lnP<5.):    
        #return 0.0

    #if (-20.<lnB<10.) and (-15.<lnC<5.) and (-10.<lnL<15.) and (-10<lnP<15.):    
        #return 0.0   #works for most of them

    if (-25.<lnB<5.) and (-15.<lnC<5.) and (-10.<lnL<15.) and (-10<lnP<15.):    
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

  #added to allows of angles
  pharr = np.arctan2(stokesU_R,stokesQ_R)
  phase = np.mean(pharr)*180./np.pi
  phsig = np.sign(phase)


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

  print("Period: ",phsig * np.pi/np.exp(ml[3])," rad/m^2")

  rm_value = np.pi/np.exp(ml[3])

  rm_celerite.append(rm_value)
  
  MLlnB = ml[0]
  MLlnC = ml[1]
  MLlnL = ml[2]
  MLlnP = ml[3]
 
  p = np.array([MLlnB,MLlnC,MLlnL,MLlnP])
  gp.set_parameter_vector(p)
  ml_logL = gp.log_likelihood(y_1) + gp.log_likelihood(y_2)
  print("ML logL:", ml_logL)

 

  count+=1
  print ("Count: {}".format(count))






  """
  ax = pl.subplot(111)

  pl.plot(t1,mu, linestyle='none', marker='.', color = 'red')
  pl.plot(t1,mu1, linestyle='none', marker='.', color = 'red')

  pl.plot(lam2_R,stokesQ_R, linestyle='none', marker='.', color = 'darkblue')
  pl.plot(lam2_R,stokesU_R, linestyle='none', marker='.', color = 'darkorange')

  ax.fill_between(t1,mu-std,mu+std, facecolor='lightgray') #, lw=0, interpolate=True)
  ax.fill_between(t1,mu1-std1,mu1+std1, facecolor='lightgray')

  pl. title('Maximum likelihood prediction-Emcee')
  pl.ylabel('stokes parameters ')
  pl.xlabel('$\lambda^2 [m^2]$')

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
                         

  #figure.show() 
  figure.savefig("celerite2a.png")
  """ 


  




np.savetxt('rm_celerite.txt', rm_celerite)  #save in .txt format
np.save('rm_celerite', rm_celerite)   #save file in .npy format





  

 



