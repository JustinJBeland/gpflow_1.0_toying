import matplotlib.pyplot as plt
import gpflow
import numpy as np
from f import *

# initialize the dataset and plot
t = 8
X = np.reshape(np.random.rand(t,1),(-1,1))
Y = np.reshape(f(X), (-1,1))
plt.figure(figsize=(12,6))
plt.plot(X,Y,'*', color='red', markersize=12, label='$\mathcal{D}^{1:t}$')
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$f(x)$', fontsize=20)
plt.legend(fontsize=20)
#plt.show()
plt.savefig('dataset')
plt.close()


# model construction
k = gpflow.kernels.RBF(1, ARD=True)
meanf = gpflow.mean_functions.Zero()
m = gpflow.models.GPR(X, Y, k, meanf)
m.likelihood.variance = 0.01
m.kern.variance.trainable = True
m.likelihood.variance.trainable = False
print(m.read_trainables())

# plotting function for gp
def plot(m, name):
    xx = np.linspace(0.0, 1.0, 100).reshape(100, 1)
    mean, var = m.predict_y(xx)
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, '*', markersize=12, color='red', label='$\mathcal{D}^{1:t}$')
    plt.plot(xx, mean, 'C0', lw=2)
    plt.fill_between(xx[:,0],
                     mean[:,0] - 2*np.sqrt(var[:,0]),
                     mean[:,0] + 2*np.sqrt(var[:,0]),
                     color='C0', alpha=0.2, label='$f\sim\mathcal{GP}$')
    plt.xlim(0.0, 1.0)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$f(x)$', fontsize=20)
    plt.legend(fontsize=20, loc=2)
    plt.savefig(name)
    plt.close()
    #plt.show()

print('THIS IS THE PRIOR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
plot(m, 'prior')
print(m)

# optimize hyperparams
gpflow.train.ScipyOptimizer().minimize(m)
m.compile()
print('THIS IS THE POSTERIOR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
plot(m, 'posterior')
print(m)

# print hyperparams
print('variance    :', m.kern.variance.value)
print('lengthscales:', m.kern.lengthscales.value)

# use mcmc to estimate hyperparams
m.clear()
m.kern.lengthscales.prior = gpflow.priors.Gamma(1., 1.)
m.kern.variance.prior = gpflow.priors.Gamma(1., 1.)
m.compile()
print('THIS IS AFTER MCMC INITIALIZED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print(m)

# plot the samples for the iterative scheme
sampler = gpflow.train.HMC()
samples = sampler.sample(m, num_samples=500000, epsilon=0.05, lmin=10, lmax=20, logprobs=False)

for i, col in samples.iteritems():
    plt.plot(col, label=col.name)
plt.legend(loc=0)
plt.xlabel('hmc iteration')
plt.ylabel('parameter value')
plt.savefig('mcmc_iterations')
plt.close()
