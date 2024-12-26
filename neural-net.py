import numpy as np
import math

class Module:
    def step(self, lrate): pass  # For modules w/o weights

# Linear modules
#
# Each linear module has a forward method that takes in a batch of
# activations A (from the previous layer) and returns
# a batch of pre-activations Z.
#
# Each linear module has a backward method that takes in dLdZ and
# returns dLdA. This module also computes and stores dLdW and dLdW0,
# the gradients with respect to the weights.
class Linear(Module):
    def __init__(self, m, n):
        self.m, self.n = (m, n)  # (in size, out size)
        self.W0 = np.zeros([self.n, 1])  # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-.5), [m, n])  # (m x n)

    def forward(self, A):
        self.A = A   # (m x b)
        self.Z = self.W.T @ self.A + self.W0 # (n x b)
        return self.Z

    def backward(self, dLdZ):  # dLdZ is (n x b), uses stored self.A
        self.dLdW  = self.A @ dLdZ.T  
        self.dLdW0 = dLdZ @ np.ones((len(self.A.T), 1))
        return self.W @ dLdZ        # (m x b)

    def step(self, lrate):  # Gradient descent step
        self.W  = self.W - lrate * self.dLdW 
        self.W0 = self.W0 - lrate * self.dLdW0 

# Activation modules
#
# Each activation module has a forward method that takes in a batch of
# pre-activations Z and returns a batch of activations A.
#
# Each activation module has a backward method that takes in dLdA and
# returns dLdZ, with the exception of SoftMax, where we assume dLdZ is
# passed in.
class Tanh(Module):  # Layer activation
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):  # Uses stored self.A
        return dLdA * (1 - self.A ** 2) # (?, b)


class ReLU(Module):  # Layer activation
    def forward(self, Z):
        self.A = Z * (Z > 0)  # (?, b)
        return self.A

    def backward(self, dLdA):  # uses stored self.A
        return dLdA * np.where(self.A > 0, 1, 0)  # (?, b)


class SoftMax(Module):  # Output activation
    def forward(self, Z):
        ez = np.exp(Z - np.max(Z, axis=0))
        return ez / ez.sum(axis=0) # (?, b)

    def backward(self, dLdZ):  # Assume that dLdZ is passed in
        return dLdZ

    def class_fun(self, Ypred):  # Return class indices
        classes = []
        for point in Ypred.T:
            classes.append(np.argmax(point)) 

        classes = np.array(classes)
        classes.shape = (len(Ypred.T), 1)

        return classes

# Loss modules
#
# Each loss module has a forward method that takes in a batch of
# predictions Ypred (from the previous layer) and labels Y and returns
# a scalar loss value.
#
# The NLL module has a backward method that returns dLdZ, the gradient
# with respect to the preactivation to SoftMax (note: not the
# activation!), since we are always pairing SoftMax activation with
# NLL loss
class NLL(Module):  # Loss
    def forward(self, Ypred, Y):
        eps = 1e-15
        Ypred = np.clip(Ypred, eps, 1 - eps)
        
        self.Ypred = Ypred
        self.Y = Y
        
        loss = -np.sum(Y * np.log(Ypred) + (1 - Y) * np.log(1 - Ypred), axis=1)
        return np.mean(loss).item()

    def backward(self):  # Use stored self.Ypred, self.Y
        return self.Ypred - self.Y # (?, b)

class Sequential:
    def __init__(self, modules, loss):            
        self.modules = modules
        self.loss = loss

    def mini_gd(self, X, Y, iters, lrate, notif_each=None, K=10):
        D, N = X.shape

        np.random.seed(0)
        num_updates = 0
        indices = np.arange(N)
        while num_updates < iters:

            np.random.shuffle(indices)
            X = X[:, indices]
            Y = Y[:, indices]
            for j in range(1, math.floor(N/K) + 1):
                if num_updates >= iters: break

                for i in range(1, K + 1):
                    start_idx = (j-1) * K + (i-1)
                    end_idx = (j-1) * K + i
                
                    if end_idx > N:
                        continue
                    Ypred = self.forward(X[:, start_idx:end_idx])
                    self.loss.forward(Ypred, Y[:, start_idx:end_idx])
                    self.backward(self.loss.backward())
                    self.step(lrate)
                
                num_updates += 1

    def forward(self, Xt):                        
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):                   
        for m in self.modules[::-1]: delta = m.backward(delta)

    def step(self, lrate):    
        for m in self.modules: m.step(lrate)
          

class BatchNorm(Module):    
    def __init__(self, m):
        np.random.seed(0)
        self.eps = 1e-20
        self.m = m  # number of input channels
        
        # Init learned shifts and scaling factors
        self.B = np.zeros([self.m, 1])
        self.G = np.random.normal(0, 1.0 * self.m ** (-.5), [self.m, 1])
        
    # Works on m x b matrices of m input channels and b different inputs
    def forward(self, A): # A is m x K: m input channels and mini-batch size K
        # Store last inputs and K for next backward() call
        self.A = A
        self.K = A.shape[1]
        
        self.mus = np.mean(A, axis=1, keepdims=True)
        self.vars = np.var(A, axis=1, keepdims=True)

        # Normalize inputs using their mean and standard deviation
        self.norm = (A - self.mus) / np.sqrt(self.vars + self.eps)

        # Return scaled and shifted versions of self.norm
        return self.G * self.norm + self.B

    def backward(self, dLdZ):
        # Re-usable constants
        std_inv = 1/np.sqrt(self.vars+self.eps)
        A_min_mu = self.A-self.mus
        
        dLdnorm = dLdZ * self.G
        dLdVar = np.sum(dLdnorm * A_min_mu * -0.5 * std_inv**3, axis=1, keepdims=True)
        dLdMu = np.sum(dLdnorm*(-std_inv), axis=1, keepdims=True) + dLdVar * (-2/self.K) * np.sum(A_min_mu, axis=1, keepdims=True)
        dLdX = (dLdnorm * std_inv) + (dLdVar * (2/self.K) * A_min_mu) + (dLdMu/self.K)
        
        self.dLdB = np.sum(dLdZ, axis=1, keepdims=True)
        self.dLdG = np.sum(dLdZ * self.norm, axis=1, keepdims=True)
        return dLdX

    def step(self, lrate):
        self.B = self.B - lrate * self.dLdB
        self.G = self.G - lrate * self.dLdG


######################################################################
# Tests
######################################################################
def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)
  
def for_softmax(y):
    return np.vstack([1-y, y])
  
def mini_gd_test():
    np.random.seed(0)
    nn = Sequential([Linear(2,3), ReLU(), Linear(3,2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.mini_gd(X,Y, iters = 3, lrate=0.005, K=1)
    return [np.vstack([nn.modules[0].W, nn.modules[0].W0.T]).tolist(),
            np.vstack([nn.modules[2].W, nn.modules[2].W0.T]).tolist()]
  
def batch_norm_test():
    np.random.seed(0)
    nn = Sequential([Linear(2,3), ReLU(), Linear(3,2), BatchNorm(2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.mini_gd(X,Y, iters = 1, lrate=0.005, K=2)
    return [np.vstack([nn.modules[3].B, nn.modules[3].G]).tolist(), \
    np.vstack([nn.modules[3].mus, nn.modules[3].vars]).tolist(), nn.modules[3].norm.tolist()]

