import autograd.numpy as np
from autograd import grad
import pickle

# Model definition
# TODO: Below is an implementation of logisitc regression as a placeholder.
#       You must implement your own model

def relu():
    return np.maximum(0, x)

def sigmoid(z):
    z = np.clip(z, a_min=None, a_max=1)
    s = 1./(1 + np.exp(-z))
    return s

def predict(w, b, x):
    z = np.dot(x, w) + b
    return sigmoid(z)

def loss(w, b, x, y,epsilon=1e-4):
    y_pred = predict(w, b, x)
    return -np.mean(y*np.log(y_pred+epsilon)+ (1-y)*np.log(1-y_pred+epsilon))

def accuracy(w, b, x, y, thres=0.5):
    y_pred = predict(w, b, x)
    cls_pred = y_pred > thres
    return np.mean(cls_pred == y)

def loadModel(modelpath):
    with open(modelpath, 'rb') as f:
        w = pickle.load(f)
        b = pickle.load(f)
    return w, b

def saveModel(modelpath, w, b):
    with open(modelpath, 'wb') as f:
        pickle.dump(w, f)
        pickle.dump(b, f)

def hessian(x, y, w, b):
    sigmoid_probs = predict(w, b, x)
    diag = sigmoid_probs * (1 - sigmoid_probs)
    diag = diag[:, np.newaxis]
    H = np.dot(x.T, diag * x)
    H_intercept = np.sum(diag)
    H_0j = np.sum(diag * x, axis=0)
    H_full = np.vstack([np.concatenate([[H_intercept], H_0j]), np.column_stack([H_0j, H])])
    return H_full

def log_likelihood(w, b, x, y,epsilon=1e-4):
    y_pred = predict(w, b, x)
    return np.sum(y * np.log(y_pred + epsilon - 1) + (1 - y) * np.log(1 - y_pred + epsilon - 1))



class OrgLogReg:
    def __init__(self,precision='float64'):
        self.precision = precision
        np.set_printoptions(precision=10)
        return
    def gen_w_b(self,dim):
        np.random.seed(10)
        w = np.random.rand(dim)
        b = 0.
        return w,b
    def relu(self,x):
        return np.maximum(0, x)
    def sigmoid(self,z):
        z = np.clip(z, a_min=None, a_max=0.8)
        s = 1./(1 + np.exp(-z))
        return s
    def predict(self,w, b, x):
        z = np.dot(x, w) + b
        return self.sigmoid(z)
    def loss(self,w, b, x, y,epsilon=1e-4):
        y_pred = self.predict(w, b, x)
        return -np.mean(y*np.log(y_pred+epsilon) + (1-y)*np.log(1-y_pred+epsilon))
