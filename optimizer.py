import autograd.numpy as np
from autograd import grad
import pickle
from tqdm import tqdm
import time

from ioutils import * # load data
from model import accuracy,saveModel,hessian,log_likelihood

def simple_gd(w, b, x, y, loss_gradient_w, loss_gradient_b, learning_rate = 0.01, max_iter=100):
    for i in range(max_iter):
        w -= loss_gradient_w(w, b, x, y) * learning_rate
        print("W SHAPE: ", w.shape)
        print(w)
        b -= loss_gradient_b(w, b, x, y) * learning_rate
        print("B SHAPE: ", b.shape)
        print(b)
    return w, b


class Optimizers:
    def __init__(self,precision='float64'):
        self.precision = precision
        return
    def simple_gd(self, w, b, x, y, loss_gradient_w, loss_gradient_b, learning_rate = 0.01, max_iter=100):
        for i in range(max_iter):
            w -= loss_gradient_w(w, b, x, y).astype(self.precision) * learning_rate
            b -= loss_gradient_b(w, b, x, y).astype(self.precision) * learning_rate
        return w, b

    def simple_stoch_gd(self,w, b, x, y, loss_gradient_w, loss_gradient_b, learning_rate=0.01, max_iter=100, batch_size=100,plots=False):
        if plots ==True:
            loss_arr = [] 
            acc_arr = [] 
        
        n_samples = len(y)
        for i in tqdm(range(max_iter)):
            idx = np.random.randint(0, n_samples, batch_size)  # Randomly pick one or more indices
            x_batch = x[idx]
            y_batch = y[idx]
    
            if learning_rate == 'random':
                possible_lrs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
                learning_rate = np.random.choice(possible_lrs)
            
            w -= loss_gradient_w(w, b, x_batch, y_batch).astype(self.precision) * learning_rate
            b -= loss_gradient_b(w, b, x_batch, y_batch).astype(self.precision) * learning_rate

            if plots == True:
                acc = accuracy(w, b, x, y)
                loss_ = loss(w, b, x, y)
                loss_arr.append(loss_)
                acc_arr.append(acc)
            
        if plots == True:
            return w, b, loss_arr, acc_arr
        return w,b

    def adagrad(self,w, b, x, y, loss_gradient_w, loss_gradient_b, 
                learning_rate=0.01, max_iter=100, batch_size=100, epsilon=1e-4,mini_batch=True):


        w_scale_factor = np.zeros_like(w).astype(self.precision)
        b_scale_factor = np.zeros_like(b).astype(self.precision)

        for i in tqdm(range(max_iter)):
            if mini_batch== True:
                idx = np.random.randint(0, len(y), batch_size)
                x_batch = x[idx]
                y_batch = y[idx]

                w_grad = loss_gradient_w(w, b, x_batch, y_batch).astype(self.precision)
                b_grad = loss_gradient_b(w, b, x_batch, y_batch).astype(self.precision)

            else:
                w_grad = loss_gradient_w(w, b, x, y).astype(self.precision)
                b_grad = loss_gradient_b(w, b, x, y).astype(self.precision)

            #print("WGRAD",w_grad)
    
            w_scale_factor += w_grad**2

            #print("WSCALE",w_scale_factor)
            b_scale_factor += b_grad**2
    
            w -= learning_rate * w_grad / (np.sqrt(w_scale_factor)+epsilon)
            #print('NEW W',w)
            b -= learning_rate * b_grad / (np.sqrt(b_scale_factor)+epsilon)

           # print(w,b)

        return w,b

    def rmsprop(self,w, b, x, y, loss_gradient_w, loss_gradient_b, 
                learning_rate=0.01, max_iter=100, batch_size=100,epsilon=1e-4,decay_rate=0.5,mini_batch=True):

        w_scale_factor = np.zeros_like(w)
        b_scale_factor = np.zeros_like(b)

        for i in tqdm(range(max_iter)):
            if mini_batch== True:
                idx = np.random.randint(0, len(y), batch_size)
                x_batch = x[idx]
                y_batch = y[idx]
    
                w_grad = loss_gradient_w(w, b, x_batch, y_batch).astype(self.precision)
                b_grad = loss_gradient_b(w, b, x_batch, y_batch).astype(self.precision)
    
            else:
                w_grad = loss_gradient_w(w, b, x, y).astype(self.precision)
                b_grad = loss_gradient_b(w, b, x, y).astype(self.precision)
    
            #print("WGRAD",w_grad)
    
            w_scale_factor = decay_rate*w_scale_factor + (1-decay_rate)*w_grad**2
    
            #print("WSCALE",w_scale_factor)
            b_grad = np.clip(b_grad, -1, 1)
            b_scale_factor += decay_rate*b_scale_factor + (1-decay_rate)*b_grad**2
    
            w -= learning_rate * w_grad / (np.sqrt(w_scale_factor+epsilon))
            #print('NEW W',w)
            b -= learning_rate * b_grad / (np.sqrt(b_scale_factor+epsilon))

           # print(w,b)

        return w,b
    
    def adam(self,w, b, x, y, loss_gradient_w, loss_gradient_b, 
             learning_rate=0.01, max_iter=100, batch_size=100, 
             epsilon=1e-4,
            beta_1=.9,beta_2=0.9,mini_batch=True,plots=False):
        
        if plots ==True:
            loss_arr = [] 
            acc_arr = [] 
        
        w_scale_factor = np.zeros_like(w).astype(self.precision)
        b_scale_factor = np.zeros_like(b).astype(self.precision)

        w_moment_1 = np.zeros_like(w).astype(self.precision)
        w_moment_2 = np.zeros_like(w).astype(self.precision)

        b_moment_1 = np.zeros_like(b).astype(self.precision)
        b_moment_2 = np.zeros_like(b).astype(self.precision)
        
        for i in tqdm(range(max_iter)):
            if mini_batch== True:
                idx = np.random.randint(0, len(y), batch_size)
                x_batch = x[idx]
                y_batch = y[idx]
    
                w_grad = loss_gradient_w(w, b, x_batch, y_batch).astype(self.precision)
                b_grad = loss_gradient_b(w, b, x_batch, y_batch).astype(self.precision)
    
            else:
                w_grad = loss_gradient_w(w, b, x, y).astype(self.precision)
                b_grad = loss_gradient_b(w, b, x, y).astype(self.precision)
    
            w_moment_1 = beta_1 * w_moment_1 + (1-beta_1)*w_grad
            w_moment_2= beta_2 * w_moment_2 + (1-beta_2)*w_grad**2
    
            b_moment_1 = beta_1 * b_moment_1 + (1-beta_1)*b_grad
            b_moment_2= beta_2 * b_moment_2 + (1-beta_2)*b_grad**2
    
            w -= learning_rate * w_moment_1 /(np.sqrt(w_moment_2+epsilon))
            b -= learning_rate * b_moment_1 /(np.sqrt(b_moment_2+epsilon))

            if plots == True:
                acc = accuracy(w, b, x, y)
                loss_ = lr.loss(w, b, x, y)
                loss_arr.append(loss_)
                acc_arr.append(acc)
            
        if plots == True:
            return w, b, loss_arr, acc_arr

        return w,b

    def newton(self,w, b, x, y, loss_gradient, max_iter=1000,conv_threshold=1e-10,cond_threshold=1e10):
        '''https://thelaziestprogrammer.com/sharrington/math-of-machine-learning/solving-logreg-newtons-method'''
        print(max_iter)
        delta_l = np.inf
        l = log_likelihood(w, b, x, y)
        i = 0
        
        for i in tqdm(range(max_iter)):
            #print('iter',i)
            #print('w',w)
            #print('b',b)

            w_grad, b_grad = loss_gradient(w, b, x, y)
            
            #w_grad=w_grad#.astype(self.precision)
            #b_grad=b_grad#.astype(self.precision)
            
            #w_grad = np.clip(w_grad, -1, 1)
            #b_grad = np.clip(b_grad, -1, 1)

            #print('wgrad',w_grad,'b_grad',b_grad)
            H = hessian(x, y, w, b)
            H_reg = H + 1e-1 * np.eye(H.shape[0])
            cond_num = np.linalg.cond(H_reg)

            if cond_num > cond_threshold:
                H_reg = H + 1 * np.eye(H.shape[0])

            
            H_inv = np.linalg.inv(H_reg)
            #print(H_inv)
            update = np.dot(H_inv, np.concatenate([w_grad, [b_grad]]))

            
            w -= update[:-1]
            b -= update[-1]
    
            l_new = log_likelihood(w, b, x, y)
            delta_l = l - l_new
            l = l_new



            #print('l',l)
    
            if abs(delta_l)<conv_threshold:
                return w,b
    
        return w, b