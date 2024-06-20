from ioutils import *
from model import *
from optimizer import *
from tqdm import tqdm 
import pandas as pd
import matplotlib.pyplot as plt 
import sys 


def plot_training_loss(loss_arr):
    loss_df = pd.DataFrame(loss_arr, columns=['Training Loss'])
    #print(loss_df)
    loss_df.plot.line()
    plt.title('Training Loss Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

def plot_training_acc(acc_arr):
    acc_df = pd.DataFrame(acc_arr, columns=['Accuracy'])
    #print(loss_df)
    acc_df.plot.line()
    plt.title('Training Accuracy Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()


def logistic_regression_adam_final():
    print('Loading the banana quality dataset with.....')
    print('Adding 3rd degree polynomials... \n')
    x_train, y_train, x_test, y_test = readDataPoly(degree=3)
    N_train = x_train.shape[0]
    N_test = x_test.shape[0]
    dim = x_test.shape[1]
    print('....Initailzing Logistic Regression... \n')
    lr = OrgLogReg(precision='float16')

    w,b = lr.gen_w_b(dim)
    loss_gradient_w = grad(lr.loss, 0)
    loss_gradient_b = grad(lr.loss, 1)

    print("Initial loss:", lr.loss(w, b, x_test, y_test))
    print("Initial accuracy:", accuracy(w, b, x_test, y_test))

    max_iter = 1000
    learning_rate = 0.03
    batch_size = 100
    beta_1 = .9
    beta_2 = 0.9

    print('Initailzing Optimizer Adam with float16 precision \n')
    op = Optimizers(precision='float16')

    print('adam \n')
    w,b = lr.gen_w_b(dim)
    w,b = op.adam(w, b, x_train, y_train, loss_gradient_w, loss_gradient_b,
                  max_iter=max_iter,learning_rate=learning_rate,beta_1=beta_1,beta_2=beta_2,batch_size=batch_size)
    
    print("Train loss:", lr.loss(w, b, x_train, y_train))
    print("Train accuracy:", accuracy(w, b, x_train, y_train))
    
    print("Validation loss:", lr.loss(w, b, x_test, y_test))
    print("Validation accuracy:", accuracy(w, b, x_test, y_test))
    print('\n')

    print('Saving model')
    # Save the model
    #saveModel('model.pkl', w, b) 
    saveModel('another_model.pkl', w, b) # changed the time of model file so original model is not overwritten by tester

def run_all_optimizers():
    x_train, y_train, x_test, y_test = readDataPoly(degree=3)


    N_train = x_train.shape[0]
    N_test = x_test.shape[0]
    dim = x_test.shape[1]
    
    
    lr = OrgLogReg()
    
    w,b = lr.gen_w_b(dim)
    
    loss_gradient_w = grad(lr.loss, 0)
    loss_gradient_b = grad(lr.loss, 1)
    
    print("Initial loss:", lr.loss(w, b, x_test, y_test))
    print("Initial accuracy:", accuracy(w, b, x_test, y_test))
    
    max_iter = 1000
    learning_rate = 0.03
    batch_size = 200
    decay_rate = 0.01
    beta_1 = .9
    beta_2 = 0.99
    
    op = Optimizers()
    
    print('\n')
    
    
    print('simple_gd')
    w,b = lr.gen_w_b(dim)
    w,b = op.simple_gd(w, b, x_train, y_train, loss_gradient_w, loss_gradient_b,
                       max_iter=2000,learning_rate=learning_rate)

    print("Train loss:", lr.loss(w, b, x_train, y_train))
    print("Train accuracy:", accuracy(w, b, x_train, y_train))
    print("Validation loss:", lr.loss(w, b, x_test, y_test))
    print("Validation accuracy:", accuracy(w, b, x_test, y_test))
    print('\n')
    
    print('simple_stochastic gd')
    w,b = lr.gen_w_b(dim)
    w,b = op.simple_stoch_gd(w, b, x_train, y_train, loss_gradient_w, loss_gradient_b,
                        max_iter=max_iter,learning_rate=learning_rate,batch_size=batch_size)

    print("Train loss:", lr.loss(w, b, x_train, y_train))
    print("Train accuracy:", accuracy(w, b, x_train, y_train))
    print("Validation loss:", lr.loss(w, b, x_test, y_test))
    print("Validation accuracy:", accuracy(w, b, x_test, y_test))
    print('\n')
    
    
    print('custom stochastic gd')
    w,b = lr.gen_w_b(dim)
    w,b = op.simple_stoch_gd(w, b, x_train, y_train, loss_gradient_w, loss_gradient_b,
                        max_iter=max_iter,learning_rate='random',batch_size=batch_size)
    print("Train loss:", lr.loss(w, b, x_train, y_train))
    print("Train accuracy:", accuracy(w, b, x_train, y_train))
    print("Validation loss:", lr.loss(w, b, x_test, y_test))
    print("Validation accuracy:", accuracy(w, b, x_test, y_test))
    print('\n')
    
    
    
    print('adagrad')
    w,b = lr.gen_w_b(dim)
    w,b = op.adagrad(w, b, x_train, y_train, loss_gradient_w, loss_gradient_b,
                     max_iter=max_iter,learning_rate=learning_rate,batch_size=batch_size)
    print("Train loss:", lr.loss(w, b, x_train, y_train))
    print("Train accuracy:", accuracy(w, b, x_train, y_train))
    print("Validation loss:", lr.loss(w, b, x_test, y_test))
    print("Validation accuracy:", accuracy(w, b, x_test, y_test))
    print('\n')
    
    
    
    print('rmsprop')
    w,b = lr.gen_w_b(dim)
    w,b = op.rmsprop(w, b, x_train, y_train, loss_gradient_w, loss_gradient_b,
                     max_iter=max_iter,learning_rate=learning_rate,batch_size=batch_size,decay_rate=decay_rate)
    print("Train loss:", lr.loss(w, b, x_train, y_train))
    print("Train accuracy:", accuracy(w, b, x_train, y_train))
    print("Validation loss:", lr.loss(w, b, x_test, y_test))
    print("Validation accuracy:", accuracy(w, b, x_test, y_test))
    print('\n')
    
    
    
    print('adam')
    w,b = lr.gen_w_b(dim)
    w,b = op.adam(w, b, x_train, y_train, loss_gradient_w, loss_gradient_b,
                  max_iter=max_iter,learning_rate=learning_rate,beta_1=beta_1,beta_2=beta_2,batch_size=100)
    
    print("Train loss:", lr.loss(w, b, x_train, y_train))
    print("Train accuracy:", accuracy(w, b, x_train, y_train))
    print("Validation loss:", lr.loss(w, b, x_test, y_test))
    print("Validation accuracy:", accuracy(w, b, x_test, y_test))
    print('\n')
    
    
    
    print('newton')
    x_train, y_train, x_test, y_test = readData()


    N_train = x_train.shape[0]
    N_test = x_test.shape[0]
    dim = x_test.shape[1]
    
    
    w,b = lr.gen_w_b(dim)
    loss_gradient = grad(lr.loss, (0, 1))
    w, b = op.newton(w, b, x_train, y_train, loss_gradient, max_iter=1000)
    
    print("Train loss:", lr.loss(w, b, x_train, y_train))
    print("Train accuracy:", accuracy(w, b, x_train, y_train))
    print("Validation loss:", lr.loss(w, b, x_test, y_test))
    print("Validation accuracy:", accuracy(w, b, x_test, y_test))
    print('\n')
    

    
 

if __name__ == '__main__':
    if len(sys.argv) > 1:  
        if sys.argv[1] == 'all':
            run_all_optimizers()
        else:
            print("Unknown command. Running default function.")
            logistic_regression_adam_final()
    else:
        logistic_regression_adam_final()
