import data
import numpy as np
import random
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
random.seed(42)

input_dim=784
hidden=100
global M_1
M_1 = np.eye(hidden,input_dim)
global M_2
M_2 = np.eye(10,hidden)
T = 40000
T_2=10 #inner level running time
s_prop_1=0.001 #Variance of proposal distribution q_1
s_prop_2=0.0005 #Variance of proposal distribution q_2

#Temperature vector:
a_1=2*10**(-6)
a_2=10**(-6)


Hist_train = np.zeros(T) #Training errors
Hist_test = np.zeros(T) #Testing errors

#initialization
W_1 = np.zeros((hidden,input_dim))
W_2 = np.zeros((10,hidden))

def loss(w1,w2):
    global M_1
    global M_2
    lost=np.exp(-neural_net_loss(data.train_x, data.train_y_one_hot,w1+M_1,w2+M_2));
    return lost

def neural_net_loss(input_instance, input_label,V_1,V_2):
    x_1 = np.maximum((V_1@input_instance),np.zeros((np.shape(V_1@input_instance))))
    x_2 = V_2@x_1
    Net_output=x_2
    for i in range(np.max(np.shape(Net_output))):
        temp = 0
        for k in range(np.min(np.shape(Net_output))):
            temp += np.exp(k)
        for j in range(np.min(np.shape(Net_output))):
            Net_output[j][i] = np.exp(x_2[j][i])/temp
    Loss_output = np.sum(np.square(input_label-Net_output),axis=0)
    loss_amount = np.mean(Loss_output)
    return loss_amount


for t in range(T):
    print(t)
    num_A=0
    W_1_hat = W_1 + np.random.normal(0,s_prop_1,size = (hidden,input_dim))
    for u in range(T_2):
        W_2_hat = W_2 + np.random.normal(0,s_prop_2,size = (10,hidden))
        b=(loss(W_1,W_2_hat))/(loss(W_1,W_2)**(1/a_2))
        v = random.random()
        if(v<b):
            W_2 = W_2_hat
            print("W_2 Changed")
        num_A=num_A+(loss(W_1_hat,W_2))/(loss(W_1,W_2)**(1/a_2));
    norm_W_1 = np.linalg.norm(W_1)
    norm_W_2 = np.linalg.norm(W_2)
    train_error=neural_net_loss(data.train_x, data.train_y_one_hot,W_1+M_1,W_2+M_2)
    test_error=neural_net_loss(data.test_x, data.test_y_one_hot,W_1+M_1,W_2+M_2)
    Hist_train[t]=train_error
    Hist_test[t]=test_error
    A=num_A/(T_2)
    C=A**(a_2/(a_1+a_2))
    w = random.random()
    if(w<C):
        W_1=W_1_hat
        print('W_1 Changed')

#Next Step is to Plot data

plt.plot(range(T),Hist_train)
plt.plot(range(T),Hist_test)
