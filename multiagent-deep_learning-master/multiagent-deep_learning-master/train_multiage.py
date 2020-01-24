# cording: utf-8

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import networkgraph
from agent import Agent
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

# from progressbar import ProgressBar

path = os.path.dirname(os.path.abspath(__file__))


"""
start:FLAGS
"""

flag_cmpltGraph = 0
flag_staticStep = 0
flag_communicate = 1
flag_overfittest = 0

"""
end:FLAGS
"""

if flag_cmpltGraph:
    n = 8
    maxdeg = n-1
    Gadj = np.ones([n,n]) - np.eye(n)
else:
    (n) = np.loadtxt(path+'/n.dat').astype(int)
    (maxdeg) = np.loadtxt(path+'/Graph_maxdeg.dat')
    Gadj = np.loadtxt(path+'/Graph_adjMat.dat')

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

if flag_overfittest:
    ### overfit test
    datasize= min(t_train.size, 2500*n)
    x_train = x_train[:datasize]
    t_train = t_train[:datasize]
    ### split data
    x_train_split = np.split(x_train, n)
    t_train_split = np.split(t_train, n)
else:
    ### split data
    x_train_split = np.split(x_train, n)
    t_train_split = np.split(t_train, n)

# max_epochs = 101
max_epochs = 201
each_train_size = x_train_split[0].shape[0]
batch_size = min(100, each_train_size)
np.savetxt(path+'/max_epochs.dat', [max_epochs])

Agent.n = n
Agent.maxdeg, Agent.AdjG_init = maxdeg, Gadj
Agent.train_size, Agent.batch_size = each_train_size, batch_size

# weight decay（荷重減衰）の設定 =====================
# 要は正則化パラメータ ===============================
# weight_decay_lambda = 0 # weight decayを使用しない場合
weight_decay_lambda = 0.03
# ====================================================

if flag_staticStep:
    optimizer = SGD(lr=lambda s:0.01)
else:
    optimizer = SGD(lr=lambda s:0.04*1/s)

agents = [Agent(idx, x_train_split[idx], t_train_split[idx], x_test, t_test, 
                optimizer, weight_decay_lambda) for idx in range(n)]

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(each_train_size / batch_size, 1)
epoch_cnt = 0

# p = ProgressBar(0,max_epochs-1)
print("START")

for k in range(1000000000):
    # ## communication
    # if flag_communicate:
    #     for i in range(n):
    #         for j in np.nonzero(agents[i].AdjG)[0]:
    #             agents[i].receive(j, *agents[j].send(k,i))
    #     for i in range(n):
    #         agents[i].consensus()

    for i in range(n):
        agents[i].update(epoch_cnt+1)

    train_loss_age = [agents[i].train_loss for i in range(n)]
    train_loss = np.mean(train_loss_age)
    train_loss_list.append(train_loss)

    if k % iter_per_epoch == 0:
        if flag_communicate:
            ## communication
            for i in range(n):
                for j in np.nonzero(agents[i].AdjG)[0]:
                    agents[i].receive(j, *agents[j].send(k,i))
            for i in range(n):
                agents[i].consensus()

        for i in range(n):
            agents[i].calcLoss()
        
        train_acc_age = [agents[i].train_acc for i in range(n)]
        test_acc_age = [agents[i].test_acc for i in range(n)]
        train_acc = np.mean(train_acc_age)
        test_acc = np.mean(test_acc_age)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))
        # for i in range(n):
        #     print("  " + "age" + str(i) + ":" + "train acc:" + str(train_acc_age[i]) + ", test acc:" + str(test_acc_age[i]))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break
        # p.update(epoch_cnt)

# p.finish()

if flag_communicate:
    suffix = "_withCom"
else:
    suffix = "_withoutCom"

np.savetxt(path+'/train_acc_list'+suffix+'.dat', train_acc_list)
np.savetxt(path+'/test_acc_list'+suffix+'.dat', test_acc_list)
np.savetxt(path+'/train_loss_list'+suffix+'.dat', train_loss_list)
np.savetxt(path+'/iterperepoch.dat', [iter_per_epoch])

markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.figure()
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs k")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig(path+"/accuracy"+suffix+".png")
plt.show()

plt.figure()
x = np.arange(iter_per_epoch * (max_epochs-1) +1)
plt.plot(x, train_loss_list, label='train', linewidth=1.5, marker='o', markevery=iter_per_epoch)
plt.xlabel("iterations")
plt.ylabel("loss (mean squared error)")
# plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig(path+"/loss"+suffix+".png")
plt.show()

z = np.array([agents[i].z_vec for i in range(n)])
print('z:')
print(z)
W = np.array([agents[i].WeiG for i in range(n)])
print('Weights:')
print(W)