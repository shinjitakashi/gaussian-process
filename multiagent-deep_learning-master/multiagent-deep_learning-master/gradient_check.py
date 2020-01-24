# cording: utf-8

"""
gradient_check.py

[誤差逆伝搬により，正しく勾配が求められ，正しいアドレスに格納できているか] を確認するファイルです．
上手く更新が行われない場合の原因究明にお役立てください．
"""

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import networkgraph
from agent import Agent
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD

path = os.path.dirname(os.path.abspath(__file__))

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
n = 1
maxdeg = n-1
Gadj = np.ones([n,n]) - np.eye(n)

x_train_split = np.split(x_train, n)
t_train_split = np.split(t_train, n)

max_epochs = 101
each_train_size = x_train_split[0].shape[0]
batch_size = min(100, each_train_size)

Agent.n = n
Agent.maxdeg, Agent.AdjG_init = maxdeg, Gadj
Agent.train_size, Agent.batch_size = each_train_size, batch_size

weight_decay_lambda = 0 

agents = [Agent(idx, x_train_split[idx], t_train_split[idx], x_test, t_test, 
                SGD(lr=lambda s:0.01), weight_decay_lambda) for idx in range(n)]

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(each_train_size / batch_size, 1)
epoch_cnt = 0

#####################

grad_numerical = agents[0].degub_numericalGrad()
grad_backprop  = agents[0].debug_backpropGrad()

### 数値勾配と誤差逆伝搬により求めた勾配が一致していることを確認します．
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))

######################

# network = MultiLayerNetExtend(input_size=784, hidden_size_list=[50], output_size=10,
#             weight_decay_lambda=0,
#             use_dropout=False, dropout_ration=0.0, use_batchnorm=False)
# x_batch = x_train[:3]
# t_batch = t_train[:3]
# grad_numerical = network.numerical_gradient(x_batch,t_batch)
# grad_backprop = network.gradient(x_batch,t_batch)

# for key in grad_numerical.keys():
#     diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
#     print(key + ":" + str(diff))

#######################