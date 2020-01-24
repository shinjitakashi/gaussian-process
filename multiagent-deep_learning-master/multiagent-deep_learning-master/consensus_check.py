# cording: utf-8

"""
consensus_check.py

[通信を行うことに際して，正しいアドレスに値が格納されているか] の確認を行うファイルです．
ソースコードの一部を変更したとき，又は上手く学習を行わない場合に等にご活用ください．
"""

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import networkgraph
from agent import Agent
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

path = os.path.dirname(os.path.abspath(__file__))

print("start")

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
n = 4
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


## start test
params = agents[0].layer.params.copy()
ano_params = agents[1].layer.params.copy()

for i in range(n):
    for j in np.nonzero(agents[i].AdjG)[0]:
        agents[j].receive(i, *agents[i].send(i,j))
agents[0].consensus()
con_params = agents[0].layer.params.copy()

### age0 と age1 の各パラメータが違う値であることを確認します．
print("age0 vs age1")
for key in params.keys():
    diff = np.average( np.abs(params[key] - ano_params[key]) )
    print(key + ":" + str(diff))
### age0 が他のエージェントと通信を行った場合，age0 のパラメータが変化することを確認します．
print("age0 vs consensused age0")
for key in params.keys():
    diff = np.average( np.abs(params[key] - con_params[key]) )
    print(key + ":" + str(diff))
### 通信後の age0 のパラメータが，通信前の age1 のパラメータと異なることを確認します．
print("age1 vs consensused age0")
for key in params.keys():
    diff = np.average( np.abs(ano_params[key] - con_params[key]) )
    print(key + ":" + str(diff))