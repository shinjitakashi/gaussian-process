# cording: utf-8

import numpy as np
from collections import OrderedDict
import copy

# from common.multi_layer_net import MultiLayerNet
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD

class Agent:

    """ 各エージェントの動作を main からこっちに書き写す形で． """

    n = int()
    AdjG_init = np.zeros((n,n)) #require to be {0 or 1} to all arguments
    WeiG_init = np.zeros((n,n))
    maxdeg = int()

    train_size = 0
    batch_size = 100

    # weight graph 作成規則 =====
    # wtype = "maximum-degree"
    wtype = "local-degree"
    # ===========================
    
    def __init__(self, idx, x_train, t_train, x_test, t_test, optimizer, weight_decay_lambda=0.0):
        self.idx = idx
        # self.layer = MultiLayerNet(input_size=784, hidden_size_list=[100], output_size=10,
        #                             weight_decay_lambda=weight_decay_lambda)
        self.layer = MultiLayerNetExtend(input_size=784, hidden_size_list=[50,50,50], output_size=10,
                                    weight_decay_lambda=weight_decay_lambda,
                                    use_dropout=False, dropout_ration=0.0, use_batchnorm=False)
        self.optimizer = optimizer
        self.rec_param = np.array([{} for i in range(self.n)])

        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

        self.z_vec = np.zeros(self.n)
        self.z_vec[idx] = 1
        self.rec_z = np.zeros((self.n,self.n))

        self.AdjG = np.zeros(self.n) #require to be {0 or 1} to all arguments
        self.WeiG = np.zeros(self.n)
        if np.all(self.WeiG_init == 0):
            self.makeWeiGraph(self.AdjG_init)
        else:
            self.WeiG = self.WeiG_init[self.idx]
        self.makeAdjGraph()

        self.train_loss = 0
        self.train_acc = 0
        self.test_acc = 0

    def send(self, k, agent):
        """sending params to other nodes (return "self.layer.params"): send(agent)"""
        return (self.layer.params.copy(), self.z_vec.copy())

    def receive(self, agent, getparams, getz):
        """receiving other node's params: receive(agent, new_params)"""
        self.rec_param[agent] = getparams.copy()
        self.rec_z[agent] = getz.copy()

    def selectData(self, train_size, batch_size):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        return x_batch, t_batch

    def consensus(self):
        self.weightConsensus()
        self.subvalConsensus()

    def weightConsensus(self):
        for key in self.layer.params.keys():
            self.layer.params[key] *= self.WeiG[self.idx]
            for idn in np.nonzero(self.AdjG)[0]:
                self.layer.params[key] += self.WeiG[idn] * self.rec_param[idn][key]

    def subvalConsensus(self):
        self.rec_z[self.idx] = self.z_vec
        self.z_vec = np.dot(self.WeiG, self.rec_z)

    def update(self, k=1):
        x_batch, t_batch = self.selectData(self.train_size, self.batch_size)
        grads = self.layer.gradient(x_batch, t_batch)
        self.optimizer.update(self.layer.params, grads, self.z_vec[self.idx], k)
        # self.optimizer.update(self.layer.params, grads)

    def calcLoss(self):
        self.train_acc = self.layer.accuracy(self.x_train, self.t_train)
        self.test_acc = self.layer.accuracy(self.x_test, self.t_test)
        self.train_loss = self.layer.loss(self.x_train, self.t_train)

    def makeAdjGraph(self):
        """make Adjecency Graph"""
        self.AdjG = self.AdjG_init[self.idx]
    
    def makeWeiGraph(self, lAdjG): 
        """make Weight matrix"""
        if self.n is 1:
            tmpWeiG = np.ones([1])
        else:
            if self.wtype == "maximum-degree":
                tmpWeiG = (1 / (self.maxdeg+1)) * lAdjG[self.idx]
                tmpWeiG[self.idx] = 1 - np.sum(tmpWeiG)
            elif self.wtype == "local-degree":
                ### count degrees ###
                #degMat = np.kron(np.dot(lAdjG,np.ones([self.n,1])), np.ones([1,self.n]))
                degMat = np.kron(np.dot(lAdjG,np.ones([self.n,1]))+1, np.ones([1,self.n]))
                ### take max() for each elements ###
                degMat = np.maximum(degMat, degMat.T)
                ### divide for each elememts ###
                tmpAllWeiG = lAdjG / degMat
                selfDegMat = np.eye(self.n) - np.diag(np.sum(tmpAllWeiG,axis=1))
                tmpAllWeiG = tmpAllWeiG + selfDegMat
                tmpWeiG = tmpAllWeiG[self.idx,:]
            else:
                try:
                    raise ValueError("Error: invalid weight-type")
                except ValueError as e:
                    print(e)
        self.WeiG = tmpWeiG
        


    ########
    # debugging functions
    ########
    def degub_numericalGrad(self):
        return self.layer.numerical_gradient(self.x_train[:3], self.t_train[:3])
    
    def debug_backpropGrad(self):
        return self.layer.gradient(self.x_train[:3], self.t_train[:3])

    def debug_consensus(self):
        params = self.layer.params.copy()
        self.weightConsensus()
        self.subvalConsensus()
        if self.idx == 0:
            ano_params = self.layer.params.copy()
            for key in params.keys():
                diff = np.average( np.abs(params[key] - ano_params[key]) )
                print(key + ":" + str(diff))
