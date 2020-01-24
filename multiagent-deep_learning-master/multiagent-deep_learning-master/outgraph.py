# cording: utf-8

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import networkx as nx

path = os.path.dirname(os.path.abspath(__file__))

(n) = np.loadtxt(path+'/n.dat').astype(int)
(maxdeg) = np.loadtxt(path+'/Graph_maxdeg.dat')
Gadj = np.loadtxt(path+'/Graph_adjMat.dat')

iter_per_epoch = np.loadtxt(path+'/iterperepoch.dat')
(maxepoch) = np.loadtxt(path+'/maxepoch.dat').astype(int)

suffix = "_withCom"
train_acc_com = np.loadtxt(path+'/train_acc_list'+suffix+'.dat')
test_acc_com = np.loadtxt(path+'/test_acc_list'+suffix+'.dat')
train_loss_com = np.loadtxt(path+'/train_loss_list'+suffix+'.dat')

suffix = "_withoutCom"
train_acc_sin = np.loadtxt(path+'/train_acc_list'+suffix+'.dat')
test_acc_sin = np.loadtxt(path+'/test_acc_list'+suffix+'.dat')
train_loss_sin = np.loadtxt(path+'/train_loss_list'+suffix+'.dat')


#####

### Do not use type-3-fonts.
# plt.rcParams['ps.useafm'] = True
# plt.rcParams['pdf.use14corefonts'] = True
# plt.rcParams['text.usetex'] = True

font = {# 'family' : 'serif',
        # 'weight' : 'bold',
        'size'   : 15
        } 

plt.rc('font', **font)

ls = ['-', '--', '-.', ':', '-', '--', '-.', ':', ]

#####

plt.figure(figsize=(6,6))
G = nx.from_numpy_matrix(Gadj, create_using=nx.DiGraph)
pos = nx.circular_layout(G)
nx.draw(G, pos, font_size=8)
plt.savefig(path+"/figs/graph_n"+str(n)+".png")
plt.savefig(path+"/figs/graph_n"+str(n)+".pdf")
plt.show()


###

markers = {'train': 'o', 'test': 's'}

plt.figure(figsize=(8,6))
x = np.arange(maxepoch)
plt.plot(x, train_acc_com, marker='o', color='darkred', label='communicate_train', markevery=10, linestyle=ls[0])
plt.plot(x, test_acc_com, marker='s', color='indianred', label='communicate_test', markevery=10, linestyle=ls[1])
plt.plot(x, train_acc_sin, marker='o', color='midnightblue', label='single_train', markevery=10, linestyle=ls[2])
plt.plot(x, test_acc_sin, marker='s', color='slateblue', label='single_test', markevery=10, linestyle=ls[3])
plt.xlabel("epochs k")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
### x軸に100刻みにで小目盛り(minor locator)表示
# plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(50))
### y軸に1刻みにで小目盛り(minor locator)表示
plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(0.2))
### 小目盛りに対してグリッド表示
plt.grid(which='minor', axis='y')
plt.legend(loc='lower right')
plt.savefig(path+"/figs/accuracy.png")
plt.savefig(path+"/figs/accuracy.pdf")
plt.show()

###

plt.figure(figsize=(8,6))
x = np.arange(iter_per_epoch * (maxepoch-1) +1)
plt.plot(x, train_loss_com, color='darkred', label='communicate', linewidth=2.0, linestyle=ls[0])
plt.plot(x, train_loss_sin, color='midnightblue', label='single', linewidth=2.0, linestyle=ls[2])
plt.xlabel("iterations")
plt.ylabel("loss (mean squared error)")
### y軸に1刻みにで小目盛り(minor locator)表示
plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(200))
### 小目盛りに対してグリッド表示
plt.grid(which='minor', axis='y')
# plt.ylim(0, 1.0)
plt.legend(loc='upper right')
plt.savefig(path+"/figs/loss.png")
plt.savefig(path+"/figs/loss.pdf")
plt.show()