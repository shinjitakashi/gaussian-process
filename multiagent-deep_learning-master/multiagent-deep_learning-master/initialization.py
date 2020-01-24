#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import networkgraph as NetworkGraph

path = os.path.dirname(os.path.abspath(__file__))

def main():
    n = 15
    makeDir()
    makeNetwork(n)
    np.savetxt(path+'/n.dat', [n])


def makeDir():
    if not os.path.isdir(path+'/figs'):
        os.mkdir(path+'/figs')


def makeNetwork(n):
    (G, adjMat, maxdeg) = NetworkGraph.connected_directed_networkgraph(n)
    # (G, adjMat, maxdeg) = NetworkGraph.connected_wattzstrogatz_networkgraph(n,k=4,p=0.4)
    
    pos = nx.circular_layout(G)
    nx.draw(G, pos, font_size=8)
    plt.savefig(path+"/figs/graph_n"+str(n)+".png")
    plt.savefig(path+"/figs/graph_n"+str(n)+".pdf")
    plt.show()

    np.savetxt(path+'/Graph_adjMat.dat', adjMat)
    np.savetxt(path+'/Graph_maxdeg.dat', [maxdeg])


if __name__ == "__main__":
    main()