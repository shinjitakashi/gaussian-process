#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 09:35:30 2019

@author: makoto_
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))

def main():
    (n) = np.loadtxt(path+'/n.dat').astype(int)

    Adj = np.loadtxt(path+'/Graph_adjMat.dat')
    G = nx.from_numpy_matrix(Adj, create_using=nx.MultiDiGraph())
    # pos = nx.spring_layout(G, iterations=200)
    pos = nx.circular_layout(G)

    plt.figure()
    labels = {}
    for i in range(n):
        labels[i] = r"{0}".format(i+1)

    nx.draw(G, pos, font_size=15, labels=labels, node_color="lightblue", edgecolors="#6fbbd3", node_size=450)
    # nx.draw_networkx_nodes(G, pos, node_size=20, alpha=1.0, node_color="lightblue")
    # nx.draw_networkx_edges(G, pos, width=2)
    # nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.savefig(path+"/figs/graph_n20_labeled.png")
    plt.savefig(path+"/figs/graph_n20_labeled.pdf")
    plt.show()

if __name__ == "__main__":
    main()