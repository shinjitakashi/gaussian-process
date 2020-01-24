# cording: utf-8

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from retrying import retry
from common.cyclelist import cycle

@retry(stop_max_attempt_number=10)
def connected_directed_networkgraph(n=20):
    G = nx.DiGraph()
    nodes = [i for i in range(1,n+1)]
    cyclenodes = cycle(nodes)
    G.add_nodes_from(nodes)   # add n nodes

    # ## agent につき1～2本の in/out edge を生成
    # inum = np.random.choice((1,2,),1)[0]
    # onum = np.random.choice((1,2,),1)[0]
    for i in range(0,n):
        ## agent につき1～2本の in/out edge を生成
        inum = np.random.choice((1,2,3,),1)[0]
        onum = np.random.choice((1,2,3,),1)[0]
        ## 各agentの周辺4 nodeへランダムに edge 生成
        neighbors = cyclenodes.forward(i,4) + cyclenodes.backward(i,4)
        ineighbors = np.random.choice(neighbors,inum,replace=False)
        oneighbors = np.random.choice(neighbors,onum,replace=False)
        for j in ineighbors:
            G.add_edge(nodes[i],j)
        for j in oneighbors:
            G.add_edge(j,nodes[i])
    
    if not nx.is_strongly_connected(G):
        raise Exception()
    else:
        print("connected")

    (adjMat, maxdeg) = __network_constructure(G)

    return (G, adjMat, maxdeg)

def connected_wattzstrogatz_networkgraph(n, k=3, p=0.4, s=None):
    WSG = nx.connected_watts_strogatz_graph(n,k,p,tries=100,seed=s)
    (adjMat, maxdeg) = __network_constructure(WSG)
    return (WSG, adjMat, maxdeg)

def __network_constructure(G):
    adjMat = np.array(nx.to_numpy_matrix(G)).T
    if type(G) is nx.MultiDiGraph or type(G) is nx.DiGraph:
        maxdeg = max(dict(G.in_degree).values())
    else:
        maxdeg = max(dict(G.degree).values())
    return (adjMat, maxdeg)

def __main():
    G = connected_directed_networkgraph()
    
    pos = nx.circular_layout(G)
    nx.draw(G, pos, font_size=8)
    # plt.savefig(path+"/graph_n20.png")
    # plt.savefig(path+"/graph_n20.pdf")
    plt.show()

if __name__ == "__main__":
    __main()