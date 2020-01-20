import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LN
import networkx as nx
import copy
from scipy.optimize import minimize

class Kernel:
    def __init__(self,param,bound=None):
        """パラメータをカーネル変数に入れる
            param (np.array) : ハイパーパラメータ
            bound (np.array) : ハイパーパラメータの範囲
        """
        self.param = np.array(param)
        if(bound==None):
            bound = np.zeros([len(param),2])
            bound[:,1] = np.inf
        self.bound = np.array(bound)

    def __call__(self,x1,x2) -> float:
        """ ガウスカーネルを計算する。
            k(x1, x2) = a1*exp(-s*|x - x2|^2)

        Args:
            x1 (np.array)   : 入力値1
            x2 (np.array)   : 入力値2
            param (np.array): ガウスカーネルのパラメータ

        Returns:
            float: ガウスカーネルの値
        """
        a1,s,a2 = self.param
        return a1**2*np.exp(-0.5*((x1-x2)/s)**2) + a2**2*(x1==x2)

class Gausskatei_agent:
    """実際にガウス過程のハイパーパラメータ推定を行うエージェント
    1: kernelをinitializationにて，self.kernelに入れる．また，最適化に必要な各パラメータも同様にclass内に代入する．
    2: N個のエージェントをつくり，協調的に最適化を行っていく．
    
    
    """
    def __init__(self, kernel, N: int, n: int, weight: np.array, name: int, stepsize: float, eventtrigger: float):
        self.kernel = kernel
        self.N = N #agentの数
        self.n = n #agentの持つ変数の次元
        self.name = name
        self.weight = weight
        self.stepsize = stepsize
        self.eventtrigger = eventtrigger

        self.initial_state()

    #Initialization : agentの初期状態を決定する
    def initial_state(self):
        self.theta_i = self.kernel.param #agentのcost functionの決定変数
        self.theta = np.zeros([self.N, self.n])
        self.theta_send = np.zeros([self.N, self.n])
    
    

    def gakushuu(self,x0: np.array, y0: np.array):
        """カーネル行列: Kを計算する

        Args:
            x0 (np.array) : 既知のデータx0
            y0 (np.array) : 既知のデータy0
        """
        self.x0 = x0
        self.y0 = y0
        self.k00 = self.kernel(*np.meshgrid(x0,x0))
        self.k00_1 = np.linalg.inv(self.k00)

    def yosoku(self,x: np.array) -> np.array:
        """
        
        Args:
            k00_1 (np.array)    : K00の逆行列
            x0 (np.array)       : 既知のデータx0(N)
            x (np.array)        : 未知の入力データx(M)
            k10 (np.array)      : N×Mのカーネル行列
            k11 (np.array)      : M×Mのカーネル行列

        return:
            mu (np.array)       : 平均行列 (M×1)
            std (np.array)      : 標準偏差行列 (M×1) 
        """
        k00_1 = self.k00_1
        k01 = self.kernel(*np.meshgrid(self.x0,x,indexing='ij'))
        k10 = k01.T
        k11 = self.kernel(*np.meshgrid(x,x))

        mu = k10.dot(k00_1.dot(self.y0))
        sigma = k11 - k10.dot(k00_1.dot(k01))
        std = np.sqrt(sigma.diagonal())
        return mu,std

    def logyuudo(self,param=None): # 対数尤度
        if(param is None):
            k00 = self.k00
            k00_1 = self.k00_1
        else:
            self.kernel.param = param
            k00 = self.kernel(*np.meshgrid(self.x0,self.x0))
            k00_1 = np.linalg.inv(k00)
        return -(np.linalg.slogdet(k00)[1]+self.y0.dot(k00_1.dot(self.y0)))
    
    #Compute the step-size
    def step_size(self, t: int, stepsize: float):
        return stepsize / (t+1)

    #Send the state to the neighbor agents　
    def send(self, j: int):
        self.theta_send[j] = self.theta_i
        return self.theta_i, self.name
    
    #Receive the states from the neighbor agents
    def receive(self, theta_j: np.array, name: int):
        self.theta[name] = theta_j
        
    #Compute the event_trigger_figure(閾値) eventtrigger / (t+1)
    def event_trigger(self, t: int, eventrigger: float):
        return eventrigger / (t + 1)
    
    def kgrad (self, xi: np.array ,xj: np.array ,d) -> float:
        """アルゴリズムに必要な勾配

        Args:
            d (int)     : thetaの次元

        return:
            勾配 (int)  : 勾配
        """
        if d == 0:
            return 2*self.theta_i[0]*np.exp(-0.5*self.theta_i[1]*np.linalg.norm(xi-xj)**2)
        elif d == 1:
            return self.theta_i[0]**2*np.exp(-0.5*(np.linalg.norm(xi-xj)/self.theta_i[1])**2)*(-(np.linalg.norm(xi-xj)/self.theta_i[1]))*(-np.linalg.norm(xi-xj)/self.theta_i[1]**2)
        elif d == 2:
            return (xj==xi)

    def kernel_matrix_grad(self, xd: np.array) -> np.array:
        """各ハイパーパラメータに対するカーネル行列の勾配の計算
        Args:
            grad_K (np.array) : カーネルの勾配行列(len(xd)×len(xd)×3)
        """
        self.grad_K = np.zeros((len(xd), len(xd), 3))
        
        for i in range(len(xd)):
            for j in range(len(xd)):
                for q in range(3):
                    self.grad_K[i][j][q] = self.kgrad(xd[i], xd[j], q)
    
    def grad_optim(self, xd: np.array, y: np.array) -> np.array:
        """目的関数の勾配
        Args:
            KD_00 (np.array) : カーネル行列
        """
        KD_00 = self.kernel(*np.meshgrid(xd,xd))
        try:
            KD_00_1 = np.linalg.inv(KD_00)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('疑似逆行列つかったよ')
                KD_00_1 = np.linalg.pinv(KD_00)
            
        self.kernel_matrix_grad(xd)
        
        self.grad = np.zeros(3)

        for d in range(3):
            self.grad[d] = -np.trace(KD_00_1 @ self.grad_K[:,:,d]) + (KD_00_1 @ y).T @ self.grad_K[:,:,d] @ (KD_00_1 @ y)

    def saitekika(self,xd: np.array, yd: np.array, t: int): # パラメータを調整して学習
        """ハイパーパラメータの最適化
        x_i(t+1) = x_i(t) + Σp_ij(x_ji(t)-x_ij(t)) - a(t)∇f_i(x_i(t))

        Args:
            xd (np.array)       : エージェントiに与えられる最適化用のデータセット
            yd (np.array)        : エージェントiに与えられる最適化用のデータセット
            param (np.array)    : パラメータ
            stepsize (float)    : ステップサイズ関数
            grad_f (np.array)   : 勾配(3×1)
            theta (np.array)    : Θ(3×1)
            x0 (np.array)       : エージェントが保有する予測用のデータセット
        """
        #update
        self.diff = self.theta - self.theta_send
        self.grad_optim(xd, yd)
        self.theta_i = self.theta_i + np.dot(self.weight, self.diff) - self.step_size(t, self.stepsize) * self.grad
        #他のAgentとの通信
        self.theta_send[self.name] = self.theta_i
        self.theta[self.name] = self.theta_i
        #新たなカーネルの更新，カーネル行列の計算
        self.kernel.param = self.theta_i
        self.k00 = self.kernel(*np.meshgrid(x0,x0))
        try:
            self.k00_1 = np.linalg.inv(self.k00)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('疑似逆行列つかったよ')
                self.k00_1 = np.linalg.pinv(self.k00)

#Parameters
#Number of agents
N = 3

#Number of dimensions of the decision variable
n = 3

#Coefficient of decision of stepsize : a(t) = a / t
stepsize = 0.01
        
# Coefficient of the edge weight  w_if = wc / max_degree
wc = 0.8

#Number of iterations
iteration = 30

# Interval for figure plot 
fig_interval = 200

#Coefficient of decision of stepsize : E_ij(t) = E(t) = eventtrigger / (t+1)
eventtrigger = 0

# Randomization seed
# np.random.seed(9)
# ========================================================================================================================== #
# Communication Graph
A = np.array(
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]])

G = nx.from_numpy_matrix(A)

# Weighted Stochastic Matrix P
a = np.zeros(N)

for i in range(N):
    a[i] = copy.copy(wc / nx.degree(G)[i])

P = np.zeros((N, N))
for i in range(N):
    for j in range(i, N):
        if i != j and A[i][j] == 1:
            a_ij = min(a[i], a[j])
            P[i][j] = copy.copy(a_ij)
            P[j][i] = copy.copy(a_ij)

for i in range(N):
    sum = 0.0
    for j in range(N):
        sum += P[i][j]
    P[i][i] = 1.0 - sum


def y(x): # 実際の関数
    return 5*np.sin(np.pi/15*x)*np.exp(-x/50)

#gp_Agentsを作成
Kernel_array = []
Gp_Agent_array = []
bound = [[1e-2,1e2],[1e-2,1e2],[1e-2,1e2]]
param0 = [[1.6,4.5,3.4], [2.8, 4.3, 0.6], [2.8, 0.95, 2]]
find_point = 100 # 既知の点の数
x0 = np.random.uniform(0,100,find_point) # 既知の点
y0 = y(x0) + np.random.normal(0,1,find_point)
for i in range(N):
    np.random.seed(i)
    Kernel_array.append(Kernel(param0[i], bound))
    Gp_Agent_array.append(Gausskatei_agent(Kernel_array[i], N, n, P[i], i, stepsize, eventtrigger))
    Gp_Agent_array[i].xd = np.random.uniform(0, 100, 50) #最適化用のデータセット
    Gp_Agent_array[i].yd = y(Gp_Agent_array[i].xd) + np.random.normal(0, 1, 50)
    Gp_Agent_array[i].gakushuu(x0, y0)


Agents = copy.deepcopy(Gp_Agent_array)

for i in range(N):
    for j in range(N):
        if i!=j and A[i][j]==1:
            #Send the state to the neighbor agents at initial time 
                state, name = Agents[i].send(j)

                #Receive the state from the neighbor agents at initial time
                Agents[j].receive(state, name)

for i in range(N):
    print('Agents{}'.format(i))
    print('a=%.6f, s=%.6f, w=%.6f'%tuple(Agents[i].kernel.param))

for t in range(iteration):
    print('{}th iteration'.format(t))
    # Transfer data among agents
    for i in range(N):
        for j in range(N):
            if i != j and A[i][j] == 1:
                if LN.norm(Agents[i].theta_i - Agents[i].theta_send[j], ord=1) > Agents[i].event_trigger(t+1, Agents[i].eventtrigger):
                    #Send the state to the neighbor agents
                    state, name = Agents[i].send(j)
                    #Receive the state from the neighbor agents
                    Agents[j].receive(state, name)

    #Update the state
    for i in range(N):
        Agents[i].saitekika(sorted(Agents[i].xd), sorted(Agents[i].yd), t)
        
for i in range(N):
    print('Agents{}'.format(i))
    print('a=%.6f, s=%.6f, w=%.6f'%tuple(Agents[i].kernel.param))

plt.figure(figsize=[5,8])
x1 = np.linspace(0,100,600) 
for i in range(N):
    print('Agent {}'.format(i))
    plt.plot(Gp_Agent_array[i].x0, Gp_Agent_array[i].y0, '. ')
    mu, std = Agents[i].yosoku(x1)
    plt.plot(x1,y(x1),'--r')
    plt.plot(x1,mu,'g')
    plt.fill_between(x1,mu-std,mu+std,alpha=0.2,color='g')
    plt.title('a=%.6f, s=%.6f, w=%.6f'%tuple(Agents[i].kernel.param))
    plt.show()