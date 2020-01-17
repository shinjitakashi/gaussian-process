import numpy as np
import matplotlib.pyplot as plt

class Kernel:
    def __init__(self,param,bound=None):
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

class Gausskatei:
    def __init__(self,kernel):
        self.kernel = kernel

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

    def kgrad (self, xi: np.array ,xj: np.array ,d) -> float:
        """アルゴリズムに必要な勾配

        Args:
            d (int)     : thetaの次元

        return:
            勾配 (int)  : 勾配
        """
        theta = self.kernel.param
        if d == 0:
            return 2*theta[0]*np.exp(-0.5*theta[1]*np.linalg.norm(xi-xj)**2)
        elif d == 1:
            return theta[0]**2*np.exp(-0.5*(np.linalg.norm(xi-xj)/theta[1])**2)*(-(np.linalg.norm(xi-xj)/theta[1]))*(-np.linalg.norm(xi-xj)/theta[1]**2)
        elif d == 2:
            return (xj==xi)

    def kernel_matrix_grad(self, xd: np.array) -> np.array:
        self.grad_K = np.zeros((len(xd), len(xd), 3))
        
        for i in range(len(xd)):
            for j in range(len(xd)):
                for q in range(3):
                    self.grad_K[i][j][q] = self.kgrad(xd[i], xd[j], q)
    
    def grad_optim(self, xd: np.array, y: np.array) -> np.array: 
        KD_00 = self.kernel(*np.meshgrid(xd,xd))
        KD_00_1 = np.linalg.inv(KD_00)

        self.kernel_matrix_grad(xd)
        
        self.grad = np.zeros(3)

        for d in range(3):
            self.grad[d] = np.trace(KD_00_1 @ self.grad_K[:,:,d]) - (KD_00_1 @ y).T @ self.grad_K[:,:,d] @ (KD_00_1 @ y)

    def saitekika(self,xd: np.array, y: np.array, kurikaeshi=1000): # パラメータを調整して学習
        """ハイパーパラメータの最適化
        x_i(t+1) = x_i(t) + Σp_ij(x_ji(t)-x_ij(t)) - a(t)∇f_i(x_i(t))

        Args:
            xD (np.array)       : エージェントiに与えられるデータセット
            y (np.array)        : エージェントiに与えられるデータセット
            param (np.array)    : パラメータ
            stepsize (float)    : ステップサイズ関数
            grad_f (np.array)   : 勾配(3×1)
            theta (np.array)    : Θ(3×1)

        Return:
            next_theta (np.array)   : optimization_theta (3×1)
        """

        self.theta = self.kernel.param

        for t in range(kurikaeshi):
            self.grad_optim(xd, y)
            self.theta = self.theta - (0.01/(t+1))*self.grad

        self.kernel.param = self.theta
        self.k00 = self.kernel(*np.meshgrid(x0,x0))
        self.k00_1 = np.linalg.inv(self.k00)

def y(x): # 実際の関数
    return 5*np.sin(np.pi/15*x)*np.exp(-x/50)

n = 100 # 既知の点の数
x0 = np.random.uniform(0,100,n) # 既知の点
y0 = y(x0) + np.random.normal(0,1,n)
param0 = [2,0.4,1.5] # パラメータの初期値
bound = [[1e-2,1e2],[1e-2,1e2],[1e-2,1e2]] # 下限上限
kernel = Kernel(param0,bound)
x1 = np.linspace(0,100,200) #予測用のデータ
gp = Gausskatei(kernel)
gp.gakushuu(x0, y0)
plt.figure(figsize=[5,8])
for i in [0,1]:
    if(i):
        gp.saitekika(x0,y0,1000) # パラメータを調整する
    plt.subplot(211+i)
    plt.plot(x0,y0,'. ')
    mu,std = gp.yosoku(x1)
    plt.plot(x1,y(x1),'--r')
    plt.plot(x1,mu,'g')
    plt.fill_between(x1,mu-std,mu+std,alpha=0.2,color='g')
    plt.title('a=%.3f, s=%.3f, w=%.3f'%tuple(gp.kernel.param))
plt.tight_layout()
plt.show()