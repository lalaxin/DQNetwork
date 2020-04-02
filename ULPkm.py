from scipy import optimize as opt
import numpy as np
from scipy.optimize import minimize
import math
import copy
from km2 import km

class ulpkm:
    def __init__(self,user,region,celllength,B,pb,pk):
        # 预算约束
        self.B=B
        self.user=user
        self.region=region
        # 计算的每个用户的pi
        self.pi=np.zeros(len(user))
        # 求解值
        self.x=np.zeros(len(user)*2)
        # 求解初始值
        self.x0 = np.zeros(len(user) * 2)
        self.celllength=celllength
        # 边界设置
        self.b=(0,celllength*math.pow(len(region),1/2))
        self.bnds=np.zeros(len(user))
        # 约束条件个数
        self.cons=np.zeros(len(user)+1,dtype=dict)
        self.pB=pb
        self.k=pk

        self.kmuser=copy.deepcopy(user)
        init_region=copy.deepcopy(region)
        psokm = km(region=init_region, user=self.kmuser)
        psokm.build_graph()
        self.kmuser = psokm.KM()


    #     设置边界
        for i in range (len(user)):
            self.bnds=self.b
    # 设置初始值，初始值为每个用户终点的位置
        for i in range (len(user)):
            self.x0[2*i]=self.user[i][2]
            self.x0[2*i+1]=self.user[i][3]
        for i in range (len(user)):
            self.x[2*i]=self.user[i][2]
            self.x[2*i+1]=self.user[i][3]

    # 目标函数
    def objective(self):
        temp=0
        for i in range (len(self.kmuser)):
            if(self.kmuser[i][5]!=-1 and self.kmuser[i][6]!=-1):
                temp+=math.pow(self.x[2*i]-self.kmuser[i][5],2)+math.pow(self.x[2*i+1]-self.kmuser[i][6],2)
        return lambda x: temp

    #     约束条件0:满足预算约束限制
    def constraint(self):
        temp=0
        print("lalala")
        for i in range (len(self.kmuser)):
            # 用户匹配缺车匹配成功
            if(self.kmuser[i][5]!=-1 and self.kmuser[i][6]!=-1):
                lslarr = math.sqrt(math.pow((self.user[i][0] - self.user[i][2]), 2) + math.pow(
                    (self.user[i][1] - self.user[i][3]), 2))  # 起点到终点
                lslact = math.sqrt(
                    math.pow((self.user[i][0] - self.x[2*i]), 2) + math.pow((self.user[i][1] - self.x[2*i+1]), 2))  # 起点到实际位置
                larrlact = math.sqrt(
                    math.pow((self.x[2*i] - self.user[i][2]), 2) + math.pow((self.x[2*i+1] - self.user[i][3]), 2))
                self.pi[i] = self.pB * (lslact - lslarr) + self.k * (larrlact) * (larrlact)
                if (self.pi[i] < 0):
                    self.pi[i] = 0
            # 匹配未成功，用户无需匹配，直接归还到目的地
            else:
                self.pi[i]=0
        return lambda x:-(sum(self.pi)-self.B)
    # 用户最大步行约束(0表示第0个用户)
    def constraint0(self):
        return -(math.pow(self.x[0]-self.user[0][2],2)+math.pow(self.x[1]-self.user[0][3],2)-math.pow(self.user[0][4],2))
    def constraint1(self):
        return -(math.pow(self.x[2]-self.user[1][2],2)+math.pow(self.x[3]-self.user[1][3],2)-math.pow(self.user[1][4],2))
    def constraint2(self):
        return -(math.pow(self.x[4]-self.user[2][2],2)+math.pow(self.x[5]-self.user[2][3],2)-math.pow(self.user[2][4],2))
    def constraint3(self):
        return -(math.pow(self.x[6]-self.user[3][2],2)+math.pow(self.x[7]-self.user[3][3],2)-math.pow(self.user[3][4],2))
    def constraint4(self):
        return -(math.pow(self.x[8]-self.user[4][2],2)+math.pow(self.x[9]-self.user[4][3],2)-math.pow(self.user[4][4],2))
    def constraint5(self):
        return -(math.pow(self.x[10]-self.user[5][2],2)+math.pow(self.x[11]-self.user[5][3],2)-math.pow(self.user[5][4],2))
    def constraint6(self):
        return -(math.pow(self.x[12]-self.user[6][2],2)+math.pow(self.x[13]-self.user[6][3],2)-math.pow(self.user[6][4],2))
    def constraint7(self):
        return -(math.pow(self.x[14]-self.user[7][2],2)+math.pow(self.x[14]-self.user[7][3],2)-math.pow(self.user[7][4],2))
    def constraint8(self):
        return -(math.pow(self.x[16]-self.user[8][2],2)+math.pow(self.x[17]-self.user[8][3],2)-math.pow(self.user[8][4],2))
    def constraint9(self):
        return -(math.pow(self.x[18]-self.user[9][2],2)+math.pow(self.x[19]-self.user[9][3],2)-math.pow(self.user[9][4],2))
    def constraint10(self):
        return -(math.pow(self.x[20]-self.user[10][2],2)+math.pow(self.x[21]-self.user[10][3],2)-math.pow(self.user[10][4],2))

    # 执行函数
    def ulpkm1(self):
        self.cons = list()
        print("!!!1",type(self.constraint()))
        self.cons.append({'type': 'ineq', 'fun':self.constraint()})
        for i in range (len(self.user)+1):
            print(i,len(self.user))
            if(i!=len(self.user)):
                print("i:",i)
                con = lambda x: -(math.pow(self.x[2*i]-self.user[i][2],2)+math.pow(self.x[2*i+1]-self.user[i][3],2)-math.pow(self.user[i][4],2))
                print("CON",type(con))
                self.cons.append({'type': 'ineq', 'fun': con})
        print("cons:",self.cons)
        # con00 = {'type': 'ineq', 'fun':self.constraint()}
        # con0 = {'type': 'ineq', 'fun': self.constraint0()}
        # con1 = {'type': 'ineq', 'fun': self.constraint1()}
        # con2 = {'type': 'ineq', 'fun': self.constraint2()}
        # con3 = {'type': 'ineq', 'fun': self.constraint3()}
        # con4 = {'type': 'ineq', 'fun': self.constraint4()}
        # con5 = {'type': 'ineq', 'fun': self.constraint5()}
        # con6 = {'type': 'ineq', 'fun': self.constraint6()}
        # con7 = {'type': 'ineq', 'fun': self.constraint7()}
        # con8 = {'type': 'ineq', 'fun': self.constraint8()}
        # con9 = {'type': 'ineq', 'fun': self.constraint9()}
        # con10 = {'type': 'ineq', 'fun': self.constraint10()}
        # con=([con00,con0,con1,con2,con3,con4,con5,con6,con7,con8,con9,con10])
        # for i in range(len(self.user)+1):
        #     self.cons[i]=con[i]
        print("type:",type(self.objective()))
        print("cons:",type(self.cons))

        solution = minimize(self.objective(), self.x0, method='SLSQP', bounds=self.bnds, constraints = self.cons)
        self.x=solution.x

        print(self.x,"\n")
        print("目标值",self.objective())

User=[[4, 4, 4, 1, 0.9568767390288544, -1, -1], [0, 0, 3, 0, 0.3508448982850835, -1, -1], [2, 1, 2, 4, 0.5086846864053723, -1, -1], [0, 3, 1, 2, 0.4112544710484871, -1, -1]]
Region=[[0, 2, 6, 1.0, 1.0], [0, 1, 10, 3.0, 1.0], [0, 1, 3, 1.0, 3.0], [0, 2, 9, 3.0, 3.0]]
test1=ulpkm(user=User,region=Region,celllength=2,B=0.9,pb=1,pk=2)
test1.ulpkm1()










