from pulp import *
import numpy as np
import copy
from km2 import km
import math

xcell=2
ycell=2
celllength=2


class DP:
    def __init__(self,usernum,user,region,B,pw,pb):
        self.usernum=usernum
        # X表示横坐标，Y表示纵坐标
        self.XY=np.zeros(self.usernum*2)
        # self.X=np.zeros(self.usernum,)
        # self.Y=np.zeros(self.usernum)
        self.User=user
        self.Region=region
        self.kmUser=copy.deepcopy(user)
        self.D=np.zeros(self.usernum)   #用于存每个粒子距离目标地的距离
        self.B=B #预算约束
        self.pb=pb #骑车单价
        self.pw=pw #步行单价
        self.pi=np.zeros(self.usernum)

    def kmd(self,kmX):
        # 根据粒子当前到达的位置匹配缺车区域
        for i in range(self.usernum):
            self.kmUser[i][5]=-1
            self.kmUser[i][6]=-1
            self.kmUser[i][2] = kmX[2*i]
            self.kmUser[i][3] = kmX[2*i+1]
        init_Region1=copy.deepcopy(self.Region)

        psokm = km(region=init_Region1,user=self.kmUser,cellnum=xcell*celllength)
        psokm.build_graph()
        kkmUser=psokm.KM()
        return kkmUser

    def dp(self):
        # 定义线性规划问题
        prob = LpProblem("minimized", LpMinimize)
        # 定义决策变量（每个用户实际停车的位置，一共n*2个决策变量）

        x=LpVariable.dict("x",self.XY,0,xcell*celllength,LpContinuous)
        # for i in range (self.usernum):
        #     self.X[i]=LpVariable("X",0,xcell*celllength,LpContinuous)
        #     self.Y[i] = LpVariable("Y", 0, ycell*celllength,LpContinuous)
        # 先进行km匹配，将用户与缺车区域匹配
        self.kmUser=self.kmd(self.XY)

        # 目标函数（此时的解（x,y)到匹配的缺车区域之间的距离）
        # for i in range (self.usernum):
        #     b=2*i+1
        #     if(self.kmUser[i][5]!=-1 and self.kmUser[i][6]!=-1):
        #         self.D[i]=math.pow(x[2*i]-self.kmUser[i][5],2)+math.pow(x[b]-self.kmUser[i][6],2)
        prob+=lpSum([x[2*i]*x[2*i] for i in range (len(self.kmUser))])
        # prob+=lpSum([(x[2*i]-self.kmUser[i][5])*(x[2*i]-self.kmUser[i][5])+(x[2*i+1]-self.kmUser[i][6])*(x[2*i+1]-self.kmUser[i][6]) for i in range (len(self.kmUser)) if(self.kmUser[i][5]!=-1 and self.kmUser[i][6]!=-1)]),"abc"
        # 添加约束条件
        # 约束条件3：最大步行距离

        for i in range(self.usernum):
            prob += x[2*i] >= self.User[i][2] - self.User[i][4]
            prob += x[2 * i] <= self.User[i][2] + self.User[i][4]
            prob += x[2 * i + 1] >= self.User[i][3] - self.User[i][4]
            prob += x[2 * i + 1] <= self.User[i][3] + self.User[i][4]

        for i in range (self.usernum):
            # 约束条件1：用户步行的最大距离
            prob+=math.pow(x[2*i]-self.User[i][2],2)+math.pow(x[2*i+1]-self.User[i][3],2)<=math.pow(self.kmUser[i][4],2)
            if(self.kmUser[i][5]!=-1 and self.kmUser[i][6]!=-1):
                lslarr = math.sqrt(math.pow((self.kmUser[i][0] - self.User[i][2]), 2) + math.pow(
                    (self.kmUser[i][1] - self.User[i][3]), 2))  # 起点到终点
                lslact = math.sqrt(
                    math.pow((self.kmUser[i][0] - x[2*i]), 2) + math.pow((self.kmUser[i][1] - x[2*i+1]), 2))  # 起点到实际位置
                larrlact = math.sqrt(
                    math.pow((x[2*i] - self.User[i][2]), 2) + math.pow((x[2*i+1] - self.User[i][3]), 2))
                self.pi[i] = self.pB * (lslact - lslarr) + self.k * (larrlact) * (larrlact)
                if (self.pi[i] < 0):
                    self.pi[i] = 0

#         约束条件2：用户步行所花成本和<=预算约束
        prob+=sum(self.pi)<=self.B



        # 写入LP文件
        prob.writeLP("minimized.lp")
        # 模型求解
        prob.solve()

        print("\n","status:",LpStatus[prob.status],"\n")
        for v in prob.variables():
            print("\t",v.name,"=",v.varValue,"\n")
        print("minid",value(prob.objective))

User=[[4, 4, 4, 1, 0.9568767390288544, -1, -1]]
Region=[[0, 2, 6, 1.0, 1.0], [0, 1, 10, 3.0, 1.0], [0, 1, 3, 1.0, 3.0], [0, 2, 9, 3.0, 3.0]]
dp1=DP(usernum=1,user=User,region=Region,B=1,pw=2,pb=1)

dp1.dp()






