from helloword import User
from helloword import Region
from helloword import matchregion
from helloword import usernum
from helloword import cell
from helloword import celllength
from km2 import km

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time
import copy


print("用户向量2：")
print(User)

# init_Region1=copy.deepcopy(Region)
# init_Region2=copy.deepcopy(Region)
#迭代次数 T

Color=['b','c','g','k','m','r','w','y','darkviolet','brown']

class Circle:

    def __init__(self, radius: 'float', x_center: 'float', y_center: 'float'):
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center
        self.x_max = x_center + radius
        self.y_max = y_center + radius
        self.x_min = x_center - radius
        self.y_min = y_center - radius

    def randPoint(self) -> 'List[float]':
        while True:
            res_x = random.uniform(self.x_min, self.x_max)
            res_y = random.uniform(self.y_min, self.y_max)
            dis = (res_x - self.x_center)**2 + (res_y - self.y_center)**2
            if dis <= self.radius**2:
                return [res_x, res_y]

class PSO():
    def __init__(self,pN,dim,max_iter,pB,k,User,flag):   #flag=0表示greedy，flag=1表示km
        self.w=0.8   #惯性因子
        self.c1=2  #学习因子
        self.c2=2  #学习因子
        self.r1=0.6
        self.r2=0.3 #r1和r2表示0到1之间的随机数
        self.pN=pN  #粒子数量
        self.dim=dim #粒子维度，用于求点的位置（相当于用户数）
        self.max_iter=max_iter #迭代次数
        self.X=np.zeros((self.pN,self.dim),dtype=list) #np.zeros表示生成一个数值为0的数组，左边表示形式表示 生成一个二维数组，其中pN表示行数，dim表示列数
        self.V=np.zeros((self.pN,self.dim),dtype=list)  #所有粒子的位置(用坐标表示，是一个向量的形式，)和速度
        self.maxV=np.zeros(self.dim)
        #self.minV=np.zeros(self.dim)
        self.pbest=np.zeros((self.pN,self.dim),dtype=list)
        self.gbest=np.zeros(self.dim,dtype=list)   #个体经历的最佳位置和全局最佳位置
        self.User=copy.deepcopy(User)
        self.initUser=copy.deepcopy(User)
        self.greedyUser=copy.deepcopy(User)
        self.kmUser=copy.deepcopy(User)
        self.flag=flag

        self.p_fit=np.zeros(self.pN)   #每个个体的历史最佳适应值
        self.fit=1e10  #全局最佳适应值
        self.D=np.zeros(self.dim)   #用于存每个粒子距离目标地的距离

        #给定用户骑车的单价以及步行成本
        self.pB=pB   #用户骑行的单价（按米计算）
        self.k=k   #用户步行的成本单价
        self.pi=np.zeros(self.dim)
        self.B=1  #预算采用用了就减的模式？

    #X的输入表示的是一个粒子的多维数据（每一维对应一个用户数据,相当于一个大的user）（用户数多余缺车数则按缺车匹配，匹配成功给用户赋相应的值）X只表示当前粒子的位置，不能表示整个user
    def kmd(self,kmX):
        # 根据粒子当前到达的位置匹配缺车区域
        for i in range(usernum):
            self.kmUser[i][5]=-1
            self.kmUser[i][6]=-1
            self.kmUser[i][2] = kmX[i][0]
            self.kmUser[i][3] = kmX[i][1]
        init_Region1=copy.deepcopy(Region)

        psokm = km(region=init_Region1,user=self.kmUser)
        psokm.build_graph()
        kkmUser=psokm.KM()
        return kkmUser

    def greedyd(self,greedyX):
        init_Region2=copy.deepcopy(Region)
        for i in range(usernum):
            self.greedyUser[i][5]=-1
            self.greedyUser[i][6]=-1
            self.greedyUser[i][2]=greedyX[i][0]
            self.greedyUser[i][3] = greedyX[i][1]
        ggreedyUser = matchregion(init_Region2, self.greedyUser)
        return ggreedyUser


    #目标函数(输入是一个粒子的多维数据，输出是一个粒子的多维数据的距离)
    def function(self,X):
        #根据当前粒子位置来匹配停车区域
        self.pi = np.zeros(self.dim)
        self.D = np.zeros(self.dim)
        # 越界呃粒子的值，给其一个很大的返回值

        if(self.flag==0):
            self.User=self.kmd(X)
        if(self.flag==1):
            self.User=self.greedyd(X)
        print("self_User",self.User)
        # 匹配成功后再计算适应值
        #larrlact有一个限制，即用户行走的最大距离（步行限制）
        for i in range (self.dim):
            #只有匹配成功的用户才能用来计算适应值（）
            if(self.User[i][5]!=-1 and self.User[i][6]!=-1):
                lslarr = math.sqrt(math.pow((self.User[i][0] - self.initUser[i][2]), 2) + math.pow((self.User[i][1] - self.initUser[i][3]), 2))  #起点到终点
                lslact = math.sqrt(math.pow((self.User[i][0] - X[i][0]), 2) + math.pow((self.User[i][1] - X[i][1]), 2))  #起点到实际位置
                larrlact = math.sqrt(math.pow((X[i][0] - self.initUser[i][2]), 2) + math.pow((X[i][1] - self.initUser[i][3]), 2))
                self.pi[i]=self.pB*(lslact-lslarr)+self.k*(larrlact)*(larrlact)
        #计算每一维的d再求其平方和，若越界则给其给其惩罚

        print("sum_pi",sum(self.pi))

        #计算距离时（适应值）所有的点都求（根据用户来算）
        if(sum(self.pi)<=self.B):
            for i in range (self.dim):
                if (self.User[i][5] != -1 and self.User[i][6] != -1):
                    self.D[i]=math.pow((X[i][0]-self.User[i][5]),2)+math.pow((X[i][1]-self.User[i][6]),2)
                    # self.D[i]=math.sqrt(math.pow((X[i][0]-self.User[i][5]),2)+math.pow((X[i][1]-self.User[i][6]),2))
            # print("D",i,self.D)
            if(sum(self.D)>0):
                return sum(self.D)
            else:
                return 1e10
        else:
            return 1e10  #若越界则返回很大一个值(设定有问题？并没有惩罚)（超出范围的点是无效的，需要对其进行一定的限制）


    #初始化种群
    def init_Population(self):
        for i in range(self.dim):
            self.gbest[i]=[cell*celllength,cell*celllength]
        #每一个粒子包含user个用户数据
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j]=Circle(self.initUser[j][4],self.initUser[j][2],self.initUser[j][3]).randPoint()  #初始化每个粒子的位置（在用户的步行半径内随机选一个点）
                self.V[i][j]=[random.uniform(-1,1),random.uniform(-1,1)]
                # 初始化每个粒子的速度范围（一般设置在所走的距离的10%-20%）
                self.maxV[j]=self.initUser[j][4]/50
                if (self.V[i][j][0] > self.maxV[j]):
                    self.V[i][j][0] = self.maxV[j]
                if (self.V[i][j][1] > self.maxV[j]):
                    self.V[i][j][1] = self.maxV[j]

                elif (self.V[i][j][0] < -self.maxV[j]):
                    self.V[i][j][0] = -self.maxV[j]
                elif (self.V[i][j][1] < -self.maxV[j]):
                    self.V[i][j][1] = -self.maxV[j]

                if (self.X[i][j][0] < 0):
                    self.X[i][j][0] = 0
                elif (self.X[i][j][0] > celllength * cell):
                    self.X[i][j][0] = celllength * cell
                if (self.X[i][j][1] < 0):
                    self.X[i][j][1] = 0
                elif (self.X[i][j][1] > celllength * cell):
                    self.X[i][j][1] = celllength * cell
                #每初始化一次位置就计算一次距离以及价格
                # plt.xlabel("x", size=14)
                # plt.ylabel("y", size=14)
            self.pbest[i]=self.X[i]
            tmp=self.function(self.X[i])  #输入的是一个粒子的多维数据
            # print("init_temp",tmp)
            self.p_fit[i]=tmp
            if tmp < self.fit:
                self.fit=tmp
                self.gbest=self.X[i]

        for i in range(usernum):
            plt.scatter(self.initUser[i][2],self.initUser[i][3],marker='x',c=Color[i%10])  #用户的目的地
        for i in range(cell*cell):
            if(Region[i][2]>0):
                plt.scatter(Region[i][3], Region[i][4], marker='o', c=Color[i%10])  # 期望停车区域坐标
        plt.pause(1)
        plt.clf()

        # print("p_fit初始",self.p_fit)
        # print("X初始化",self.X)
        # print("init_V",self.V)

    #更新粒子位置
    def iterator(self):
        fitness=[]
        for t in range(self.max_iter):
            # print("第",t,"代")
            # print("X",self.X)
            # print("V",self.V)

            # for k in range(self.pN):   #更新gbest/pbest
            #     temp=self.function(self.X[k])  #第k个粒子
            #     print("temp",k,temp)
            #     if temp<self.p_fit[k]:   #更新个体最优
            #         self.p_fit[k]=temp
            #         self.pbest[k]=self.X[k]
            #         if self.p_fit[k] <self.fit :   #更新全局最优,具有多个元素的数组的真值是不明确的。使用a.any()或a.all()
            #             self.gbest=self.X[k]
            #             # for m in range (usernum):
            #             #     print("第k个粒子",k,"(",self.User[m][5],self.User[m][6],")")
            #             # print(self.initUser[k][1])
            #             self.fit=self.p_fit[k]

            # print(t)
            # print("fit",self.fit)
            # print("p_fit",t,self.p_fit)
            # print("gbest",self.gbest)
            for i in range (self.pN):   #更新每个粒子的位置（多维数据时更新多维的位置）
                temp = self.function(self.X[k])  # 第k个粒子
                print("temp", k, temp)
                if temp < self.p_fit[k]:  # 更新个体最优
                    self.p_fit[k] = temp
                    self.pbest[k] = self.X[k]
                    if self.p_fit[k] < self.fit:
                        self.gbest = self.X[k]
                        # for m in range (usernum):
                        #     print("第k个粒子",k,"(",self.User[m][5],self.User[m][6],")")
                        # print(self.initUser[k][1])
                        self.fit = self.p_fit[k]

                # 若超过预算限制，则temp=1e10，此时粒子不知如何处理
                if(temp==1e10):
                    a
                else:
                    for j in range (self.dim):
                        #更新速度，满足约束条件，若越界则将其设为边界值（圆形边界值如何设置？）淘汰越界的点（某一维，根据该以为的数据来更新）
                        self.V[i][j][0]=self.w*self.V[i][j][0]+self.c1*self.r1*(self.pbest[i][j][0]-self.X[i][j][0])+self.c2*self.r2*(self.gbest[j][0]-self.X[i][j][0])
                        self.V[i][j][1] = self.w * self.V[i][j][1] + self.c1 * self.r1 * (self.pbest[i][j][1] - self.X[i][j][1]) + self.c2 * self.r2 * (self.gbest[j][1] - self.X[i][j][1])
                        if(self.V[i][j][0]>self.maxV[j]):
                            self.V[i][j][0]=self.maxV[j]
                        if (self.V[i][j][1] > self.maxV[j]):
                            self.V[i][j][1] = self.maxV[j]
                        elif(self.V[i][j][0]<-self.maxV[j]):
                            self.V[i][j][0] = -self.maxV[j]
                        elif (self.V[i][j][1] < -self.maxV[j]):
                            self.V[i][j][1] = -self.maxV[j]
                            # self.V[i][j][1] = -self.maxV[j]

                        self.X[i][j][0]=self.X[i][j][0]+self.V[i][j][0]
                        self.X[i][j][1] = self.X[i][j][1] + self.V[i][j][1]
                        #更新位置时还需要增加限制条件，超出限制条件时要给一定的惩罚
                        if(math.sqrt(math.pow((self.X[i][j][0] - self.initUser[j][2]), 2) + math.pow((self.X[i][j][1] - self.initUser[j][3]), 2))>self.initUser[j][4]):   #用户出界，则选择这条路线上最远的一个点
                            a=self.X[i][j][0]
                            b=self.X[i][j][1]
                            self.X[i][j][0]=self.initUser[j][2]-self.initUser[j][4]*(self.initUser[j][2]-a)/math.sqrt(math.pow((a - self.initUser[j][2]), 2) + math.pow((b - self.initUser[j][3]), 2))

                            self.X[i][j][1] = self.initUser[j][3] - self.initUser[j][4] * (self.initUser[j][3] - b) / math.sqrt(math.pow((a - self.initUser[j][2]), 2) + math.pow((b - self.initUser[j][3]), 2))

                        if(self.X[i][j][0]<0):
                            # self.V[i][j][0]=0
                            self.X[i][j][0]=0
                        elif(self.X[i][j][0]>celllength*cell):
                            # self.V[i][j][0] = 0
                            self.X[i][j][0]=celllength*cell
                        if (self.X[i][j][1] < 0):
                            # self.V[i][j][1] = 0
                            self.X[i][j][1] = 0
                        elif (self.X[i][j][1] > celllength * cell):
                            # self.V[i][j][1] = 0
                            self.X[i][j][1] = celllength * cell


                        plt.scatter(self.X[i][j][0], self.X[i][j][1], marker='.',c=Color[i%10])
            for i in range(usernum):
                plt.scatter(self.initUser[i][2], self.initUser[i][3], marker='x', c=Color[i%10])  # 用户的目的地
            for i in range(cell * cell):
                if (Region[i][2] > 0):
                    plt.scatter(Region[i][3], Region[i][4], marker='o', c=Color[i%10])  # 期望停车区域坐标
            plt.pause(0.2)
            plt.clf()
            plt.show()

            # print("gbest2")
            # print("X",i,self.X)
            fitness.append(self.fit)
            # print(self.gbest)
            print(self.fit) #输出最优值
        # print("gbest",self.gbest)

        return fitness
#在代码中如何判断用户到达的点


T=100
#km
plt.ion()
my_pso1=PSO(pN=20,dim=usernum,max_iter=T,pB=1,k=1,User=User,flag=0)
my_pso1.init_Population()
fitness2=my_pso1.iterator()
plt.ioff()
print("fitness2",fitness2)

#greedy
plt.ion()
my_pso2=PSO(pN=20,dim=usernum,max_iter=T,pB=1,k=1,User=User,flag=1)
my_pso2.init_Population()
fitness1=my_pso2.iterator()
plt.ioff()
print("fitness1",fitness1)
# 用户有多的，即有一些用户不需要移动，这种情况还未处理

# 画图
plt.figure(1)
plt.xlabel("iterators",size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, T)])  #迭代次数
plt.plot(0,3,color='g')
fitness1 = np.array(fitness1)
plt.plot(t, fitness1,label="greedy", color='b', linewidth=1)
fitness2 = np.array(fitness2)
plt.plot(t, fitness2,label="km", color='r', linewidth=1)
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)  #图例显示
plt.show()

