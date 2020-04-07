"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import copy

from km2 import km
from helloword import matchregion
import matplotlib.pyplot as plt
import time
# import gym
from torch.autograd import Variable

random.seed(1)


# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000

# 参数设置
T=5  #时间时段
walknum=10 #步行的距离分为10个步行长度来采取动作
RB=20  #预算约束
# 横向网格数
cell=2
# 纵向网格数
# ycell=2
# 单个网格长度
celllength=3
regionnum=cell*cell #区域个数
pricek=2
usernum=10 #用户数为10

# 定义区域(（用户数，区域内车辆数,区域内缺车的数量,中心点横坐标，中心点纵坐标）)
init_region = list()
for i in range(regionnum):
    # print(i)
    regionn =[0,random.randint(1,2),random.randint(0,10),(i%cell)*celllength+celllength/2,(int(i/cell))*celllength+celllength/2]
    # print(r)
    init_region.append(regionn)
    # print(region)

# 用户需求,# 定义用户数组（起点横坐标，起点纵坐标，终点横坐标，终点纵坐标，最大步行距离,期望停车区域横坐标，期望停车区域纵坐标）
def init_user_demand():
    userdemand=[[[0]for i in range (usernum)] for t in range (T)]
    for t in range (T):
        for i in range (usernum):
            userdemand[t][i]=[random.randint(0,celllength*cell),random.randint(0,celllength*cell),random.randint(0,celllength*cell),random.randint(0,celllength*cell),random.uniform(0,celllength/2),-1,-1]
    return userdemand
# 初始化状态
def init_state():
    # 状态最后一项是剩余预算，后面是每个区域的（车数）,区域从1开始计数
    s=[[0] for i in range (regionnum+1)]
    for i in range (regionnum+1):
        s[i]=random.randint(0,10) #第i个区域的车的供应量
    s[regionnum]=RB
            # s[2*i]=random.randint(0,10)  #第i个区域的用户需求数
    return s
# 输入状态值，输出所有的动作值（根据强化学习的方法选取动作）
# 更新神经网络参数

# 神经网络输出的action的个数（对于每一个用户而言，有多少个用户输出多少个action） 输出的action包含所有的用户的动作，是一个集合，（动作就是用户到达的区域）
# 从剩余预算中选取一个预算作为当前时段的钱
N_ACTIONS = 1000
# 接收的observation维度
N_STATES = regionnum+1
class Net(nn.Module):
    # 定义卷积层
    def __init__(self, ):
        # 给父类nn.module初始化
        super(Net, self).__init__()
        # nn.Linear用于设置网络中的全连接层（全连接层的输入输出都是二维张量）nn.linear(in_features,out_features)in_features指size of input sample，out_features指size of output sample
        # 定义网络结构y=w*x+b weight[out_features,in_features],w,b是神经网络的参数，我们的目的就是不断更新神经网络的参数来优化目标函数
        # 我们只需将输入的特征数和输出的特征数传递给torch.nn.Linear类，就会自动生成对应维度的权重参数和偏置
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization,权重初始化，利用正态进行初始化
        # 第二层神经网络，用于输出action
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    # 执行数据的流动
    def forward(self, x):
        # 输入的x先经过定义的第一层全连接层
        x = self.fc1(x)
        # 官方提供的激活函数
        x = F.relu(x)
        # 经过第二层后输出action
        actions_value = self.out(x)
        return actions_value
class DQN(object):
    def __init__(self):
        # 声明网络的实例：eval_net和target_net
        self.eval_net, self.target_net = Net(), Net()
        # 计步器，记录学习了多少步，epsilon会根据counter不断地提高
        self.learn_step_counter = 0                                     # for target updating

        self.memory_counter = 0                                         # for storing memory

        # 初始化记忆库，第一个参数就是记忆库的大小，第二个参数是一条数据中（s,a,r,s）中的个数总和
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory

        # 定义神经网络的优化器
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        # 定义损失函数
        self.loss_func = nn.MSELoss()

    # 把observation放入神经网络，输出每一个action的Q值，选取最大的
    def choose_action(self, observation):
        #torch.unsqueeze(input,dim) 这个函数主要是对数据维度进行扩充,需要通过dim指定位置，给指定位置加上维数为1的维度
        # floattensor是pytorch的基本变量类型,torch.FloatTensor(x)是将x转换成该类型
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)
        # 这里只输入一个sample
        if np.random.uniform() < EPSILON:   # greedy
            # 神经网络输出所有的action的值
            actions_value = self.eval_net.forward(observation)
            # 选取一个值最大的action
            action = torch.max(actions_value, 1)[1].data.numpy()
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index？
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  #？
        return action

    # 定义样本池
    def store_transition(self, s, a, r, s_):
        #  np.hastack 水平合并 数组
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        # 样本池按列存放样本，每个样本为一个行向量
        #循环存储每个行动样本，index用于循环覆盖的起始行数
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update (看有没有到换参数的时候)如需更新target参数，则更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 针对做过的动作b_a,来选q_eval的值，q_eval w.r.t the action in experience
        # q_eval是Q_估计神经网络输出的值
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # q_next是Q_target神经网络输出的值
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # 计算q_target
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)

        # 计算神经网络的损失
        loss = self.loss_func(q_eval, q_target)
        # 清空优化器里的梯度信息
        self.optimizer.zero_grad()
        # 误差反向传播
        loss.backward()
        # 更新参数
        self.optimizer.step()
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
    def __init__(self,pN,dim,max_iter,pB,k,User,region,flag,B):   #flag=0表示greedy，flag=1表示km
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
        self.B=B  #预算采用用了就减的模式？
        self.region=region

    #X的输入表示的是一个粒子的多维数据（每一维对应一个用户数据,相当于一个大的user）（用户数多余缺车数则按缺车匹配，匹配成功给用户赋相应的值）X只表示当前粒子的位置，不能表示整个user
    def kmd(self,kmX):
        # 根据粒子当前到达的位置匹配缺车区域
        for i in range(usernum):
            self.kmUser[i][5]=-1
            self.kmUser[i][6]=-1
            self.kmUser[i][2] = kmX[i][0]
            self.kmUser[i][3] = kmX[i][1]
        init_Region1=copy.deepcopy(self.region)

        psokm = km(region=init_Region1,user=self.kmUser)
        psokm.build_graph()
        kkmUser=psokm.KM()
        return kkmUser

    def greedyd(self,greedyX):
        init_Region2=copy.deepcopy(self.region)
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
        # print("self_User",self.User)
        # 匹配成功后再计算适应值
        #larrlact有一个限制，即用户行走的最大距离（步行限制）
        for i in range (self.dim):
            #只有匹配成功的用户才能用来计算适应值（）
            if(self.User[i][5]!=-1 and self.User[i][6]!=-1):
                lslarr = math.sqrt(math.pow((self.User[i][0] - self.initUser[i][2]), 2) + math.pow((self.User[i][1] - self.initUser[i][3]), 2))  #起点到终点
                lslact = math.sqrt(math.pow((self.User[i][0] - X[i][0]), 2) + math.pow((self.User[i][1] - X[i][1]), 2))  #起点到实际位置
                larrlact = math.sqrt(math.pow((X[i][0] - self.initUser[i][2]), 2) + math.pow((X[i][1] - self.initUser[i][3]), 2))
                self.pi[i]=self.pB*(lslact-lslarr)+self.k*(larrlact)*(larrlact)
                if (self.pi[i] < 0):
                    self.pi[i] = 0
        #计算每一维的d再求其平方和，若越界则给其给其惩罚

        # print("sum_pi",sum(self.pi))

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
                    self.X[i][j][0] = celllength *cell
                if (self.X[i][j][1] < 0):
                    self.X[i][j][1] = 0
                elif (self.X[i][j][1] > celllength * cell):
                    self.X[i][j][1] = celllength * cell

            self.pbest[i]=self.X[i]
            tmp=self.function(self.X[i])  #输入的是一个粒子的多维数据
            # print("init_temp",tmp)
            self.p_fit[i]=tmp
            if tmp < self.fit:
                self.fit=tmp
                self.gbest=self.X[i]


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


            # print(t)
            # print("fit",self.fit)
            # print("p_fit",t,self.p_fit)
            # print("gbest",self.gbest)
            for i in range (self.pN):   #更新每个粒子的位置（多维数据时更新多维的位置）
                temp = self.function(self.X[i])  # 第k个粒子
                # print("temp", i, temp)
                if temp < self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if self.p_fit[i] < self.fit:
                        self.gbest = self.X[i]
                        # for m in range (usernum):
                        #     print("第k个粒子",k,"(",self.User[m][5],self.User[m][6],")")
                        # print(self.initUser[k][1])
                        self.fit = self.p_fit[i]

                # 若超过预算限制，则temp=1e10，此时粒子不知如何处理

                if (temp == 1e10):
                    for j in range(self.dim):
                        # 更新速度，满足约束条件，若越界则将其设为边界值（圆形边界值如何设置？）淘汰越界的点（某一维，根据该以为的数据来更新）
                        if (self.X[i][j][0] < self.initUser[j][2]):
                            self.V[i][j][0] = abs(self.V[i][j][0])
                        if (self.X[i][j][1] < self.initUser[j][3]):
                            self.V[i][j][1] = abs(self.V[i][j][1])
                        if (self.X[i][j][0] > self.initUser[j][2]):
                            self.V[i][j][0] = -abs(self.V[i][j][0])
                        if (self.X[i][j][1] > self.initUser[j][3]):
                            self.V[i][j][1] = -abs(self.V[i][j][1])
                        if (self.V[i][j][0] > self.maxV[j]):
                            self.V[i][j][0] = self.maxV[j]
                        if (self.V[i][j][1] > self.maxV[j]):
                            self.V[i][j][1] = self.maxV[j]
                        elif (self.V[i][j][0] < -self.maxV[j]):
                            self.V[i][j][0] = -self.maxV[j]
                        elif (self.V[i][j][1] < -self.maxV[j]):
                            self.V[i][j][1] = -self.maxV[j]
                            # self.V[i][j][1] = -self.maxV[j]

                        self.X[i][j][0] = self.X[i][j][0] + self.V[i][j][0]
                        self.X[i][j][1] = self.X[i][j][1] + self.V[i][j][1]
                        # 更新位置时还需要增加限制条件，超出限制条件时要给一定的惩罚
                        if (math.sqrt(math.pow((self.X[i][j][0] - self.initUser[j][2]), 2) + math.pow(
                                (self.X[i][j][1] - self.initUser[j][3]), 2)) > self.initUser[j][
                            4]):  # 用户出界，则选择这条路线上最远的一个点
                            a = self.X[i][j][0]
                            b = self.X[i][j][1]
                            self.X[i][j][0] = self.initUser[j][2] - self.initUser[j][4] * (
                                    self.initUser[j][2] - a) / math.sqrt(
                                math.pow((a - self.initUser[j][2]), 2) + math.pow((b - self.initUser[j][3]), 2))

                            self.X[i][j][1] = self.initUser[j][3] - self.initUser[j][4] * (
                                    self.initUser[j][3] - b) / math.sqrt(
                                math.pow((a - self.initUser[j][2]), 2) + math.pow((b - self.initUser[j][3]), 2))

                        if (self.X[i][j][0] < 0):
                            # self.V[i][j][0]=0
                            self.X[i][j][0] = 0
                        elif (self.X[i][j][0] > celllength * cell):
                            # self.V[i][j][0] = 0
                            self.X[i][j][0] = celllength * cell
                        if (self.X[i][j][1] < 0):
                            # self.V[i][j][1] = 0
                            self.X[i][j][1] = 0
                        elif (self.X[i][j][1] > celllength * cell):
                            # self.V[i][j][1] = 0
                            self.X[i][j][1] = celllength * cell

                else:
                    for j in range(self.dim):
                        # 更新速度，满足约束条件，若越界则将其设为边界值（圆形边界值如何设置？）淘汰越界的点（某一维，根据该以为的数据来更新）
                        self.V[i][j][0] = self.w * self.V[i][j][0] + self.c1 * self.r1 * (
                                    self.pbest[i][j][0] - self.X[i][j][0]) + self.c2 * self.r2 * (
                                                      self.gbest[j][0] - self.X[i][j][0])
                        self.V[i][j][1] = self.w * self.V[i][j][1] + self.c1 * self.r1 * (
                                    self.pbest[i][j][1] - self.X[i][j][1]) + self.c2 * self.r2 * (
                                                      self.gbest[j][1] - self.X[i][j][1])
                        if (self.V[i][j][0] > self.maxV[j]):
                            self.V[i][j][0] = self.maxV[j]
                        if (self.V[i][j][1] > self.maxV[j]):
                            self.V[i][j][1] = self.maxV[j]
                        elif (self.V[i][j][0] < -self.maxV[j]):
                            self.V[i][j][0] = -self.maxV[j]
                        elif (self.V[i][j][1] < -self.maxV[j]):
                            self.V[i][j][1] = -self.maxV[j]
                            # self.V[i][j][1] = -self.maxV[j]

                        self.X[i][j][0] = self.X[i][j][0] + self.V[i][j][0]
                        self.X[i][j][1] = self.X[i][j][1] + self.V[i][j][1]
                        # 更新位置时还需要增加限制条件，超出限制条件时要给一定的惩罚
                        if (math.sqrt(math.pow((self.X[i][j][0] - self.initUser[j][2]), 2) + math.pow(
                                (self.X[i][j][1] - self.initUser[j][3]), 2)) > self.initUser[j][
                            4]):  # 用户出界，则选择这条路线上最远的一个点
                            a = self.X[i][j][0]
                            b = self.X[i][j][1]
                            self.X[i][j][0] = self.initUser[j][2] - self.initUser[j][4] * (
                                        self.initUser[j][2] - a) / math.sqrt(
                                math.pow((a - self.initUser[j][2]), 2) + math.pow((b - self.initUser[j][3]), 2))

                            self.X[i][j][1] = self.initUser[j][3] - self.initUser[j][4] * (
                                        self.initUser[j][3] - b) / math.sqrt(
                                math.pow((a - self.initUser[j][2]), 2) + math.pow((b - self.initUser[j][3]), 2))

                        if (self.X[i][j][0] < 0):
                            # self.V[i][j][0]=0
                            self.X[i][j][0] = 0
                        elif (self.X[i][j][0] > celllength * cell):
                            # self.V[i][j][0] = 0
                            self.X[i][j][0] = celllength * cell
                        if (self.X[i][j][1] < 0):
                            # self.V[i][j][1] = 0
                            self.X[i][j][1] = 0
                        elif (self.X[i][j][1] > celllength * cell):
                            # self.V[i][j][1] = 0
                            self.X[i][j][1] = celllength * cell

            # print("gbest2")
            # print("X",i,self.X)
            fitness.append(self.fit)
            # print(self.gbest)
            # print(self.fit) #输出最优值
        # print("gbest",self.gbest)

        return self.gbest,self.fit
def run_this():
    dqn = DQN()
    print('\nCollecting experience...')
    # run this
    sumreward=[]
    for i_episode in range(100):
        # 初始化环境
        sum_r=0
        # done=False
        s = init_state()
        s_=copy.deepcopy(s)
        ep_r = 0
        init_user = init_user_demand()
        user=copy.deepcopy(init_user)
        region=copy.deepcopy(init_region)
        for t in range (T):
            # dqn.choose_action()一下只能输出一个动作，动作的维度是用户数，输出的a是每一个用户采取的哪一个动作
            # 首先根据当前observation选取一个动作
            # 预算用完或不够，不能进行重平衡，则d=用户目的地到缺车区域(先匹配，再计算d即kmd)(有问题，应该是一个状态更新后来根据预算是否为0来求回报)
            if (s_[regionnum] <= 0):
                r=0
                # 选取动作
                action = dqn.choose_action(s)
                RB_t = (action / N_ACTIONS) * s[regionnum]
                # 计算回报（预算用完时，用户直接将车还到目的地区域，然后匹配其期望还车区域，计算d，我们的d是用户的还车区域到期望还车区域的距离d）
                init_User = copy.deepcopy(user[t])
                pso = km(region=region, user=init_User)
                pso.build_graph()
                uuser = pso.KM()
                for i in range(usernum):
                    if (uuser[i][5] != -1 and uuser[i][6] != -1):
                        r += -math.pow(uuser[i][5] - uuser[i][2], 2) + math.pow(uuser[i][6] - uuser[i][3], 2)
                #     更新状态(取车区域-1，还车区域+1)
                for i in range(usernum):
                    if (user[t][i][0] == cell * celllength):
                        tempa = int(user[t][i][1] / celllength) * cell + int(user[t][i][0] / celllength) - 1
                    elif (user[t][i][1] == cell * celllength):
                        tempa = int(user[t][i][1] / celllength) * cell + int(user[t][i][0] / celllength) - cell
                    elif (user[t][i][0] == cell * celllength & user[t][i][1] == cell * celllength):
                        tempa = int(cell * cell - 1)
                    else:
                        tempa = int(user[t][i][1] / celllength) * cell + int(user[t][i][0] / celllength)
                    # print(a)
                    if (tempa <= cell * cell):
                        region[tempa][1] -= 1
                        s_[tempa] -= 1

                for i in range (usernum):
                    if (uuser[i][2] == cell * celllength):
                        tempb = int(uuser[i][3] / celllength) * cell + int(uuser[i][2] / celllength) - 1
                    elif (uuser[i][3] == cell * celllength):
                        tempb=int(uuser[i][3] / celllength) * cell + int(uuser[i][2] / celllength) - cell
                    elif (uuser[i][3] == cell * celllength and uuser[i][1] == cell * celllength):
                        tempb = cell * cell - 1
                    else:
                        tempb = int(tempuser[i][3] / celllength) * cell + int(tempuser[i][2] / celllength)
                # print(a)
                    if (tempb <= cell * cell):
                        s_[tempb]+=1
                s_[regionnum]-=RB_t
                print(RB_t)

            else:
                if(t!=T-1):
                    # 先判断当前时段用户所在的区域以及此时区域内车的数量
                    for i in range(regionnum):
                        region[i][1]=s[i]
                        region[i][0]=0

                    # 判断每个用户在哪骑车，并更新一下状态（区域内车辆数-1）
                    for i in range(usernum):
                        if (user[t][i][0] == cell * celllength):
                            tempa = int(user[t][i][1] / celllength) * cell + int(user[t][i][0] / celllength) - 1
                        elif (user[t][i][1] == cell * celllength):
                            tempa=int(user[t][i][1] / celllength) * cell + int(user[t][i][0] / celllength) - xcell
                        elif (user[t][i][0] == cell * celllength & user[t][i][1] == cell * celllength):
                            tempa =int(cell*cell - 1)
                        else:
                            tempa = int(user[t][i][1] / celllength) * cell + int(user[t][i][0] / celllength)
                        # print(a)
                        if (tempa <= cell*cell):
                            region[tempa][1]-=1
                            s_[tempa]-=1

                    #判断下一时刻到来的用户数所在区域，并获得该区域在下一时刻的缺车数
                    for i in range(usernum):
                        if (user[t+1][i][0] == cell * celllength):
                            tempaa = int(user[t+1][i][1] / celllength) * cell + int(user[t+1][i][0] / celllength) - 1
                        elif (user[t+1][i][1] == cell * celllength):
                            tempaa=int(user[t+1][i][1] / celllength) * cell + int(user[t+1][i][0] / celllength) - cell
                        elif (user[t+1][i][0] == cell * celllength & user[t+1][i][1] == cell * celllength):
                            tempaa =int(cell*cell - 1)
                        else:
                            tempaa = int(user[t+1][i][1] / celllength) * cell + int(user[t+1][i][0] / celllength)
                        # print(a)
                        if (tempaa <= cell*cell):
                            region[tempaa][0]+=1
                    for i in range(regionnum):
                        region[i][2] = region[i][0] - region[i][1]


                action = dqn.choose_action(s)
                RB_t=(action/N_ACTIONS)*s[regionnum]
                my_pso1 = PSO(pN=50, dim=usernum, max_iter=200, pB=1, k=1, User=user[t], flag=0,B=RB_t,region=region)
                my_pso1.init_Population()
                # tempuser为各个用户的终点，tempfit为最小d
                tempuser,tempfit = my_pso1.iterator()

                # 判断用户还车的区域来更新状态
                for i in range (usernum):
                    if (tempuser[i][0] == cell * celllength):
                        tempb = int(tempuser[i][1] / celllength) * cell + int(tempuser[i][0] / celllength) - 1
                    elif (tempuser[i][1] == cell * celllength):
                        tempb=int(tempuser[i][1] / celllength) * cell + int(tempuser[i][0] / celllength) - cell
                    elif (tempuser[i][0] == cell * celllength and tempuser[i][1] == cell * celllength):
                        tempb = cell * cell - 1
                    else:
                        tempb = int(tempuser[i][1] / celllength) * cell + int(tempuser[i][0] / celllength)
                # print(a)
                    if (tempb <= cell * cell):
                        s_[tempb]+=1
                s_[regionnum]-=RB_t
                print(RB_t)


                    # done = True  # 若RB<0，预算算完，则之后的状态的d直接根据用户的目的地及需求点算
                # reward是不是计算错了怎么可能这么大
                r = -tempfit
            # 更新reward，即粒子群算法计算出来的d(d是最小化，reward是最大化)
            sum_r+=r

            # take action 与环境交互，施加改动作在环境中，得到下一个observation_,以及reward，done表示是否终结
            # 根据当前动作和状态来得到下一状态以及reword
            ep_r += r
            dqn.store_transition(s,action,r,s_)

            # s首先需要存储记忆，记忆库中有一些东西之后才能学习（前200步都是在存储记忆,大于200之后每5步学习一次）
            if dqn.memory_counter > MEMORY_CAPACITY and dqn.memory_counter%5==0:
                dqn.learn()
                # if done:
                #     print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))

            s = s_
            print("第t时的奖励",t,r)
        print(sum_r)
        sumreward.append(sum_r)

        print(i_episode)
    plt.figure(1)
    plt.xlabel("iterators", size=14)
    plt.ylabel("reward", size=14)
    t = np.array([t for t in range(0, 100)])  # 迭代次数
    plt.plot(0, 3, color='g')
    fitness1 = np.array(sumreward)
    plt.plot(t, fitness1, label="greedy", color='b', linewidth=1)
    plt.show()
    # plt.plot(i_episode, sumreward, label="reward", color='b', linewidth=1)


run_this()