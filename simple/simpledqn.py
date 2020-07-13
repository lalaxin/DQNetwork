"""
考虑用户圆形区域取车，即周围区域，临近区域的中心点在步行距离范围内时，可取车
两个用户，两个缺车，不考虑用户最大步行距离，只在当前区域还车
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import copy

import xlrd

from simple.simplekm2 import km
from ulpregion import ulpkm

import matplotlib.pyplot as plt
import time
from torch.autograd import Variable

from users3_1 import getuser

random.seed(1)
# Hyper Parameters
BATCH_SIZE = 128


LR = 0.004             # learning rate
EPSILON = 0               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY =11000

# 参数设置
T=12 #时间时段
RB=50000000#预算约束
# 横向网格数
cell=4
# 单个网格长度
celllength=300
regionnum=cell*cell #区域个数
EPISODE=4000 #迭代次数
# 记录损失
loss=[]

usernum=20 #用户数为10

# # 随机数据
# （用户数，区域内车辆数,区域内缺车的数量,中心点横坐标，中心点纵坐标），初始化区域时只需初始化当前区域内的车辆数即可，然后根据用户到来信息求得用户数和缺车数
init_region = list()
number=0
region0=[0,1,1,0,0,0,0,1,2,0,2,2,1,0,2,1]
for i in range(regionnum):
    regionn =[0,0,0,(i%cell)*celllength+celllength/2,(int(i/cell))*celllength+celllength/2]
    init_region.append(regionn)

# 用户需求,T个时间段的用户需求# 定义用户数组（起点横坐标，起点纵坐标，终点横坐标，终点纵坐标，最大步行距离,期望停车区域横坐标，期望停车区域纵坐标）
def init_user_demand():
    userdemand=[[[0]for i in range (usernum)] for t in range (T)]
    for t in range (T):
        for i in range (usernum):
            # 用户的位置全部用区域表示
            userdemand[t][i]=[random.randint(0, cell*celllength),random.randint(0, cell*celllength),(random.randint(0, cell - 1) + 1 / 2) * celllength,(random.randint(0, cell - 1) + 1 / 2) * celllength,cell*celllength*1.4,-1,-1]
            # userdemand[t][i]=[(random.randint(0, cell - 1) + 1 / 2) * celllength,(random.randint(0, cell - 1) + 1 / 2) * celllength,(random.randint(0, cell - 1) + 1 / 2) * celllength,(random.randint(0, cell - 1) + 1 / 2) * celllength,cell*celllength*1.4,-1,-1]
            # userdemand[t][i]=[random.randint(0,celllength*cell),random.randint(0,celllength*cell),random.randint(0,celllength*cell),random.randint(0,celllength*cell),cell*celllength*1.4,-1,-1]
    print("userdemand",userdemand)
    return userdemand
init_user=init_user_demand()

def init_user_region():
    number=0
    for i in range(len(init_user[0])):
        if (init_user[0][i][0] == cell * celllength and init_user[0][i][1] == cell * celllength):
            tempa = int(cell * cell - 1)
        elif (init_user[0][i][0] == cell * celllength):
            tempa = int(init_user[0][i][1] / celllength) * cell + int(init_user[0][i][0] / celllength) - 1
        elif (init_user[0][i][1] == cell * celllength):
            tempa = int(init_user[0][i][1] / celllength) * cell + int(init_user[0][i][0] / celllength) - cell
        else:
            tempa = int(init_user[0][i][1] / celllength) * cell + int(init_user[0][i][0] / celllength)
        if (tempa < cell * cell):
            init_region[tempa][1]+=1
    for i in range(regionnum):
        number += init_region[i][1]
    print("number",number)
init_user_region()
print("initregion",init_region)

# 初始化状态，直接根据region来初始化状态

# 真实数据
# init_region = list()
# def init_region2():
#     userregion = []
#     excel = xlrd.open_workbook("../userregion.xlsx")
#     sheet = excel.sheet_by_name("sheet1")
#     userregion=sheet.row_values(0)
#     print("userregion",userregion)
#     for i in range(regionnum):
#         regionn = [0, int(userregion[i]), 0, (i % cell) * celllength + celllength / 2,(int(i / cell)) * celllength + celllength / 2]
#         # regionn =[0,int(userregion[i]*99/539)+1,0,(i%cell)*celllength+celllength/2,(int(i/cell))*celllength+celllength/2]
#         init_region.append(regionn)
#         # print(region)
# init_region2()
#
#
# number=0
# # region0=[0,1,1,0,0,0,0,1,2,0,2,2,1,0,2,1]
# for i in range(regionnum):
#       number += init_region[i][1]
# print("initregion",init_region)
# print("number",number)
#
# # 真实需求
# def init_user_demand():
#     # 将excel的数据存于数组中
#     userdemand=getuser().getusers()
#     return userdemand
# init_user = init_user_demand()

def init_state():
    #状态应包含这一时间段每个区域的（第二个regionnum）用户数，(第1个regionnum)车的供应数以及下一阶段该区域的缺车数（第三个regionnum）
    s=[0 for i in range (3*regionnum+1)]
    for i in range (regionnum):
        s[i]=init_region[i][1] #第i个区域的车的供应量
    s[3*regionnum]=RB
    return s
# 输入状态值，输出所有的动作值（根据强化学习的方法选取动作）
# 更新神经网络参数

# 神经网络输出的action的个数（对于每一个用户而言，有多少个用户输出多少个action） 输出的action包含所有的用户的动作，是一个集合，（动作就是用户到达的区域）
# 从剩余预算中选取一个预算作为当前时段的钱
sum_loss=[]
N_ACTIONS = 100
# 接收的observation维度(区域当前供应，上个阶段每个区域的用户数，上个阶段每个区域到达的用户数，预算约束)
# 采取的动作是分配的预算，即预算应该只与当前区域的状态，当前用户到来的状态以及下个时间段的缺车有关
N_STATES = 3*regionnum+1
class Net(nn.Module):
    # 定义卷积层
    def __init__(self, ):
        # 给父类nn.module初始化
        super(Net, self).__init__()
        # nn.Linear用于设置网络中的全连接层（全连接层的输入输出都是二维张量）nn.linear(in_features,out_features)in_features指size of input sample，out_features指size of output sample
        # 定义网络结构y=w*x+b weight[out_features,in_features],w,b是神经网络的参数，我们的目的就是不断更新神经网络的参数来优化目标函数
        # 我们只需将输入的特征数和输出的特征数传递给torch.nn.Linear类，就会自动生成对应维度的权重参数和偏置
        self.fc1 = nn.Linear(N_STATES, 200)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization,权重初始化，利用正态进行初始化
        self.fc2 = nn.Linear(200, 200)
        self.fc2.weight.data.normal_(0, 0.1)
        # self.fc3 = nn.Linear(128, 128)
        # self.fc3.weight.data.normal_(0, 0.1)
        # 第二层神经网络，用于输出action
        self.out = nn.Linear(200, N_ACTIONS)
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
        self.EPSILON=0
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
        if (self.EPSILON < 0.9):
            self.EPSILON += 0.0001
        # 这里只输入一个sample
        if np.random.uniform() < self.EPSILON:   # greedy
            # print("self.Epsion",self.EPSILON)
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
        sum_loss.append(loss.item())
        print("损失",loss)
        # 清空优化器里的梯度信息
        self.optimizer.zero_grad()
        # 误差反向传播
        loss.backward()
        # 更新参数
        self.optimizer.step()

class greedy():
    def greedyuser(self):
        greedyu=copy.deepcopy(init_user_demand())
        greedyrb=copy.deepcopy(RB)
        eachrb=greedyrb/T
        # 获得当前状态找到缺车区域
        # 给用户进行贪心匹配缺车区域，并预算约束
        # 再根据非线性优化计算得到改时间段的d
        # 再计算所有时间段的d之和

def run_this():
    dqn = DQN()
    print('\nCollecting experience...')
    # run this
    ll=[]
    sumreward=[]
    # init_user = init_user_demand()
    init_s=init_state()
    for i_episode in range(EPISODE):
        # 初始化环境
        r=0
        sum_r=0
        # done=False
        # 初始化状态以及下一时间段的状态
        user = copy.deepcopy(init_user)
        region = copy.deepcopy(init_region)
        preregion=copy.deepcopy(region)
        s = copy.deepcopy(init_s)
        s_=copy.deepcopy(init_s)
        s0=copy.deepcopy(s)
        removeuser=list()
        removeuserreggion=list()
        preremove=list()
        preremoveregion=list()


        for t in range (T):
            # 由于用户的骑行时间一般为十分钟，每十分钟处理一次用户，自行车还车是一个阶段
            # 进入下一个时间槽后，将当前状态的用户数以及下一状态的用户还车位置都清0
            # print(user[1])
            for i in range (regionnum):
                s[regionnum + i] = 0
                s_[2 * regionnum + i] = 0
                region[i][0]=0

                # preregion[i][1]=s[i]
            preuser=copy.deepcopy(user[t])

            # 计算未优化前用户的缺车数（来作为reward的分母）[根据用户的初始还车区域和新用户到来后来计算]
            # 将这一阶段的用户的还车区域存于一个数组中，用于下一阶段计算reward(直接计算preregion（每个区域的车的数量）)
            # 判断用户的初始骑车区域，同时并更新状态，更新到s_
            for i in range(len(user[t])):
                if (user[t][i][0] == cell * celllength and user[t][i][1] == cell * celllength):
                    tempa = int(cell * cell - 1)
                elif (user[t][i][0] == cell * celllength):
                    tempa = int(user[t][i][1] / celllength) * cell + int(user[t][i][0] / celllength) - 1
                elif (user[t][i][1] == cell * celllength):
                    tempa = int(user[t][i][1] / celllength) * cell + int(user[t][i][0] / celllength) - cell
                else:
                    tempa = int(user[t][i][1] / celllength) * cell + int(user[t][i][0] / celllength)
                if (tempa < cell * cell):
                    if (s_[tempa] <= 0):
                        # 将多余的用户存于数组中，循环结束后再删除,同时存取没取到车的用户和位置
                        removeuser.append(user[t][i])
                        removeuserreggion.append(i)
                        removeuserreggion.append(tempa)
                        s[regionnum + tempa] += 1
                    else:
                        s[regionnum + tempa] += 1
                        s_[tempa]-=1


                    if(t!=0):
                        if(preregion[tempa][1]<=0):
                            preremove.append(preuser[i])
                            preremoveregion.append(tempa)
                        else:
                            preregion[tempa][1]-=1

            for i in range(len(removeuser)):
                # 当前区域离开的用户应到其周围区域去骑车(即附近且有车的区域去骑车),这个惩罚算的是调度时那个阶段的
                #         将没有骑到车的无效用户移除
                user[t].remove(removeuser[i])

                # 计算下一时间段的缺车区域
            if (t != T - 1):
                for i in range(len(user[t + 1])):
                    if (user[t + 1][i][0] == cell * celllength and user[t + 1][i][1] == cell * celllength):
                        tempa = int(cell * cell - 1)
                    elif (user[t + 1][i][0] == cell * celllength):
                        tempa = int(user[t + 1][i][1] / celllength) * cell + int(user[t + 1][i][0] / celllength) - 1
                    elif (user[t + 1][i][1] == cell * celllength):
                        tempa = int(user[t + 1][i][1] / celllength) * cell + int(
                            user[t + 1][i][0] / celllength) - cell
                    else:
                        tempa = int(user[t + 1][i][1] / celllength) * cell + int(user[t + 1][i][0] / celllength)
                    if (tempa < cell * cell):
                        region[tempa][0] += 1
            regionnn = []
            # print("region[0],下个是时间段的用户需求",region)
            for i in range(regionnum):
                region[i][1] = s_[i]
                if (region[i][0] - region[i][1] > 0):
                    region[i][2] = region[i][0] - region[i][1]
                else:
                    region[i][2] = 0
                regionnn.append(region[i][0])
            for i in range(regionnum):
                s[2 * regionnum + i] = region[i][2]
            # print("regionnn下一时段缺车",sum(regionnn),regionnn)


            if(t!=0):

                r=((len(user[t])/(usernum*12))*10)-0.6
                print("len(remove).len(preremove)", len(removeuser), len(preremove))
                print("action", action,"     RB_t", RB_t,"     r", r,"     t",t)
                # print("action",action)

                # print("user(t)",user[t])
                dqn.store_transition(s0, action, r, s)
                # s首先需要存储记忆，记忆库中有一些东西之后才能学习（前200步都是在存储记忆,大于200之后每5步学习一次）
                if dqn.memory_counter > MEMORY_CAPACITY:
                    dqn.learn()
                sum_r += r

            # 用户取车后的状态+用户还车的区域即为preregion，preregion用于下一阶段计算用户的缺车情况
            for i in range (regionnum):
                preregion[i][1]=s_[i]

            # 计算用户的本来还车区域(还要计算离开的用户)
            for i in range(len(user[t])):
                if (user[t][i][2] == cell * celllength and user[t][i][3] == cell * celllength):
                    tempa = int(cell * cell - 1)
                elif (user[t][i][2] == cell * celllength):
                    tempa = int(user[t][i][3] / celllength) * cell + int(user[t][i][2] / celllength) - 1
                elif (user[t][i][3] == cell * celllength):
                    tempa = int(user[t][i][3] / celllength) * cell + int(user[t][i][2] / celllength) - cell
                else:
                    tempa = int(user[t][i][3] / celllength) * cell + int(user[t][i][2] / celllength)
                if (tempa < cell * cell):
                    preregion[tempa][1]+=1
            preregionn=[]
            for i in range (len(preregion)):
                preregionn.append(preregion[i][1])
            # print("preregion本来还",preregionn)


            # print(dqn.eval_net.fc1.weight)
            a= dqn.choose_action(s)
            if isinstance(a,int):
                action=copy.deepcopy(a)
            else:
                action=a[0]
            RB_t = (action / N_ACTIONS) * s[3 * regionnum]
            # RB_t=0
            # RB_t=10000000


            # 计算sumlackbike
            sumlackbike = 0
            for i in range(len(region)):
                if (region[i][2] > 0):
                    sumlackbike += region[i][2]
            # print("sumlackbike",t+1,sumlackbike)


            # 当有不缺车的情况发生时，会有很大奖励
            # 执行重平衡任务，来得到用户的还车地点
            # print("len(user[t])",len(user[t]))
            if(len(user[t])!=0 and sumlackbike!=0):
                kmtest = km(region, user[t], celllength,RB_t,0.1,0.01,cell)
                tempuser=kmtest.finaluser_greedy()
                # ulp1 = ulpkm(user=user[t], region=region, pB=1, k=100, B=RB_t, cell=cell, celllength=celllength)
                # tempuser, tempfit = ulp1.run()
                # tempuser为各个用户的终点，tempfit为最小d
                # 判断用户还车的区域来更新状态
                for i in range(len(user[t])):
                    if (tempuser[i][5] == cell * celllength and tempuser[i][6] == cell * celllength):
                        tempb = cell * cell - 1
                    elif (tempuser[i][5] == cell * celllength):
                        tempb = int(tempuser[i][6] / celllength) * cell + int(tempuser[i][5] / celllength) - 1
                    elif (tempuser[i][6] == cell * celllength):
                        tempb = int(tempuser[i][6] / celllength) * cell + int(
                            tempuser[i][5] / celllength) - cell
                    else:
                        tempb = int(tempuser[i][6] / celllength) * cell + int(tempuser[i][5] / celllength)
                    # 得到上一阶段的用户还车地点来更新s_
                    if (tempb <= cell * cell):
                        s_[tempb] += 1
            # 没有缺车的情况
            elif(sumlackbike==0):
                for i in range(len(user[t])):
                    if (user[t][i][2] == cell * celllength and user[t][i][3] == cell * celllength):
                        tempb = cell * cell - 1
                    elif (user[t][i][2] == cell * celllength):
                        tempb = int(user[t][i][3] / celllength) * cell + int(user[t][i][2] / celllength) - 1
                    elif (user[t][i][3] == cell * celllength):
                        tempb = int(user[t][i][3] / celllength) * cell + int(
                            user[t][i][2] / celllength) - cell
                    else:
                        tempb = int(user[t][i][3] / celllength) * cell + int(user[t][i][2] / celllength)
                    # 得到上一阶段的用户还车地点来更新s_
                    if (tempb <= cell * cell):
                        s_[tempb] += 1
                r=1
            else:
                r=0

            # print("s_实际还",s_)
                # balancer = tempfit


            s_[3*regionnum]-=RB_t
            # 计算当前T个时间段的总reward
            del removeuser[:]
            del preremove[:]
            del removeuserreggion[:]
            del preremoveregion[:]
            s0=copy.deepcopy(s)
            s = copy.deepcopy(s_)
        print("这一代的总奖励",sum_r)
        sumreward.append(sum_r)
        last_100_avgs = np.array(sumreward[max(0, i_episode - 100):i_episode + 1]).mean()
        ll.append(last_100_avgs)
        print(i_episode)
    plt.figure(1)
    plt.xlabel("iterators", size=14)
    plt.ylabel("reward", size=14)
    t = np.array([t for t in range(0, EPISODE)])  # 迭代次数
    plt.plot(0, 3, color='g')
    fitness1 = np.array(sumreward)
    # fitness2=np.array(sum_loss)
    plt.plot(t, ll, label="greedy", color='b', linewidth=1)
    # plt.plot(t, fitness2, label="loss", color='r', linewidth=1)
    plt.show()

def plotLearning(scores, filename, x=None, window=5):
    print("------------------------Saving Picture--------------------------------")
    N = len(scores)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])

    if x is None:
        x = [i for i in range(N)]

    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)

    plt.legend(labels=['DQN'])

    plt.savefig(filename)




run_this()