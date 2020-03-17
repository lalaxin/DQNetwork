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
xcell=2
# 纵向网格数
ycell=2
# 单个网格长度
celllength=3
regionnum=xcell*ycell #区域个数
pricek=2
usernum=10 #用户数为10

# 定义区域(随机车辆数，中心点横坐标，中心点纵坐标)
region = list()
for i in range(regionnum):
    # print(i)
    r =[(i%xcell)*celllength+celllength/2,(int(i/xcell))*celllength+celllength/2]
    # print(r)
    region.append(r)
    # print(region)

# 用户需求,每个时间段用户需求随机到来（随机初始化用户数量）以及用户的起始区域，目的地，用户实际到达的区域以及我们希望用户到达的区域
def init_user_demand():
    userdemand=[[[0]for i in range (usernum)] for t in range (T)]
    for t in range (T):
        for i in range (usernum):
            userdemand[t][i]=[random.randint(0,regionnum-1),random.randint(0,regionnum-1),0,0]
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
N_ACTIONS = usernum*9
# 接收的observation维度
N_STATES = regionnum+1

# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

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


def run_this():
    dqn = DQN()
    print('\nCollecting experience...')
    # run this
    for i_episode in range(400):
        # 初始化环境
        s = init_state()
        s_=[]
        ep_r = 0
        user = init_user_demand()
        for t in range (T):
            # 计算reward（由于下一时刻用户需求到来之后才能计算d，所以先将状态，动作对存起来）
            # 计算reward时也涉及到用户的匹配问题，用户匹配附近的哪个还车区域会使得整个用户的d最小
            r=0
            if (t == 0):
                r = 0
            else:
                s_1=copy.deepcopy(s_)
                for i in range (usernum):
                    if(s_1[user[t][i][0]]!=0):
                        s_1[user[t][i][0]]-=1
                        r+=0   #如果用户在本区域有车，则将其reward设为0
                    else:
                        temp=[user[t][i][0]-xcell-1,user[t][i][0]-xcell,user[t][i][0]-xcell+1,user[t][i][0]-1,user[t][i][0]+1,user[t][i][0]+xcell-1,user[t][i][0]+xcell,user[t][i][0]+xcell+1]
                        # temp=[1,2,3,4,5,6,7,8]
                        # 将该区域的临近区域表示出来，注意边界区域
                        if(user[t][i][0]//celllength==0):
                            temp.remove(user[t][i][0]-xcell-1)
                            temp.remove(user[t][i][0]-xcell)
                            temp.remove(user[t][i][0]-xcell+1)
                        if(user[t][i][1]%celllength==0):
                            temp.remove(user[t][i][0]-1)
                            temp.remove(user[t][i][0]+xcell-1)
                            if (user[t][i][0]-xcell-1) in temp:
                                temp.remove(user[t][i][0]-xcell-1)
                        if(user[t][i][0]%celllength==xcell-1):
                            temp.remove(user[t][i][0]+1)
                            temp.remove(user[t][i][0]+xcell+1)
                            if 3 in temp:
                                temp.remove(user[t][i][0]-xcell+1)
                        if (user[t][i][1] % celllength == ycell - 1):
                            temp.remove(user[t][i][0]+xcell)
                            if 6 in temp:
                                temp.remove(user[t][i][0]+xcell-1)
                            if 8 in temp:
                                temp.remove(user[t][i][0]+xcell+1)
                        # 从temp中有车的区域随机取一个取车,并将其reward设为-d
                        for i in range (len(temp)):
                            temp_=temp[random.randint(0,len(temp)-1)]
                            if (s_1[temp_]!=0):
                                if(abs(temp_-user[t][i][0])==2):
                                    r+=-math.sqrt(2*celllength*celllength)
                                s_1[temp_]-=1
                                break
                            else:
                                temp.remove(temp_)
                        # 若该区域所有的临近区域都无车，则给一个很大的惩罚
                        if not temp:
                            r += -1e10
            # 存储记忆
            if(t!=0):
                dqn.store_transition(s,a,r,s_)
                # 第二轮进行状态转移
                s=copy.deepcopy(s_)

            # dqn.choose_action()一下只能输出一个动作，动作的维度是用户数，输出的a是每一个用户采取的哪一个动作
            # 首先根据当前observation选取一个动作
            a = dqn.choose_action(s)

            # take action 与环境交互，施加改动作在环境中，得到下一个observation_,以及reward，done表示是否终结
            # 根据当前动作和状态来得到下一状态以及reword
            for i in range(len(a)):
                # 期望用户到达的区域
                user[t][i][3]=a[i]//walknum  #取整，判断选取动作属于哪一个区域
                # (x,y)为用户到达的位置
                x=region[user[t][i][3]][1]-(region[user[t][i][3]][1]-region[user[t][i][0]][1])*a[i]%walknum/walknum
                y=region[user[t][i][3]][2]-(region[user[t][i][3]][2]-region[user[t][i][0]][2])*a[i]%walknum/walknum
                # 用户实际到达的区域
                user[t][i][2]=(y//celllength)*xcell+(x//celllength)

            #     更新s_
                s_[user[t][i][0]]=-1
                s_[user[t][i][2]]=+1
                # 先计算给用户步行激励的钱
                userprice=pricek*(math.pow(region[user[t][i][1]][1]-region[user[t][i][2]][1],2)+math.pow(region[user[t][i][1]][2]-region[user[t][i][2]][2],2))
                s_[regionnum]-=userprice      #更新预算约束，若RB<=此用户的激励预算，则本轮学习结束
                if(s_[regionnum]<0):
                    done=True   #若RB<0，预算算完，则本轮学习结束

            ep_r += r

            # s首先需要存储记忆，记忆库中有一些东西之后才能学习（前200步都是在存储记忆,大于200之后每5步学习一次）
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))

            if done:
                break
            # s = s_


run_this()