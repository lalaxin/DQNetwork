"""
使用的是自己写的二部图匹配
"""
import operator
import random
import time

import numpy as np
# # from helloword import cellnum
# from helloword import Region
# from helloword import User
# # from helloword import usernum
import math
import copy


# print("22",Region)
class km():
    def __init__(self,region,user,cellength,B,pB,k):
        self.sumlackbike=0
        self.ep=1e-10
        self.region=region
        self.user=user
        self.usernum=len(self.user)
        self.cellength = cellength
        self.B = B
        self.pB = pB
        self.k = k
        # self.cellnum=cellnum

    # km最优匹配
    # 车匹配区域，左边表示车，相当于用户数，右边表示区域内缺车数（即缺的车数总和）
    # 生成的数组（用户横坐标，用户纵坐标，区域缺车横坐标，区域缺车纵坐标，权值（两者之间的距离1/d））
    def build_graph(self):
        for i in range(len(self.region)):
            if (self.region[i][2] > 0):
                self.sumlackbike += self.region[i][2]
        self.adj = [[0 for i in range(int(self.sumlackbike))] for i in range(int(self.usernum))]
        for i in range(self.usernum):
            a = 0
            for j in range(len(self.region)):
                if (self.region[j][2] > 0):
                    for k in range(self.region[j][2]):
                        a += 1
                        # 计算用户目的地和各个缺车区域之间的距离
                        d = -math.sqrt(math.pow((self.user[i][2] - self.region[j][3]), 2) + math.pow((self.user[i][3] - self.region[j][4]), 2))
                        # 将大于用户最大步行距离
                        if (-d > self.cellength):
                            d = -np.inf

                        self.adj[i][a - 1] = [self.user[i][2], self.user[i][3], self.region[j][3], self.region[j][4], d]

        self.adj_matrix = [[0 for i in range(self.sumlackbike)] for i in range(self.usernum)]
        for i in range(self.usernum):
            for j in range(self.sumlackbike):
                self.adj_matrix[i][j] = self.adj[i][j][4]

        if(self.sumlackbike<self.usernum):
            self.adj_matrix=np.transpose(self.adj_matrix)


        # 有可能出现usernum或sumlackbike为0的情况
        # print("sumlackbike", self.sumlackbike)


        self.label_left=np.max(self.adj_matrix,axis=1) # 取大函数，axis=1表示横向
        self.label_right=np.zeros(max(self.usernum,self.sumlackbike))
        self.match_right=np.ones(max(self.usernum,self.sumlackbike))*np.nan   #初始化匹配结果
        self.visit_left=np.zeros(min(self.usernum,self.sumlackbike))
        self.visit_right=np.zeros(max(self.usernum,self.sumlackbike)) # visit_right表示右边的缺车是否匹配
        self.slack_right=np.ones(max(self.usernum,self.sumlackbike))*np.inf # np.inf表示最大的正数,记录每个汉子如果能被妹子倾心最少还需要多少期望值




    # 寻找增广路，深度优先
    def find_path(self,i):
        self.visit_left[i] = 1
        for j, match_weight in enumerate(self.adj_matrix[i]):  # enumerate是内置函数，输出的是（计数值，值）
            if self.visit_right[j]: continue  # 已被匹配（解决递归中的冲突）（跳过当前循环的剩余语句，进行下一句循环）
            gap =self.label_left[i] + self.label_right[j] - match_weight  # （匹配时两方的期望和要等于两人之间的好感度即match_right）计算代沟
            # print("gap",i,j,gap)
            if gap <= self.ep:
                # 找到可行匹配
                self.visit_right[j] = True
                if np.isnan(self.match_right[j]) or self.find_path(int(self.match_right[j])):  # j未被匹配，或虽然j已被匹配，但是j的已匹配对象有其他可选
                    self.match_right[j] = i
                    return True
            else:
                # 计算变为可行匹配需要的顶标改变量
                if self.slack_right[j] > gap: self.slack_right[j] = gap  # slack理解为该缺车为了被用户匹配，还需要多少期望值，来最小化它
        return False

    # km主函数
    def KM(self):
        self.build_graph()
        distance=0
        if (self.usernum <= self.sumlackbike):  # 每个用户都可以被匹配的情况下
            for i in range(self.usernum):
                #重置辅助变量
                self.slack_right=np.ones(max(self.usernum,self.sumlackbike))*np.inf
                while True:  # 为每个用户找到缺车区域，若找不到就降低期望值，直到找到为止(由于两边数量不一样，不一定给每个用户都能找到匹配# )
                    #每一轮需重置辅助变量
                    if(self.label_left[i]==-np.inf):
                        break
                    self.visit_left = np.zeros(min(self.usernum, self.sumlackbike))
                    self.visit_right = np.zeros(max(self.usernum, self.sumlackbike))  # visit_right表示右边的缺车是否匹配
                    if self.find_path(i): break  # 如果找到匹配项就退出，未找到则降低期望值
                    d = np.inf  # 最小可降低的期望值，其表示最大的正数
                    for j, slack in enumerate(self.slack_right):
                        if not self.visit_right[j] and slack < d:
                            d = slack
                    for k in range(self.usernum):
                        if self.visit_left[k]: self.label_left[k] -= d  # 所有访问过的用户降低期望值
                    for n in range(self.sumlackbike):
                        if self.visit_right[n]:
                            self.label_right[n] += d  # 所有访问过的男生增加期望值
                    # print("match_right", self.match_right)
                    # print("slack_right", self.slack_right)
                    # print("label_left", self.label_left)
                    # print("label_right", self.label_right)



            for t,weight in enumerate(self.match_right):
                if not np.isnan(self.match_right[t]):
                    distance += self.adj[int(weight)][t][4]
                    self.user[int(weight)][5]=self.adj[int(weight)][t][2]
                    self.user[int(weight)][6]=self.adj[int(weight)][t][3]

        else:  # 用户数大于缺车数时，反过来，用户来匹配缺车数
            # 初始化辅助变量
            for i in range(self.sumlackbike):
                self.slack_right = np.ones(max(self.usernum, self.sumlackbike)) * np.inf
                while True:  #
                    if (self.label_left[i]==-np.inf):
                        break
                    self.visit_left = np.zeros(min(self.usernum, self.sumlackbike))
                    self.visit_right = np.zeros(max(self.usernum, self.sumlackbike))  # visit_right表示右边的缺车是否匹配
                    if self.find_path(i):
                        break  # 如果找到匹配项就退出，未找到则降低期望值

                    d = np.inf  # 最小可降低的期望值
                    for j, slack in enumerate(self.slack_right):
                        if not self.visit_right[j] and slack < d:    #and两者都满足 visit—_righe为0且slack<d时成立
                            d = slack
                    for k in range(self.sumlackbike):
                        if self.visit_left[k]: self.label_left[k] -= d  # 所有访问过的女生降低期望值
                    for n in range(self.usernum):
                        if self.visit_right[n]:
                            self.label_right[n] += d  # 所有访问过的男生增加期望值
                    # print("match_right", self.match_right)
                    # print("slack_right", self.slack_right)
                    # print("label_left", self.label_left)
                    # print("label_right", self.label_right)
                # print(self.match_right)
            for t,weight in enumerate(self.match_right):
                if not np.isnan(self.match_right[t]):
                    distance+=self.adj[t][int(weight)][4]
                    self.user[t][5]=self.adj[t][int(weight)][2]
                    self.user[t][6]=self.adj[t][int(weight)][3]

        # print("kmdistance",distance)
        # print("matchright",self.match_right)
        return self.user

    def finaluser_greedy(self):
        self.KM()
        weight = []
        userid = []
        value = []
        MW = self.B
        for i in range(len(self.user)):
            if (self.user[i][5] != -1 and self.user[i][6] != -1):
                lslarr = math.sqrt(
                    math.pow((self.user[i][0] - self.user[i][2]), 2) + math.pow((self.user[i][1] - self.user[i][3]),
                                                                                2))  # 起点到终点
                lslact = math.sqrt(
                    math.pow((self.user[i][0] - self.user[i][5]), 2) + math.pow((self.user[i][1] - self.user[i][6]),
                                                                                2))  # 起点到实际位置
                larrlact = math.sqrt(
                    math.pow((self.user[i][5] - self.user[i][2]), 2) + math.pow((self.user[i][6] - self.user[i][3]),
                                                                                2))
                # weight.append(self.pB * (lslact - lslarr) + self.k * (larrlact) * (larrlact))
                weight.append(self.k * (larrlact) * (larrlact))
                userid.append(i)
                value.append(1)
        print("weight前",len(weight),weight)
        init_weight = copy.deepcopy(weight)
        for i in range(len(weight)):
            min_index, min_number = min(enumerate(weight), key=operator.itemgetter(1))
            MW -= min_number
            if (MW < 0):
                break
            else:
                weight[min_index] = 1e10
                userid[min_index]=-1
        for i in range(len(weight)):
            if(weight[i]!=1e10):
                init_weight[i]=-1
        print("weight后",len(weight), weight)
        for i in range(len(userid)):
            if(userid[i]!=-1):
                self.user[userid[i]][5] = -1
                self.user[userid[i]][6] = -1
        temp=len(self.user)
        for i in range(len(self.user)):
            if(self.user[i][5]==-1 and self.user[i][6]==-1):
                temp-=1
                self.user[i][5]=self.user[i][2]
                self.user[i][6] = self.user[i][3]
        print("执行任务的用户数temp",temp)
        return self.user
            # 求出所有的匹配的t好感度之和
        # 根据match_right[i]来确定用户和那一区域匹配

    # 用于贪心策略的greedy取值
    def finaluser_greedy_greedy(self):
        self.KM()
        weight = []
        userid = []
        value = []
        MW = self.B
        for i in range(len(self.user)):
            if (self.user[i][5] != -1 and self.user[i][6] != -1):
                lslarr = math.sqrt(
                    math.pow((self.user[i][0] - self.user[i][2]), 2) + math.pow((self.user[i][1] - self.user[i][3]),
                                                                                2))  # 起点到终点
                lslact = math.sqrt(
                    math.pow((self.user[i][0] - self.user[i][5]), 2) + math.pow((self.user[i][1] - self.user[i][6]),
                                                                                2))  # 起点到实际位置
                larrlact = math.sqrt(
                    math.pow((self.user[i][5] - self.user[i][2]), 2) + math.pow((self.user[i][6] - self.user[i][3]),
                                                                                2))
                # weight.append(self.pB * (lslact - lslarr) + self.k * (larrlact) * (larrlact))
                weight.append(self.k * (larrlact) * (larrlact))
                userid.append(i)
                value.append(1)
        init_weight=copy.deepcopy(weight)
        for i in range(len(weight)):
            min_index, min_number = min(enumerate(weight), key=operator.itemgetter(1))
            MW -= min_number
            if (MW < 0):
                break
            else:
                weight[min_index] = 1e10
                userid[min_index]=-1
        for i in range(len(weight)):
            if(weight[i]!=1e10):
                init_weight[i]=-1
        for i in range(len(userid)):
            if(userid[i]!=-1):
                self.user[userid[i]][5] = -1
                self.user[userid[i]][6] = -1
        temp=len(self.user)
        for i in range(len(self.user)):
            if(self.user[i][5]==-1 and self.user[i][6]==-1):
                temp-=1
                self.user[i][5]=self.user[i][2]
                self.user[i][6] = self.user[i][3]
        print("接取到任务的用户数temp",temp)
        return init_weight,self.user
            # 求出所有的匹配的t好感度之和
        # 根据match_right[i]来确定用户和那一区域匹配

    def finaluser_bag(self):
        self.KM()
        weight = []
        userid = []
        value = []
        MW = self.B
        for i in range(len(self.user)):
            if (self.user[i][5] != -1 and self.user[i][6] != -1):
                lslarr = math.sqrt(
                    math.pow((self.user[i][0] - self.user[i][2]), 2) + math.pow((self.user[i][1] - self.user[i][3]),
                                                                                2))  # 起点到终点
                lslact = math.sqrt(
                    math.pow((self.user[i][0] - self.user[i][5]), 2) + math.pow((self.user[i][1] - self.user[i][6]),
                                                                                2))  # 起点到实际位置
                larrlact = math.sqrt(
                    math.pow((self.user[i][5] - self.user[i][2]), 2) + math.pow((self.user[i][6] - self.user[i][3]),
                                                                                2))
                weight.append(self.pB * (lslact - lslarr) + self.k * (larrlact) * (larrlact))
                userid.append(i)
                value.append(self.user[i][4])
        print("weight",weight)
        print("value",value)
        user_id=self.bag_0_1(weight,value,MW)
        # 用户有一定的概率不接受任务，即随机生成一个数，判断是否在概率内
        for i in range(len(userid)):
            if(user_id[i]!=1):
                self.user[userid[i]][5] = self.user[userid[i]][2]
                self.user[userid[i]][6] = self.user[userid[i]][3]
            else:
                if(random.random()>self.user[userid[i][4]]):
                    self.user[userid[i]][5] = self.user[userid[i]][2]
                    self.user[userid[i]][6] = self.user[userid[i]][3]
        return self.user

    def finaluser_bag_greedy(self):
        self.KM()
        weight = []
        userid = []
        value = []
        MW = self.B
        for i in range(len(self.user)):
            if (self.user[i][5] != -1 and self.user[i][6] != -1):
                lslarr = math.sqrt(
                    math.pow((self.user[i][0] - self.user[i][2]), 2) + math.pow((self.user[i][1] - self.user[i][3]),
                                                                                2))  # 起点到终点
                lslact = math.sqrt(
                    math.pow((self.user[i][0] - self.user[i][5]), 2) + math.pow((self.user[i][1] - self.user[i][6]),
                                                                                2))  # 起点到实际位置
                larrlact = math.sqrt(
                    math.pow((self.user[i][5] - self.user[i][2]), 2) + math.pow((self.user[i][6] - self.user[i][3]),
                                                                                2))
                weight.append(self.pB * (lslact - lslarr) + self.k * (larrlact) * (larrlact))
                userid.append(i)
                value.append(self.user[i][4])
        print("weight",weight)
        print("value",value)
        init_weight=copy.deepcopy(weight)
        user_id=self.bag_0_1(weight,value,MW)

        # 用户有一定的概率不接受任务，即随机生成一个数，判断是否在概率内
        for i in range(len(userid)):
            if(user_id[i]!=1):
                init_weight[i]=-1
                self.user[userid[i]][5] = self.user[userid[i]][2]
                self.user[userid[i]][6] = self.user[userid[i]][3]
            else:
                if(random.random()>self.user[userid[i][4]]):
                    self.user[userid[i]][5] = self.user[userid[i]][2]
                    self.user[userid[i]][6] = self.user[userid[i]][3]
        return init_weight,self.user


    def bag_0_1(self,weight, value, weight_most):  # return max value
        num = len(weight)
        user_id=[]
        weight.insert(0, 0)  # 前0件要用
        value.insert(0, 0)  # 前0件要用
        bag = np.zeros((num + 1, weight_most + 1), dtype=np.float)  # 下标从零开始
        for i in range(1, num + 1):
            for j in range(1, weight_most + 1):
                if weight[i] <= j:
                    bag[i][j] = max(bag[i - 1][int(round(j - weight[i]))] + value[i], bag[i - 1][j])
                else:
                    bag[i][j] = bag[i - 1][j]
        # print(bag)
        print('最大价值为:', bag[num][int(weight_most)])
        x = [0 for i in range(num)]
        j = int(weight_most)
        for i in range(num, 0, -1):
            if bag[i][int(j)] > bag[i - 1][int(j)]:
                x[i - 1] = 1
                j -= weight[i - 1]
        print("背包中所装物品为：")
        for i in range(num):
            if x[i]:
                user_id.append(1)
            else:
                user_id.append(0)
                print("第", i + 1, "个", end=' ')
        # 返回取得物品的编号
        return user_id


# cellength=2
# User=[[0,0,5,3,10,-1,-1],[0,0,5,3,10,-1,-1],[0,0,1,3,10,-1,-1],[0,0,1,3,10,-1,-1]]
# Region=[[0, 2, 0, 1.0, 1.0],[0, 2,1, 3.0, 1.0],[0, 2, 0, 5.0, 1.0],[0, 2, 0, 1.0, 3.0],[0, 2, 2, 3.0, 3.0],[0, 2, 0, 5.0, 3.0],[0, 2, 1, 1.0, 5.0],[0, 2, 0, 3.0, 5.0],[0, 2, 1, 5.0, 5.0]]
# kmtest=km(Region,User,cellength,40,1,5)
# kmtest.finaluser()
# # print(User)