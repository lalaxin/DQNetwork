"""
使用贪心进行匹配，即使用贪心，距离最短来进行匹配，为每个区域设置不同的权值，该权值直接和距离以及该区域的缺车程度相关
"""
import operator
import random
random.seed(1)

import numpy as np
import pandas as pd
import math
import copy



class greedymatch():
    # 输入的region为下一个时间段的需求
    # print(type(-np.inf))
    def __init__(self,region,user,cellength,B,pB,k,cell):
        self.sumlackbike=0
        self.ep=1e-10
        self.region=region
        self.user=user
        self.usernum=len(self.user)
        self.cellength = cellength
        self.B = B
        self.pB = pB
        self.k = k
        self.cell=cell
        self.celllength = cellength
        # self.cellnum=cellnum

    # 新建一个二位数组，即选最小的用户-单车最小值
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
                        # 当用户的目的地和缺车区域在同一区域时，不应计算成本d，距离为0，只有相邻区域才需要计算
                        if(self.region[j][3]-self.celllength/2<=self.user[i][2]<self.region[j][3]+self.celllength/2 and self.region[j][4]-self.celllength/2<=self.user[i][3]<self.region[j][4]+self.celllength/2):
                            d=0.0
                        elif(self.region[j][3]-self.celllength/2<=self.user[i][2]<self.region[j][3]+self.celllength/2 and self.user[i][3]==self.region[j][4]+self.celllength/2==self.celllength*self.cell):
                            d=0.0
                        elif (self.region[j][4] - self.celllength / 2 <= self.user[i][3] < self.region[j][4] + self.celllength / 2 and self.user[i][2] == self.region[j][3] + self.celllength / 2 == self.celllength * self.cell):
                            d = 0.0
                        elif(self.user[i][2]==self.region[j][3]+self.celllength/2==self.celllength * self.cell and self.user[i][3]==self.region[j][4]+self.celllength/2==self.celllength * self.cell):
                            d=0.0
                        else:
                            d = -math.sqrt(math.pow((self.user[i][2] - self.region[j][3]), 2) + math.pow((self.user[i][3] - self.region[j][4]), 2))
                        # 将大于用户最大步行距离
                        if (-d > self.cellength):
                            d = -np.inf

                        self.adj[i][a - 1] = [self.user[i][2], self.user[i][3], self.region[j][3], self.region[j][4], d]

        # print("self.adj",self.adj)
        self.adj_matrix = [[0 for i in range(self.sumlackbike)] for i in range(self.usernum)]
        for i in range(self.usernum):
            for j in range(self.sumlackbike):
                self.adj_matrix[i][j] = self.adj[i][j][4]
        self.adj_matrix=np.array(self.adj_matrix)
        # if(len(self.user)>self.sumlackbike):
        #     self.adj_matrix=np.transpose(self.adj_matrix)

        # for i in range (len(self.adj_matrix)):
        #     print("self.adj_matrix列数", self.adj_matrix[i])

        # print("self.adj_matrix行数", len(self.adj_matrix[0]))
        count=0
        # 数组形式输出self.adj_matrix
        print("len(user),sumlackbike",len(self.user),self.sumlackbike)
        # print("self.adj_matrix")
        # for i in range (len(self.adj_matrix)):
        #     print(self.adj_matrix[i])
        neededprice=[]
        while not ((self.adj_matrix==-np.inf).all()):
            xy = np.argmax(self.adj_matrix)
            x_length = len(self.adj_matrix[0])
            x_row = int(xy / x_length)
            x_col = xy % x_length
            x_price = self.k * self.adj_matrix[x_row][x_col] * self.adj_matrix[x_row][x_col]
            neededprice.append(x_price)

            print("x_price", x_price)
            if (x_price<self.B):
                count += 1
                self.B -= x_price
                # print("x_row", x_row)
                # print("x_col", x_col)
                # print("len(self.adj_matrix)",len(self.adj_matrix))
                # 若已匹配，则该列已被匹配，即该缺车，则将这一列都设为-inf

                for i in range(len(self.adj_matrix)):
                    # print("type(self.adj_matrix[i][x_col]", type(self.adj_matrix[i][x_col]))
                    self.adj_matrix[i][x_col] = -np.inf
                for i in range(len(self.adj_matrix[0])):
                    self.adj_matrix[x_row][i] = -np.inf
                # 匹配成功，为用户设置还车区域
                self.user[x_row][5] = self.adj[x_row][x_col][2]
                self.user[x_row][6] = self.adj[x_row][x_col][3]
            else:
                for i in range(len(self.adj_matrix)):
                    self.adj_matrix[i][x_col] = -np.inf
                for i in range(len(self.adj_matrix[0])):
                    self.adj_matrix[x_row][i] = -np.inf

        # while(self.B>=0):
        #     # print(np.max(self.adj_matrix))
        #     xy=np.argmax(self.adj_matrix)
        #     x_length=len(self.adj_matrix[0])
        #     x_row=int(xy/x_length)
        #     x_col=xy%x_length
        #     x_price=self.k*self.adj_matrix[x_row][x_col]*self.adj_matrix[x_row][x_col]
        #     neededprice.append(x_price)
        #     print("x_price",x_price)
        #     if(x_price>self.B):
        #         break
        #     else:
        #         count+=1
        #         self .B-=x_price
        #         # 若已匹配，则该列已被匹配，即该缺车，则将这一列都设为-inf
        #         for i in range (len(self.adj_matrix)):
        #             self.adj_matrix[i][x_col] = -np.inf
        #         for i in range (len(self.adj_matrix[0])):
        #             self.adj_matrix[x_row][i] = -np.inf
        #         # 匹配成功，为用户设置还车区域
        #         self.user[x_row][5]=self.adj[x_row][x_col][2]
        #         self.user[x_row][6]=self.adj[x_row][x_col][3]
        print("执行任务的人数为：",count)
        # 将未匹配到缺车的用户换到自己本来的还车点


        for i in range (len(self.user)):
            if(self.user[i][5]==-1 and self.user[i][6]==-1):
                self.user[i][5]=self.user[i][2]
                self.user[i][6] = self.user[i][3]
        # print("needprice",len(neededprice))
        # print("count",count)
        return self.user,sum(neededprice)

    def newbuild_graph(self):
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
                        # 当用户的目的地和缺车区域在同一区域时，不应计算成本d，距离为0，只有相邻区域才需要计算
                        if(self.region[j][3]-self.celllength/2<=self.user[i][2]<self.region[j][3]+self.celllength/2 and self.region[j][4]-self.celllength/2<=self.user[i][3]<self.region[j][4]+self.celllength/2):
                            d=0.0
                        elif(self.region[j][3]-self.celllength/2<=self.user[i][2]<self.region[j][3]+self.celllength/2 and self.user[i][3]==self.region[j][4]+self.celllength/2==self.celllength*self.cell):
                            d=0.0
                        elif (self.region[j][4] - self.celllength / 2 <= self.user[i][3] < self.region[j][4] + self.celllength / 2 and self.user[i][2] == self.region[j][3] + self.celllength / 2 == self.celllength * self.cell):
                            d = 0
                        elif(self.user[i][2]==self.region[j][3]+self.celllength/2==self.celllength * self.cell and self.user[i][3]==self.region[j][4]+self.celllength/2==self.celllength * self.cell):
                            d=0.0
                        else:
                            d = -math.sqrt(math.pow((self.user[i][2] - self.region[j][3]), 2) + math.pow((self.user[i][3] - self.region[j][4]), 2))
                        # 将大于用户最大步行距离
                        if (-d > self.cellength):
                            d = -np.inf

                        self.adj[i][a - 1] = [self.user[i][2], self.user[i][3], self.region[j][3], self.region[j][4], d]

        self.adj_matrix = [[0 for i in range(self.sumlackbike)] for i in range(self.usernum)]
        for i in range(self.usernum):
            for j in range(self.sumlackbike):
                self.adj_matrix[i][j] = self.adj[i][j][4]
        self.adj_matrix=np.array(self.adj_matrix)
        # if(len(self.user)<self.sumlackbike):
        #     self.adj_matrix=np.transpose(self.adj_matrix)

        # for i in range (len(self.adj_matrix)):
        #     print("self.adj_matrix列数", self.adj_matrix[i])

        # print("self.adj_matrix行数", len(self.adj_matrix[0]))
        count=0
        # 数组形式输出self.adj_matrix
        # print("self.adj_matrix")
        # for i in range (len(self.adj_matrix)):
        #     print(self.adj_matrix[i])
        neededprice=[]


        while(self.B>=0):
            # print(np.max(self.adj_matrix))
            xy=np.argmax(self.adj_matrix)
            x_length=len(self.adj_matrix[0])
            x_row=int(xy/x_length)
            x_col=xy%x_length
            x_price=self.k*self.adj_matrix[x_row][x_col]*self.adj_matrix[x_row][x_col]
            neededprice.append(x_price)
            print("x_price",x_price)
            if(x_price>self.B):
                break
            else:
                count+=1
                self .B-=x_price
                # print("type(self.adj_matrix[i][x_col]", type(self.adj_matrix[0][x_col]))
                # 若已匹配，则该列已被匹配，即该缺车，则将这一列都设为-inf
                for i in range (len(self.adj_matrix)):
                    # print("type(self.adj_matrix[i][x_col]",type(self.adj_matrix[i][x_col]))
                    self.adj_matrix[i][x_col] = -np.inf
                for i in range (len(self.adj_matrix[0])):
                    self.adj_matrix[x_row][i] = -np.inf
                # 匹配成功，为用户设置还车区域
                self.user[x_row][5]=self.adj[x_row][x_col][2]
                self.user[x_row][6]=self.adj[x_row][x_col][3]
        # print("执行任务的人数为：",count)
        # 将未匹配到缺车的用户换到自己本来的还车点


        for i in range (len(self.user)):
            if(self.user[i][5]==-1 and self.user[i][6]==-1):
                self.user[i][5]=self.user[i][2]
                self.user[i][6] = self.user[i][3]
        # print("needprice",len(neededprice))
        # print("count",count)
        return self.user,self.B



# cellength=2
# User=[[random.randint(0,6),random.randint(0,6),random.randint(0,6),random.randint(0,6),10,-1,-1],[random.randint(0,6),random.randint(0,6),random.randint(0,6),random.randint(0,6),10,-1,-1],[random.randint(0,6),random.randint(0,6),random.randint(0,6),random.randint(0,6),10,-1,-1],[random.randint(0,6),random.randint(0,6),random.randint(0,6),random.randint(0,6),10,-1,-1]]
# Region=[[0, 2, 0, 1.0, 1.0],[0, 2,1, 3.0, 1.0],[0, 2, 0, 5.0, 1.0],[0, 2, 0, 1.0, 3.0],[0, 2, 2, 3.0, 3.0],[0, 2, 0, 5.0, 3.0],[0, 2, 1, 1.0, 5.0],[0, 2, 0, 3.0, 5.0],[0, 2, 1, 5.0, 5.0]]
# kmtest=greedymatch(Region,User,cellength,100,1,5,3)
# kmtest.build_graph()
# # print(User)