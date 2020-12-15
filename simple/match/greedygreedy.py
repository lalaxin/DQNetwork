"""
贪心为每个时间段分配预算，即可愿意完成任务的用户都为其分配
"""
# 参数设置
import copy
import random
random.seed(1)


import xlrd

from simple.match.greedymatch import greedymatch

from mergeuser import mergeuser

from queue import Queue


random.seed(1)
print("random.randint                        ",random.randint(0, 120))

T=18 #时间时段
RB=10000#预算约束
# 横向网格数
cell=5
# 单个网格长度
celllength=600
regionnum=cell*cell #区域个数
EPISODE=1000 #迭代次数

# 虚拟数据
usernum=20 #用户数为10

# # # 随机数据
# # #（用户数，区域内车辆数,区域内缺车的数量,中心点横坐标，中心点纵坐标），初始化区域时只需初始化当前区域内的车辆数即可，然后根据用户到来信息求得用户数和缺车数
# init_region = list()
# number=0
# for i in range(regionnum):
#     regionn =[0,0,0,(i%cell)*celllength+celllength/2,(int(i/cell))*celllength+celllength/2]
#     init_region.append(regionn)
#
# # 用户需求,T个时间段的用户需求# 定义用户数组（起点横坐标，起点纵坐标，终点横坐标，终点纵坐标，最大步行距离,期望停车区域横坐标，期望停车区域纵坐标）
# def init_user_demand():
#     userdemand=[[[0]for i in range (usernum)] for t in range (T)]
#     for t in range (T):
#         for i in range (usernum):
#             userdemand[t][i]=[random.randint(0, cell*celllength),random.randint(0, cell*celllength),random.randint(0, cell*celllength),random.randint(0, cell*celllength),cell*celllength*1.4,-1,-1]
#             # userdemand[t][i]=[(random.randint(0, cell - 1) + 1 / 2) * celllength,(random.randint(0, cell - 1) + 1 / 2) * celllength,(random.randint(0, cell - 1) + 1 / 2) * celllength,(random.randint(0, cell - 1) + 1 / 2) * celllength,cell*celllength*1.4,-1,-1]
#             # userdemand[t][i]=[random.randint(0,celllength*cell),random.randint(0,celllength*cell),random.randint(0,celllength*cell),random.randint(0,celllength*cell),cell*celllength*1.4,-1,-1]
#     print("userdemand",userdemand)
#     return userdemand
# init_user=init_user_demand()
#
# def init_user_region():
#     number=0
#     for i in range(len(init_user[0])):
#         if (init_user[0][i][0] == cell * celllength and init_user[0][i][1] == cell * celllength):
#             tempa = int(cell * cell - 1)
#         elif (init_user[0][i][0] == cell * celllength):
#             tempa = int(init_user[0][i][1] / celllength) * cell + int(init_user[0][i][0] / celllength) - 1
#         elif (init_user[0][i][1] == cell * celllength):
#             tempa = int(init_user[0][i][1] / celllength) * cell + int(init_user[0][i][0] / celllength) - cell
#         else:
#             tempa = int(init_user[0][i][1] / celllength) * cell + int(init_user[0][i][0] / celllength)
#         if (tempa < cell * cell):
#             init_region[tempa][1]+=1
#     for i in range(regionnum):
#         number += init_region[i][1]
#     print("number",number)
# init_user_region()
# print("initregion",init_region)

## 真实数据
init_region = list()
def init_region2():
    userregion = []
    excel = xlrd.open_workbook("D:/Python/python_code/DQNetwork/userregion.xlsx")
    sheet = excel.sheet_by_name("15709")
    userregion=sheet.row_values(0)
    print("userregion",userregion)
    for i in range(regionnum):
        regionn = [0, int((1726*3.65)/320), 0, (i % cell) * celllength + celllength / 2,(int(i / cell)) * celllength + celllength / 2]
        # regionn =[0,int(userregion[i]*99/539)+1,0,(i%cell)*celllength+celllength/2,(int(i/cell))*celllength+celllength/2]
        init_region.append(regionn)
        # print(region)
init_region2()


number=0
for i in range(regionnum):
      number += init_region[i][1]
print("initregion",init_region)
print("number",number)

# 真实需求
def init_user_demand():
    # 将excel的数据存于数组中
    userdemand=mergeuser()
    return userdemand
init_user = init_user_demand()


def init_state():
    #状态应包含这一时间段每个区域的（第二个regionnum）用户数，(第1个regionnum)车的供应数以及下一阶段该区域的缺车数（第三个regionnum）
    s=[0 for i in range (3*regionnum+1)]
    for i in range (regionnum):
        s[i]=init_region[i][1] #第i个区域的车的供应量
    s[3*regionnum]=RB
    return s

if __name__ == '__main__':
    r=0
    sum_r=[]
    sum_RB=[]
    user=copy.deepcopy(init_user)
    # for i in range(len(init_user)):
    #     for j in range(len(init_user[i])):
    #         # round()函数 四舍五入
    #         user[i][j][0] = round(init_user[i][j][0])
    #         user[i][j][1] = round(init_user[i][j][1])
    #         user[i][j][2] = (init_user[i][j][2] // celllength + 1 / 2)* celllength
    #         user[i][j][3] = (init_user[i][j][3] // celllength + 1 / 2) * celllength
    region=copy.deepcopy(init_region)
    preregion=copy.deepcopy(init_region)
    bestreward=[]
    worstreward=[]
    print("preregion",preregion)
    preuser=copy.deepcopy(init_user)
    for t in range (T):
        for i in range (regionnum):
            region[i][0]=0
            preregion[i][0]=0
        removeuser=[]
        preremoveuser=[]
        # 计算离开的用户数.使用贪心分配预算
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
                    # 将多余的用户存于数组中，循环结束后再删除,同时存取没取到车的用户和位置
                if(region[tempa][1]<=0):
                    removeuser.append(user[t][i])
                else:
                    region[tempa][1]-=1
        # 不分配预算时的用户数
        for i in range(len(preuser[t])):
            if (preuser[t][i][0] == cell * celllength and preuser[t][i][1] == cell * celllength):
                tempa = int(cell * cell - 1)
            elif (preuser[t][i][0] == cell * celllength):
                tempa = int(preuser[t][i][1] / celllength) * cell + int(preuser[t][i][0] / celllength) - 1
            elif (preuser[t][i][1] == cell * celllength):
                tempa = int(preuser[t][i][1] / celllength) * cell + int(preuser[t][i][0] / celllength) - cell
            else:
                tempa = int(preuser[t][i][1] / celllength) * cell + int(preuser[t][i][0] / celllength)
            if (tempa < cell * cell):
                    # 将多余的用户存于数组中，循环结束后再删除,同时存取没取到车的用户和位置
                if(preregion[tempa][1]<=0):
                    preremoveuser.append(preuser[t][i])
                else:
                    preregion[tempa][1]-=1
        for i in range(len(removeuser)):
            # 当前区域离开的用户应到其周围区域去骑车(即附近且有车的区域去骑车),这个惩罚算的是调度时那个阶段的
            #         将没有骑到车的无效用户移除
            user[t].remove(removeuser[i])
        for i in range(len(preremoveuser)):
            #         将没有骑到车的无效用户移除
            preuser[t].remove(preremoveuser[i])
        if(t!=0 ):
            print("removeruser",len(removeuser),"preremoveuser",len(preremoveuser))
            if (len(preremoveuser) == 0 and len(removeuser) == 0):
                r = 0
            elif (len(preremoveuser) == 0 and len(removeuser) != 0):
                r = -1
            else:
                r=(len(preremoveuser)-len(removeuser))/len(preremoveuser)
            print("reward",r)
            bestreward.append(len(removeuser))
            worstreward.append(len(preremoveuser))
            sum_r.append(r)
        # 计算每个时间段的缺车数
        if (t != T - 1):
            for i in range(len(user[t + 1])):
                if (user[t + 1][i][0] == cell * celllength and user[t + 1][i][1] == cell * celllength):
                    tempa = int(cell * cell - 1)
                elif (user[t + 1][i][0] == cell * celllength):
                    tempa = int(user[t + 1][i][1] / celllength) * cell + int(user[t + 1][i][0] / celllength) - 1
                elif (user[t + 1][i][1] == cell * celllength):
                    tempa = int(user[t + 1][i][1] / celllength) * cell + int(user[t + 1][i][0] / celllength) - cell
                else:
                    tempa = int(user[t + 1][i][1] / celllength) * cell + int(user[t + 1][i][0] / celllength)
                if (tempa < cell * cell):
                    region[tempa][0] += 1
        regionnn = []
        for i in range(regionnum):
            # 根据下一时间段用户及当前区域的车的数量计算缺车数
            if (region[i][0] - region[i][1] > 0):
                region[i][2] = region[i][0] - region[i][1]
            else:
                region[i][2] = 0
            regionnn.append(region[i][2])
        # print("下个时间段的缺车数regionnn", sum(regionnn), regionnn)
        sumlackbike = 0
        for i in range(len(region)):
            if (region[i][2] > 0):
                sumlackbike += region[i][2]
        print("sumlackbike", t + 1, sumlackbike)

        # 还车时更新region[1]，进入下一层循环
        if(len(user[t])!=0 and t!=T-1 and sumlackbike!=0):
            # RB=100000000
            greedytest = greedymatch(region, user[t], celllength, RB, 0.1, 0.01, cell)
            tempuser, tempBudget = greedytest.newbuild_graph()
            print("使用掉B,剩余预算", RB-tempBudget,tempBudget,t)
            RB=tempBudget
        # 使用贪心进行调度
        #     print("len()user[t]",user[t])
        #     print("tempuser",tempuser)
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
                    region[tempb][1]+=1
        elif (sumlackbike == 0):
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
                    region[tempb][1]+=1
            r = 1
        else:
            r = 0
        # 使用贪心调度结束时区域的车的数量

        # 未进行调度
        for i in range(len(preuser[t])):
            if (preuser[t][i][2] == cell * celllength and preuser[t][i][3] == cell * celllength):
                tempb = cell * cell - 1
            elif (preuser[t][i][2] == cell * celllength):
                tempb = int(preuser[t][i][3] / celllength) * cell + int(preuser[t][i][2] / celllength) - 1
            elif (preuser[t][i][3] == cell * celllength):
                tempb = int(preuser[t][i][3] / celllength) * cell + int(
                    preuser[t][i][2] / celllength) - cell
            else:
                tempb = int(preuser[t][i][3] / celllength) * cell + int(preuser[t][i][2] / celllength)
            # 得到上一阶段的用户还车地点来更新s_
            if (tempb <= cell * cell):
                preregion[tempb][1]+=1


    print("sumr",sum(sum_r),sum_r)
    print("bestreward",bestreward)
    print("worstreward",worstreward)

# bestreward [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 1, 0, 0, 3, 0, 0, 3, 3, 0, 1, 2, 2, 0, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 4, 3, 7, 5, 4, 5, 3, 8, 4, 3, 5, 5, 7, 8, 4, 2, 3, 6, 2, 3, 4, 4, 5, 6, 5, 1, 9, 3, 14, 5, 7, 7, 10, 15, 16, 7, 14, 13, 6, 11, 9, 6, 4, 12, 5, 8, 5, 3, 7, 4, 10, 8, 8, 7, 9, 12, 5, 6, 7, 10, 10, 10, 6, 11, 6, 7, 13, 11, 3, 8, 6, 6, 9, 7, 7, 6, 12, 5]
# worstreward[0, 1, 1, 0, 2, 1, 0, 0, 2, 1, 3, 0, 0, 1, 2, 0, 0, 3, 4, 1, 3, 3, 4, 1, 5, 6, 3, 5, 3, 2, 1, 0, 2, 1, 3, 0, 0, 2, 1, 4, 3, 6, 3, 8, 7, 7, 5, 3, 7, 4, 2, 7, 6, 8, 9, 4, 1, 3, 6, 2, 1, 4, 4, 5, 6, 5, 1, 9, 3, 14, 6, 7, 7, 10, 15, 16, 7, 14, 13, 6, 11, 9, 6, 4, 12, 5, 10, 5, 3, 6, 4, 10, 8, 8, 7, 9, 12, 5, 6, 7, 10, 10, 10, 6, 11, 6, 7, 13, 11, 3, 8, 6, 6, 9, 7, 7, 6, 12, 5]
# tempremove [0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 3, 1, 2, 2, 5, 1, 1, 4, 3, 1, 2, 2, 3, 1, 3, 7, 3, 4, 2, 0, 1, 1, 1, 1, 3, 0, 0, 2, 1, 3, 2, 6, 3, 8, 7, 6, 5, 3, 5, 3, 0, 6, 5, 7, 8, 3, 1, 2, 4, 1, 0, 3, 3, 5, 5, 3, 0, 7, 3, 12, 4, 5, 5, 9, 14, 16, 7, 14, 12, 6, 11, 8, 5, 3, 11, 4, 8, 4, 2, 5, 4, 9, 7, 7, 6, 8, 11, 5, 6, 6, 9, 9, 9, 6, 10, 6, 7, 13, 11, 3, 8, 6, 6, 9, 7, 7, 6, 12, 5]
