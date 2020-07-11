"""
随机为每个时间段分配预算（或平分），即可愿意完成任务的用户都为其分配
"""
# 参数设置
import copy
import random

import xlrd

from simple.simplekm2 import km
from users3_1 import getuser

T=12 #时间时段
RB=50000000#预算约束
# 横向网格数
cell=10
# 单个网格长度
celllength=300
regionnum=cell*cell #区域个数
EPISODE=1000 #迭代次数

init_region = list()
def init_region2():
    userregion = []
    excel = xlrd.open_workbook("../userregion.xlsx")
    sheet = excel.sheet_by_name("sheet1")
    userregion=sheet.row_values(0)
    print("userregion",userregion)
    for i in range(regionnum):
        regionn = [0, int(userregion[i]), 0, (i % cell) * celllength + celllength / 2,(int(i / cell)) * celllength + celllength / 2]
        # regionn =[0,int(userregion[i]*99/539)+1,0,(i%cell)*celllength+celllength/2,(int(i/cell))*celllength+celllength/2]
        init_region.append(regionn)
        # print(region)
init_region2()


number=0
# region0=[0,1,1,0,0,0,0,1,2,0,2,2,1,0,2,1]
for i in range(regionnum):
      number += init_region[i][1]
print("initregion",init_region)
print("number",number)

# 真实需求
def init_user_demand():
    # 将excel的数据存于数组中
    userdemand=getuser().getusers()
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
    user=copy.deepcopy(init_user)
    region=copy.deepcopy(init_region)
    for t in range (T):
        avery_RB=random.random()*RB
        RB-=avery_RB
        removeuser=[]
        # 计算离开的用户数
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
        for i in range(len(removeuser)):
            # 当前区域离开的用户应到其周围区域去骑车(即附近且有车的区域去骑车),这个惩罚算的是调度时那个阶段的
            #         将没有骑到车的无效用户移除
            user[t].remove(removeuser[i])
        if(t!=0 and t!=T-1):
            r=len(user[t])
            print("reward",r)
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
            regionnn.append(region[i][1])
        # print("regionnn", sum(regionnn), regionnn)

        # 还车时更新region[1]，进入下一层循环
        kmtest = km(region, user[t], celllength, avery_RB, 1, 10)
        tempuser = kmtest.finaluser_greedy()

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


    print("sumr",sum(sum_r),sum_r)