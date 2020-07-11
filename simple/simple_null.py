"""
在没有任何操作的情况下，12个时间段之后的用户的数量
"""
import copy

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

if __name__ == '__main__':
    r = 0
    sum_r = []
    user = copy.deepcopy(init_user)
    region = copy.deepcopy(init_region)
    for t in range(T):
        avery_RB = RB / T
        removeuser = []
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
                if (region[tempa][1] <= 0):
                    removeuser.append(user[t][i])
                else:
                    region[tempa][1] -= 1
        for i in range(len(removeuser)):
            #         将没有骑到车的无效用户移除
            user[t].remove(removeuser[i])
        if (t != 0 and t != T - 1):
            r = len(user[t])
            print("reward", r)
            sum_r.append(r)
        # print("regionnn", sum(regionnn), regionnn)



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
                region[tempb][1] += 1

    print("sumr", sum(sum_r), sum_r)