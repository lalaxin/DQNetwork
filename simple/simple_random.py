"""
贪心为每个时间段分配预算，即可愿意完成任务的用户都为其分配
"""
# 参数设置
import copy

import xlrd

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
    user=copy.deepcopy(init_user)
    for t in range (T):
        avery_RB=RB/T
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
            region[i][1] = s_[i]
            if (region[i][0] - region[i][1] > 0):
                region[i][2] = region[i][0] - region[i][1]
            else:
                region[i][2] = 0
            regionnn.append(region[i][1])
        print("regionnn", sum(regionnn), regionnn)