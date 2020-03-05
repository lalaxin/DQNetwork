import numpy as np
from helloword import cellnum
from helloword import Region
from helloword import User
from helloword import usernum
import math


# km最优匹配
# 车匹配区域，左边表示车，相当于用户数，右边表示区域内缺车数（即缺的车数总和）
# 生成的数组（用户横坐标，用户纵坐标，区域缺车横坐标，区域缺车纵坐标，权值（两者之间的距离1/d））
sumlackbike=0
for i in range(cellnum):
    if (Region[i][2] > 0):
        sumlackbike += Region[i][2]
def build_graph(Region,User):
    x=usernum
    adj=[[0 for i in range(sumlackbike)] for i in range(x)]
    dmax=100000
    for i in range (x):
        a=0
        for j in range (cellnum):
            if(Region[j][2]>0):
                for k in range (Region[j][2]):
                    a+=1
                    d=math.sqrt(math.pow((User[i][2]-Region[j][3]),2)+math.pow((User[i][3]-Region[j][4]),2))
                    if(d == 0):
                        adj[i][a-1] = [User[i][2], User[i][3], Region[j][3], Region[j][4], dmax]
                    else:
                         adj[i][a-1] = [User[i][2], User[i][3], Region[j][3], Region[j][4], dmax/d]
    return adj
adj=build_graph(Region,User)
print(adj)
adj_matrix=[[0 for i in range(sumlackbike)]for i in range(usernum)]
for i in range (usernum):
    for j in range(sumlackbike):
        adj_matrix[i][j]=adj[i][j][4]
# print(adj_matrix)
#转置矩阵
def transpose(matrix):
    new_matrix=[]
    for i in range(len(matrix[0])):
        matrix1=[]
        for j in range(len(matrix)):
            matrix1.append(matrix[j][i])
        new_matrix.append(matrix1)
    return new_matrix
leftadj_matrix=transpose(adj_matrix)
if(sumlackbike>=usernum):
    # 初始化顶标，车匹配区域，左边表示车，相当于用户数，右边表示区域内缺车数（即缺的车数总和）
    label_left=np.max(adj_matrix,axis=1)  #取大函数，axis=1表示横向
    label_right=np.zeros(sumlackbike) #匹配时又边总是较大
    # 初始化匹配结果
    match_right=np.zeros(sumlackbike)  #按缺车的数量匹配结果，np.nan表示0
    # 初始化辅助变量
    visit_left=np.zeros(usernum, dtype=bool)    #visit_left表示左边的用户是否匹配
    for i in range(usernum):visit_left[i]=False
    visit_right=np.zeros(sumlackbike,dtype=bool)   #visit_right表示右边的缺车是否匹配
    for i in range(sumlackbike):visit_right[i]=False

    slack_right=np.ones(sumlackbike)*np.inf    #np.inf表示最大的正数,记录每个汉子如果能被妹子倾心最少还需要多少期望值
else:
    # 初始化顶标，车匹配区域，左边表示车，相当于用户数，右边表示区域内缺车数（即缺的车数总和）
    label_left = np.max(leftadj_matrix, axis=1)  # 取大函数，axis=1表示横向
    label_right = np.zeros(usernum)  #
    # 初始化匹配结果
    match_right = np.zeros(usernum)  # 按缺车的数量匹配结果，np.nan表示0
    # 初始化辅助变量
    visit_left = np.zeros(sumlackbike)  # visit_left表示左边的用户是否匹配
    # visit_left.fill(False)
    # for i in range(sumlackbike):visit_left[i]=False
    visit_right = np.zeros(usernum)   # visit_right表示右边的缺车是否匹配
    # for i in range(usernum):visit_right[i]=False
    slack_right = np.ones(usernum) * np.inf  # np.inf表示最大的正数,记录每个汉子如果能被妹子倾心最少还需要多少期望值

print("usernum",usernum,"sumlackbike",sumlackbike)
print("label_left",label_left)
print("label_right",label_right)
print("match_right",match_right)
print("visit_left",visit_left)
print("visit_right",visit_right)
print("slack_right",slack_right)


# 寻找增广路，深度优先
def find_rightpath(i):
    visit_left[i]=1.
    for j,match_weight in enumerate(adj_matrix[i]):  #enumerate是内置函数，输出的是（计数值，值）
        if visit_right[j]:continue  #已被匹配（解决递归中的冲突）（跳过当前循环的剩余语句，进行下一句循环）
        gap=label_left[i]+label_right[j]-match_weight #（匹配时两方的期望和要等于两人之间的好感度即match_right）计算代沟
        if gap==0 :
            # 找到可行匹配
            visit_right[j]=True
            if np.isnan(match_right[j])or find_rightpath(match_right[j]):  #j未被匹配，或虽然j已被匹配，但是j的已匹配对象有其他可选
                match_right[j]=i
                return  True
        else:
            # 计算变为可行匹配需要的顶标改变量
            if slack_right[j]>gap:slack_right[j]=gap #slack理解为该缺车为了被用户匹配，还需要多少期望值，来最小化它
    return False

def find_leftpath(i):
    visit_left[i] = 1
    for j, match_weight in enumerate(leftadj_matrix[i]):  # enumerate是内置函数，输出的是（计数值，值）
        if visit_right[j]: continue  # 已被匹配（解决递归中的冲突）（跳过当前循环的剩余语句，进行下一句循环）
        gap = label_left[i] + label_right[j] - match_weight  # （匹配时两方的期望和要等于两人之间的好感度即match_right）计算代沟
        if gap == 0:
            # 找到可行匹配
            visit_right[j] = True
            if np.isnan(match_right[j]) or find_leftpath(match_right[j]):  # j未被匹配，或虽然j已被匹配，但是j的已匹配对象有其他可选
                match_right[j] = i
                return True
        else:
            # 计算变为可行匹配需要的顶标改变量
            if slack_right[j] > gap: slack_right[j] = gap  # slack理解为该缺车为了被用户匹配，还需要多少期望值，来最小化它
    return False

# km主函数
def KM():
    if(usernum<=sumlackbike): #每个用户都可以被匹配的情况下
        for i in range(usernum):
            slack_right=np.ones(sumlackbike)*np.inf #最小化它，初始化给它最大值
            while True:  #为每个用户找到缺车区域，若找不到就降低期望值，直到找到为止(由于两边数量不一样，不一定给每个用户都能找到匹配# )
                visit_left=np.ones(usernum)*False #创建空数组
                visit_right=np.ones(sumlackbike)*False
                if find_rightpath(i): break  #如果找到匹配项就退出，未找到则降低期望值
                d=np.inf  #最小可降低的期望值，其表示最大的正数
                for j,slack in enumerate(slack_right):
                    if not visit_right[j] and slack < d:
                        d=slack
                for k in range (usernum):
                    if visit_left[k] : label_left[k] -= d #所有访问过的用户降低期望值
                for n in range (sumlackbike):
                    if visit_right[n] : label_right[n] += d #所有访问过的男生增加期望值
                    else: slack_right[j] -= d #没有访问过的缺车，因为用户的期望降低，距离匹配成功又进一步

    else:  #用户数大于缺车数时，反过来，用户来匹配缺车数
        match_right = np.zeros(usernum)  # 按缺车的数量匹配结
        # 初始化辅助变量
        for i in range(sumlackbike):
            slack_right=np.ones(usernum)*np.inf #最小化它，初始化给它最大值
            while True:  #
                visit_left=np.zeros(sumlackbike) #创建空数组
                visit_right=np.zeros(usernum)
                print(visit_left)
                if find_leftpath(i): break  #如果找到匹配项就退出，未找到则降低期望值
                d=np.inf  #最小可降低的期望值，其表示最大的正数
                for j,slack in enumerate(slack_right):
                    if not visit_right[j] and slack < d:
                        d=slack
                for k in range (sumlackbike):
                    if visit_left[k] : label_left[k] -= d #所有访问过的女生降低期望值
                for n in range (usernum):
                    if visit_right[n] : label_right[n] += d #所有访问过的男生增加期望值
                    else:
                        slack_right[j] -= d #没有访问过的缺车，因为用户的期望降低，距离匹配成功又进一步

    # 求出所有的匹配的好感度之和
    res=0
    for j in range(sumlackbike):
        if match_right[j] >= 0 and match_right[j]<sumlackbike:
            res += adj_matrix[match_right[j]][j]
    return res

res=KM()
print(res)