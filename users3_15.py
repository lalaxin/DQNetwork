"""
data3-sheet2   8:00-10:00  十分钟时间间隔
"""
import xlrd,xlwt
import datetime
import random
import pandas as pd
import numpy as np
from chinese_calendar import is_workday

class getuser_15():
    def getusers(self):
        # 打开需要操作的excel,由于地图左下角区域没有用户，所以不处理那些区域（原点定为（8000，7000））
        excel = xlrd.open_workbook("./data3.xlsx")

        # 获取excel的sheet表
        sheet = excel.sheet_by_name("Sheet2")
        starttime = datetime.datetime(2016, 8, 1, 0, 00, 00)
        endtime = datetime.datetime(2016, 8, 1, 0, 10, 00)

        #统计截至日期
        deadday = datetime.datetime(2016,8,16,0,00,00)

        onetime = []
        one = []
        user = []
        print("-------start----")
        print("start", starttime)
        print("end", endtime)

        day=0

        for i in range(1, sheet.nrows):
            row_i = sheet.row_values(i)
            d = xlrd.xldate_as_datetime(row_i[0], 0)
            # print(i,": ",row_i)
            onetime = []

            if starttime>=deadday:
                d=xlrd.xldate_as_datetime(row_i[0],0)
                print("d:",d)
                break

            if d > starttime and d <= endtime:
                onetime.append(row_i[2])
                onetime.append(row_i[4])
                onetime.append(row_i[7])
                onetime.append(row_i[9])
                onetime.append(random.uniform(0, 200))
                onetime.append(-1)
                onetime.append(-1)
                one.append(onetime)
                # print("d",d)
                # print("onetime:",onetime)
            else:
                # print("one:", one)
                # print(len(one))
                user.append(one)
                day+=1
                print("day",day)

                print("多余工作日",d)
                # print("user:", user)
                one = []
                if d >= endtime:
                    starttime = endtime
                    # print("start", starttime)
                    endtime = endtime + datetime.timedelta(minutes=10)
                    # print("end", endtime)
                    onetime.append(row_i[2])
                    onetime.append(row_i[4])
                    onetime.append(row_i[7])
                    onetime.append(row_i[9])
                    onetime.append(random.uniform(0, 500))   #用户的最大步行距离
                    onetime.append(-1)
                    onetime.append(-1)
                    # print("onetime:", onetime)
                    one.append(onetime)
                    # print(len(one))
                    # print("one:", one)
            if starttime >= deadday:
                d = xlrd.xldate_as_datetime(row_i[0], 0)
                print("d:",d)
                break
            # print("工作日",d)

        print("user长度",len(user))
        return user

    def usertoregion(self,user,region,cell,celllength):
        b=0
        for i in range(len(user)):
            if (user[i][0] == cell * celllength and user[i][1] == cell * celllength):
                a = int(cell*cell - 1)
            elif (user[i][0] == cell * celllength):
                a = int(user[i][1] / celllength) * cell + int(user[i][0] / celllength) - 1
            elif (user[i][1] == cell * celllength):
                a = int(user[i][1] / celllength) * cell + int(user[i][0] / celllength) - cell
            else:
                a = int(user[i][1] / celllength) * cell + int(user[i][0] / celllength)
            # print(a)
            if (a < cell*cell):
                region[a] += 1

            if(b<a):
                b=a
        # print(b)
        return region

def getregion():
    users=getuser_15().getusers()
    # print(len(users))
    cell=10
    celllength=300
    T=24*11
    init_region=[[0 for i in range (cell*cell)]for t in range(T)]
    # print("initregion",init_region)
    for t in range(T):
        # print("user[t]",len(users[t]),users[t])
        init_region[t]=getuser_15().usertoregion(users[t],init_region[t],cell,celllength)
        # print("init_region[t]",t,init_region[t])
    # #存入excel文件
    # init_region=np.array(init_region)
    # data=pd.DataFrame(init_region)
    #     # 写入excel文件
    # writer=pd.ExcelWriter("./regionminute10_15.xlsx")
    # data.to_excel(writer,'sheet1',float_format='%.5f')
    # writer.save()
    # writer.close()

        # for i in range(cell*cell):
        #     init_region[i]=0


getregion()
