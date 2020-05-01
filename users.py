import xlrd,xlwt
import datetime
import random

class getuser():
    def getusers(self):
        # 打开需要操作的excel,由于地图左下角区域没有用户，所以不处理那些区域（原点定为（8000，7000））
        excel = xlrd.open_workbook("./data.xlsx")

        # 获取excel的sheet表
        sheet = excel.sheet_by_name("Sheet1")
        starttime = datetime.datetime(2016,8,1,0,00,00)
        endtime = datetime.datetime(2016,8,1,1,00,00)

        onetime = []
        one = []
        user = []
        print("-------start----")
        print("start", starttime)
        print("end", endtime)

        for i in range(1,sheet.nrows):
            row_i = sheet.row_values(i)
            d = xlrd.xldate_as_datetime(row_i[0],0)
            # print(i,": ",row_i)
            onetime = []
            if d > starttime and d <= endtime:
                onetime.append(row_i[2])
                onetime.append(row_i[4])
                onetime.append(row_i[7])
                onetime.append(row_i[9])
                onetime.append(random.uniform(0,200))
                onetime.append(-1)
                onetime.append(-1)
                one.append(onetime)
                # print("d",d)
                # print("onetime:",onetime)
            else:
                print("one:",one)
                print(len(one))
                user.append(one)
                print("user:",user)
                one = []
                if d >= endtime:
                    starttime = endtime
                    print("start", starttime)
                    endtime = endtime + datetime.timedelta(hours=1)
                    print("end", endtime)
                    onetime.append(row_i[2])
                    onetime.append(row_i[4])
                    onetime.append(row_i[7])
                    onetime.append(row_i[9])
                    onetime.append(random.uniform(0, 200))
                    onetime.append(-1)
                    onetime.append(-1)
                    print("onetime:", onetime)
                    one.append(onetime)
                    print(len(one))
                    print("one:", one)

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
        print(b)
        return region

users=getuser().getusers()
print(len(users))
cell=10
celllength=300
T=23
init_region=[0 for i in range (cell*cell)]
print("initregion",init_region)
for t in range(T):
    print("user[t]",len(users[t]),users[t])
    init_region=getuser().usertoregion(users[t],init_region,cell,celllength)
    print(init_region)
    for i in range(cell*cell):
        init_region[i]=0



