import xlrd,xlwt
import datetime
import random

def getusers():
    # 打开需要操作的excel,由于地图左下角区域没有用户，所以不处理那些区域（原点定为（8000，7000））
    excel = xlrd.open_workbook("./data2.xlsx")

    # 获取excel的sheet表
    sheet = excel.sheet_by_name("Sheet1")
    starttime = datetime.datetime(2016, 8, 1, 0, 00, 00)
    endtime = datetime.datetime(2016, 8, 1, 1, 00, 00)

    #统计截至日期
    deadday = datetime.datetime(2016,8,16,0,00,00)

    onetime = []
    one = []
    user = []
    print("-------start----")
    print("start", starttime)
    print("end", endtime)

    for i in range(1, sheet.nrows):
        row_i = sheet.row_values(i)
        d = xlrd.xldate_as_datetime(row_i[0], 0)
        # print(i,": ",row_i)
        onetime = []
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
            # print("user:", user)
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
                # print("onetime:", onetime)
                one.append(onetime)
                # print(len(one))
                # print("one:", one)
        if starttime >= deadday:
            d = xlrd.xldate_as_datetime(row_i[0], 0)
            print("d:",d)
            break
    # print(user)
    return user

if __name__  == '__main__':
    u = getusers()