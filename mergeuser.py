"""
合并五天的数据为一天，从user.xlsx获取用户信息并返回
"""
import json
import copy
import xlrd


def mergeuser():
    # 打开五个user分别存与user1,user2,user3,user4,user5中
    excel = xlrd.open_workbook("D:/Python/python_code/DQNetwork/user.xlsx")

    sheet1 = excel.sheet_by_name("1081710")
    user1=sheet1.col_values(0)
    for i in range (len(user1)):
        user1[i]=json.loads(user1[i])
    print("type(user1[0]), user1",type(user1[0]), user1)

    sheet2 = excel.sheet_by_name("1082710")
    user2 = sheet2.col_values(0)
    for i in range(len(user2)):
        user2[i] = json.loads(user2[i])
    print("type(user2[0]), user2",type(user2[0]), user2)

    sheet3 = excel.sheet_by_name("1083710")
    user3 = sheet3.col_values(0)
    for i in range(len(user3)):
        user3[i] = json.loads(user3[i])
    print("type(user3[0]), user3",type(user3[0]), user3)

    sheet4 = excel.sheet_by_name("1084710")
    user4 = sheet4.col_values(0)
    for i in range(len(user4)):
        user4[i] = json.loads(user4[i])
    print("type(user4[0]), user4",type(user4[0]), user4)

    sheet5 = excel.sheet_by_name("1085710")
    user5 = sheet5.col_values(0)
    for i in range(len(user5)):
        user5[i] = json.loads(user5[i])
    print("type(user5[0]), user5",type(user5[0]), user5)

    user=copy.deepcopy(user1)
    for i in range (len(user)):
        if(user2[i]!=[]):
            for j in range (len(user2[i])):
                user[i].append(user2[i][j])
        if (user3[i] != []):
            for j in range(len(user3[i])):
                user[i].append(user3[i][j])
        if (user4[i] != []):
            for j in range(len(user4[i])):
                user[i].append(user4[i][j])
        if (user5[i] != []):
            for j in range(len(user5[i])):
                user[i].append(user5[i][j])
        # user[i].append(user3[i])
        #
        # user[i].append(user4[i])
        # user[i].append(user5[i])
        print("user",len(user[i]),user[i])
    print(user)
    return user


mergeuser()