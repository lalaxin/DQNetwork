"""
从regionminute10_15.xlsx十分钟间隔数据中读取，用其预测15号8-10点的数据，将其存到region_1中
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import copy
from torch import nn
from torch.autograd import Variable

timestep_train=12
result_train=12
regionnum=100
# 设置X,Y数据集，以look_back2为准，取第一个和第二个为数组，形成data_X,取第三个作为预测值，形成data_Y,完成训练集的提取
def create_dataset(dataset, look_back=timestep_train):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x
userpredict=[[0 for i in range(regionnum)]for i in range(result_train)]

for t in range (100):
    data_csv=pd.read_excel('./regionminute10_15.xlsx',usecols=[t])
    # 数据预处理
    data_csv = data_csv.dropna()  # 滤除缺失数据
    dataset = data_csv.values   # 获得csv的值
    dataclose=copy.deepcopy(dataset)
    dataset = dataset.astype('float32')
    max_value = np.max(dataset)  # 获得最大值
    min_value = np.min(dataset)  # 获得最小值
    scalar = max_value - min_value  # 获得间隔数量
    print("scaler",scalar)
    dataset = list(map(lambda x: x / scalar, dataset)) # 归一化
    # 创建好输入输出[输入输出对，表示一组训练数据]
    data_X, data_Y = create_dataset(dataset)
    # 划分训练集和测试集，测试集为最后24个数据（一整天）前14天数据为训练数据
    train_size = int(len(data_X) -result_train)
    test_size = len(data_X)-train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    # 设置LSTM能识别的数据类型，形成tran_X的一维两个参数的数组，train_Y的一维一个参数的数组。并转化为tensor类型
    train_X = train_X.reshape(-1, 1, timestep_train)
    train_Y = train_Y.reshape(-1, 1, 1)
    # test_X = test_X.reshape(-1, 1, 2)
    train_x = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_Y)
    # test_x = torch.from_numpy(test_X)
    model = lstm(timestep_train, 32, 1, 4)
    # 设置交叉熵损失函数和自适应梯度下降算法
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # 开始训练
    for e in range(1000):
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        # 前向传播
        out = model(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e + 1) % 100 == 0:  # 每 100 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

    # 模型预测
    model = model.eval() # 转换成测试模式

    data_X = data_X.reshape(-1, 1, timestep_train)
    data_X = torch.from_numpy(data_X)
    var_data = Variable(data_X)
    pred_test = model(var_data) # 测试集的预测结果
    # 改变输出的格式
    pred_test = pred_test.view(-1).data.numpy()
    pred_test=np.concatenate((np.zeros(result_train),pred_test))
    assert len(pred_test)==len(dataclose)

    # plt.plot(pred_test, 'r', linewidth=1,label='prediction')
    # # plt.plot(train_Y,'g',label='train_Y')
    # plt.plot(dataset, 'b',linewidth=1, label='real')
    # plt.plot((train_size,train_size),(0,1),'g--')
    # plt.legend(loc='best')
    # plt.show()

    # 反归一化(同时要处理数据为整数)
    pred_test=list(map(lambda x: x*scalar, pred_test))
    dataset=list(map(lambda x: x *scalar, dataset))
    pred_result=pred_test[-result_train:]
    data_result=dataclose[-result_train:]
    print("后24pred",pred_result)
    print("后24_dataset",data_result)
    for i in range (len(pred_result)):
        if(pred_result[i]<0):
            pred_result[i]=0
        else:
            pred_result[i]=round(pred_result[i])
        userpredict[i][t]=pred_result[i]

    print("处理后的pred",pred_result)
# print(dataset)
# print(userpredict)
# # 存入excel文件
init_region = np.array(userpredict)
data = pd.DataFrame(init_region)
# 写入excel文件
writer = pd.ExcelWriter("./region_1.xlsx")
data.to_excel(writer, 'sheet2', float_format='%.5f')
writer.save()
writer.close()
    # 画图
    # 画出实际结果和预测的结果
