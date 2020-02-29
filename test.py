"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# 设置随机数seed
np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            # 神经网络要输出多少个action的值（每个action的Q-value）
            n_actions,
            # 接收多少个observation（用feature来预测action的值）
            n_features,
            # 神经网络的学习效率
            learning_rate=0.01,
            # 强化学习的折扣因子
            reward_decay=0.9,
            # 随机-贪婪策略
            e_greedy=0.9,
            # 隔多少步将target_Q的参数变成最新的参数
            replace_target_iter=300,
            # experience replay记忆库中的容量有多少（若为200.就可以记忆下两百条（observation,action,reword,observation）这样的数据）
            memory_size=500,
            # 神经网络学习中会用到，加入这一项之后，模型的训练收敛速度会更快
            batch_size=32,
            # 不断地缩小随机的范围（随机-贪婪策略）
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions  # action num
        self.n_features = n_features  # state num
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 折扣因子
        self.epsilon_max = e_greedy  # 贪婪决策概率
        self.replace_target_iter = replace_target_iter  # target和eval的参数更新间隔步
        self.memory_size = memory_size  # 记忆库大小
        self.batch_size = batch_size  # 批量大小
        self.epsilon_increment = e_greedy_increment  # greedy变化
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0  # 计步器，记录学习了多少步，epsilon会根据counter不断地提高

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size,
                                n_features * 2 + 2))  # 初始化记忆库。第一个参数就是记忆库的大小，第二个参数是一条数据中（s,a,r,s）中的个数总和

        # consist of [target_net, evaluate_net]
        # 用这个函数来创建神经网络
        self._build_net()



        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("E:/Code/logs", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # 记录每一步的误差
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        ### eva-Network的输入s 类比 图片输入 是一个 列矩阵 形式
        ### tf.placeholder(s类型为32位浮点,[行向量],标签名字为s)
        ### 这里的 None 与 minibatch 的 size 相同
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 输入 s
        ### 用于计算loss的Q现实值。
        ### 并不是本结构的输出，但需要扩展作用域
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions],name='Q_target')  # 输入 用于计算 loss function

        ### tf.variable_scope 创建eval_net的相关变量
        ### c_names是数组，中间层神经元个数10，正态分布随机数作为权值w，常量0.1作为偏置b
        with tf.variable_scope('eval_net'):
            # n_l1表示第一层有多少个神经元，c_name用于tensflow中，w和b是建立的每一层的默认参数（神经网络中的参数）
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            # 第一层，w1 b1 以及输出 l1 = 激活函数（ input s * w1 + b1 ）
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                ## get_variable('name',[shape],<initializer>)
                ## 权值矩阵大小为 features行 * n_l1列
                ## 偏置矩阵大小为 1行 * n_l1列
                w1 = tf.get_variable('w1', [self.n_features, n_l1],initializer=w_initializer,collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1],initializer=b_initializer,collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            # 上一层输出对应输入，这一层输出对应 action，输出层
            # 第二层，w2 b2 以及输出 q_eval = l1 * w2 + b2
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                ## 权值矩阵大小为 n_l1行 * actions列
                ## 偏置矩阵大小为 1行 * actions列
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions],
                    initializer=w_initializer,
                    collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions],
                    initializer=b_initializer,
                    collections=c_names)
                # 有多少action就输出多少个值
                self.q_eval = tf.matmul(l1, w2) + b2
        ### 定义evl_Network的更新依据 loss函数
        ### reduce_mean默认对所有数求平均，squared_difference最小二乘
        ### 最小二乘的结果是1*features，平均之后是1*1
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval))
        ### 定义evl_Network训练方法
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(
                self.loss)

        # ------------------ build target_net ------------------
        ### target net是旧版的evl_Net。需要给出 q_target 的计算依据 q_next
        ### 输入为 s_ 及下一时刻的state
        ### 网络结构与evl_Net相同
        # s_表示下一个state
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='s_')  # 输入 下一时刻状态
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            # 结构与evl_Net相同
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1],
                    initializer=w_initializer,
                    collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1],
                    initializer=b_initializer,
                    collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions],
                    initializer=w_initializer,
                    collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions],
                    initializer=b_initializer,
                    collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    # 定义样本池
    def store_transition(self, s, a, r, s_):
        ### hasattr(object, name) 检查对象是否包含对应属性
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        ### np.hastack 水平合并 数组
        transition = np.hstack((s, [a, r], s_))
        ### 样本池按列存放样本，每个样本为一个行向量
        ### 循环存储每个行动样本，index用于循环覆盖的起始行数
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    # 把observation放入神经网络，输出每一个action的Q值，选取最大的
    def choose_action(self, observation):
        ### e-greedy 策略实现
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            ### feed 与 placeholder 搭配使用 ，传递值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            ### np.argmax 取出最大值的索引
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        # 把e_params的参数赋值给t_params
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        # zip() 函数用于将可迭代对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象。
        self.replace_target_op = [
            tf.assign(t, e) for t, e in zip(t_params, e_params)
        ]

    def learn(self):
        # (看有没有到换参数的时候)如需更新target参数，则更新
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 随机抽取minibatch
        ## 当样本池未满的时候，依照计数器值随机选取，否则按照池大小选取。（当记忆库中没有这么多用于存与batch_memory中的内容时，抽取已经存下来的记忆放在里面）
        if self.memory_counter > self.memory_size:
            ### numpy.random.choice(a, size=None, replace=True, p=None)
            ### 从 a 中以概率 p 随机抽取 size 个。replace表示是否放回。
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        # 选择出需要训练的样本 batch_memory（随机抽取记忆库中的样本来作为batch_memory）
        batch_memory = self.memory[sample_index, :]
        # q_next是Q_target神经网络输出的值，q_eval是Q_估计神经网络输出的值
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
            ### batch_memory格式为行向量，s(features),a,r,s_(feature)
            ### 分别取出 s 和 s_ 作为网络输入
            ### np矩阵切片 a[:,a:b:c] 从a到b列步长为c切片
            ### 下方对batch_memory切片取出s_ 和 s
                self.s_: batch_memory[:, -self.n_features:],  # 旧参数
                self.s: batch_memory[:, :self.n_features],  # 新参数
            })

        # 加上奖励和折扣因子作为q实际
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        ### numpy 数组的四个方法：ndim返回维度数，shape返回维度数组，dtype返回元素类型，astype类型转换
        ### 对 batch_memroy 切片，取出 action / reward
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        # 反向传递针对的是Q估计（q_eval）的action
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(
            q_next, axis=1)
        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # 训练Eval_Net
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={
                                         self.s:
                                             batch_memory[:, :self.n_features],
                                         self.q_target:
                                             q_target
                                     })
        ### 记录每一步更新的 cost 用于训练后画图分析
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # 观测误差曲线
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

# 更新步骤
def run():
    # 记录当前走到那一步
    step = 0
    for episode in range(300):
        # 初始化observation
        observation=env.reset()

        while True:
            env.render()
            # 首先根据当前observation选取一个动作
            action=RL.choose_action(observation)

            # 施加改动作在环境中，得到下一个observation_,以及reward，done表示是否终结
            observation_,reward,done=env.step(action)
            # 存储记忆的步骤
            RL.store_transition(observation,action,reward,observation_)
            # s首先需要存储记忆，记忆库中有一些东西之后才能学习（前200步都是在存储记忆,大于200之后每5步学习一次）
            if(step>200)and(step%5==0):
                RL.learn()
            # 状态更新
            observation=observation_
            # 若终结则退出循坏
            if done:
                break
            step+=1



if __name__ == '__main__':
    DQN = DeepQNetwork(3, 4, output_graph=True)

