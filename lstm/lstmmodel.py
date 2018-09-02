# coding=utf-8
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from getdata.insertdb import selectarray
from others.wordtopinyin import change
# 隐层数量
def get_CellSize():
    CELL_SIZE = 10
    return CELL_SIZE
#输入层
def get_InputSize():
    INPUT_SIZE = 6
    return INPUT_SIZE
#输出层
def get_OutputSize():
    OUTPUT_SIZE = 6
    return OUTPUT_SIZE
# 学习率
def get_LR():
    LR = 0.0006
    return LR
#时间步
def get_TimeStep():
    TIME_STEP = 20
    return TIME_STEP
#批训练
def get_BatchSize():
    BATCH_SIZE = 60
    return BATCH_SIZE
#循环层数目
def get_MultiRnnCell():
    number_MultiRnnCell = 10
    return number_MultiRnnCell
#创建目录
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
#预测数据处理
def predict_handle(data_predict):
    #温度
    if data_predict[0] < data_predict[1]:
        temp = data_predict[1]
        data_predict[1] = data_predict[0]
        data_predict[0] = temp
    #状态
    data_predict[2] = round(data_predict[2])
    if data_predict[2]<1:
        data_predict[2] = 1
    elif data_predict[2]>13:
        data_predict[2] = 13
    if data_predict[3]>1:
        data_predict[3] = 1
    elif data_predict[3]<0:
        data_predict[3] = 0
    if data_predict[4]>1:
        data_predict[4] = 1
    elif data_predict[4]<0:
        data_predict[4] = 0
    if data_predict[5]<0:
        data_predict[5] = 1e-7
    return data_predict
# ——————————————————导入数据——————————————————————
#表获取
def load_data(province,city):
    sql = "SELECT wdate,hightemper,lowtemper,iconstate,humidity,cloudCover,windSpeed \
    FROM weatherdata WHERE province = '%s' AND city = '%s' ORDER BY wdate ASC"%(province,city)
    s = selectarray(sql)
    table=pd.DataFrame(s,columns=["wdate","hightemper","lowtemper",
                                  "iconstate","humidity","cloudCover","windSpeed"])
    return table
#数据获取
def get_data(table):
    data = np.array([table["hightemper"], table["lowtemper"],
                     table["iconstate"], table["humidity"],
                     table["cloudCover"],table["windSpeed"]],dtype=np.float32)
    data = data.T
    return data
#数据标准化
def normalized_data(data):
    data_min = np.min(data, 0)
    data_max = np.max(data, 0)
    numerator = data - data_min
    denominator = data_max - data_min
    # 加入噪声
    normalize_data = numerator / (denominator + 1e-7)
    return normalize_data,data_max,data_min
    '''
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normal_data = (data - mean) / std
    return normal_data
    '''
# 获取训练集
def get_train_data(normalized_train_data):
    batch_size = get_BatchSize()
    time_step = get_TimeStep()
    batch_index = []
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, : ]
        y = normalized_train_data[i + time_step, : ]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y
# 获取预测数据
def get_test_data(normalized_test_data):
    time_step = get_TimeStep()
    predict_x = normalized_test_data[len(normalized_test_data)-time_step:len(normalized_test_data), : ]
    return predict_x
# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重
def get_weight():
    weights = {
        'in': tf.Variable(tf.random_normal([get_InputSize(), get_CellSize()])),
        'out': tf.Variable(tf.random_normal([get_CellSize(), get_OutputSize()]))
    }
    return weights
#输入层、输出层偏置
def get_biases():
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[get_CellSize(), ])),
        'out': tf.Variable(tf.constant(0.1, shape=[get_OutputSize(), ]))
    }
    return biases
# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    INPUT_SIZE = get_InputSize()
    CELL_SIZE = get_CellSize()
    OUTPUT_SIZE = get_OutputSize()
    number_MultiRnnCell = get_MultiRnnCell()
    weights = get_weight()
    biases = get_biases()
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, INPUT_SIZE])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, CELL_SIZE])  # 将tensor转成3维，作为lstm cell的输入
    lstmcell = tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE)
    dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstmcell,output_keep_prob=0.98)
    stack_lstm = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm]*number_MultiRnnCell)
    init_state = stack_lstm.zero_state(batch_size, dtype=tf.float32)
    with tf.variable_scope("output",reuse=tf.AUTO_REUSE):
        output_rnn, final_states = tf.nn.dynamic_rnn(stack_lstm, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1,CELL_SIZE])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    pred = tf.reshape(pred, [-1, time_step, OUTPUT_SIZE])
    pred_list = pred[:,time_step-1,:]
    pred = pred_list
    return pred, final_states
# ————————————————训练模型————————————————————
def train_lstm(batch_index,train_x,train_y,save_path):
    time_step = get_TimeStep()
    INPUT_SIZE = get_InputSize()
    OUTPUT_SIZE = get_OutputSize()
    LR = get_LR()
    X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    with tf.variable_scope('train_op',reuse=tf.AUTO_REUSE):
        train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss],
                                    feed_dict={X: (train_x[batch_index[step]:batch_index[step + 1]]),
                                               Y: (train_y[batch_index[step]:batch_index[step + 1]])})
            print("迭代次数:", i, " 损失函数值:", loss_)
        print("保存模型: ", saver.save(sess, save_path))
        print("训练完成")
# ————————————————预测模型————————————————————
def prediction(data_min,data_max,predict_x,save_path):
    time_step = get_TimeStep()
    INPUT_SIZE = get_InputSize()
    X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])
    with tf.variable_scope("sec_lstm", reuse=True):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, module_file)
        data_predict = []
        for i in range(7):
            prob = sess.run(pred, feed_dict={X: [predict_x]})
            predict = prob.reshape((-1))
            predict_x = np.vstack((predict_x[1:], predict))
            predict = np.array(predict) * (data_max - data_min + 1e-7) + data_min
            predict = predict_handle(predict)
            data_predict.append(predict)
        print(np.array(data_predict))
#训练数据
def get_train(province,city):
    mkpath = change(province) + "\\" + change(city) + "\\"
    mkdir(mkpath)
    save_path = change(province) + "\\" + change(city) + "\\model.ckpt"
    table = load_data(province,city)
    data = get_data(table)
    normalize_data,data_max,data_min = normalized_data(data)
    batch_index,train_x,train_y = get_train_data(normalize_data)
    train_lstm(batch_index,train_x,train_y,save_path)
    return
#预测数据
def get_predict(province,city):
    np.set_printoptions(suppress=True)
    load_model = change(province) + "\\" + change(city)+"\\"
    table = load_data(province, city)
    data = get_data(table)
    normalize_data, data_max, data_min = normalized_data(data)
    predict_data = get_test_data(normalize_data)
    prediction(data_min, data_max, predict_data, load_model)
    return
