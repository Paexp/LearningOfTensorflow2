import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

x_data = datasets.load_iris().data	#返回iris数据集所有输入特征
y_data = datasets.load_iris().target

np.random.seed(116)	#使用相同的seed，使输入特征/标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1
train_loss_result = []
test_acc = []
epoch = 500
loss_all = 0

for epoch in range(epoch):	# 数据集级别迭代
    for step, (x_train, y_train) in enumerate(train_db):	# batch级别迭代
        with tf.GradientTape() as tape:	# 记录梯度信息
            y=tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()
        grads = tape.gradient(loss, [w1, b1])

        w1.assign_sub(lr * grads[0])	# 参数自更新
        b1.assign_sub(lr * grads[1])

    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_result.append(loss_all / 4)
    loss_all = 0

    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1  # y为预测结果
        y = tf.nn.softmax(y)  # y符合概率分布
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype)  # 调整数据类型与标签一致
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)  # 将每个batch的correct数加起来
        total_correct += int(correct)  # 将所有batch中的correct数加起来
        total_number += x_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("---------------------------------")

plt.title('Loss Function Curve')	# 图片标题
plt.xlabel('Epoch')	# x轴名称
plt.ylabel('Acc') # y轴名称
plt.plot(train_loss_result, label = '$Loss$')	# 逐点画出test_acc值并连线
plt.legend()
plt.show()

plt.title('Acc Curve')	# 图片标题
plt.xlabel('Epoch')	# x轴名称
plt.ylabel('Acc') # y轴名称
plt.plot(test_acc, label = '$Accuracy$')	# 逐点画出test_acc值并连线
plt.legend()
plt.show()