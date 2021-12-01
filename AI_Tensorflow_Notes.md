人工智能实践：Tensorflow笔记



第一讲 

本讲目标：学会神经网络计算过程，使用基于TF2原生代码搭建你的第一个神经网络训练模型

本讲内容：

- 当今人工智能主流方向——连接主义


- 前向传播


- 损失函数


- 梯度下降


- 学习率


- 反向传播更新参数


- Tensorflow 2常用函数



1. 人工智能三学派

   人工智能：让机器具备人的思维和意识

   人工智能三学派：

   行为主义：基于控制论，构建感知-动作控制系统。

   符号主义：基于算数逻辑表达式，求解问题时先把问题描述为表达式，再求解表达式。

   连接主义：仿生学，模仿神经元连接关系。（**仿脑神经元连接**，实现感性思维，如神经网络）

   用计算机仿出神经网络连接关系：

   （1）准备数据：采集大量“特征/标签”数据

   （2）搭建网络：搭建神经网络结构

   （3）优化参数：训练网络获取最佳参数（反向传播）

   （4）应用网络：将网络保存为模型，输入新数据，输出分类或预测结果（前向传播）

2. 张量(Tensor)：多维数组（列表） 阶：张量的维数

   (1) 张量可以表示0阶到n阶的数组

   | 维数 | 阶   | 名字        | 例子                                |
   | ---- | ---- | ----------- | ----------------------------------- |
   | 0-D  | 0    | 标量 scalar | s = 1 2 3                           |
   | 1-D  | 1    | 向量 vector | v = [1, 2, 3]                       |
   | 2-D  | 2    | 矩阵 matrix | m=[[1, 2, 3], [4, 5, 6], [7, 8, 9]] |
   | n-D  | n    | 张量 tensor | t=[[[...]]]                         |

   (2) 数据类型

   - tf.int, tf.float ......

     tf.int 32, tf.float 32, tf.float 64
   
   - tf.bool
   
     tf.constant([True, False])
   
   - tf.string
   
     tf.constant("Hello, world!")
   
   （3）如何创建一个Tensor
   
   - 创建一个张量
   
     tf.sonstant(张量内容，dtype=数据类型（可选）)
     
   - 将numpy的数据类型转换为Tensor数据类型
   
     tf.convert_to_tensor(数据名，dtype=数据类型（可选）)
     
   - 创建全为0的张量
   
     tf.zeros(维度)
     
   - 创建全为1的张量
   
     tf.ones(维度)
     
   - 创建全文指定值的张量
   
     tf.fill(维度，指定值)
     
   - 生成正态分布的随机数，默认均值为0，标准差为1
   
     tf.random.normal(维度，mean=均值，stddev=标准差)
     
   - 生成截断式正态分布的随机数
   
     tf.random.truncated_normal(维度，mean=均值，stddev=标准差)
     
     保证生成值在均值附近，正负两倍标准差之内
     
   - 生成均匀分布随机数
   
     tf.random.uniform(维度，minval=最小值，maxval=最大值)
     
   - 强制tensor转换为该数据类型
   
     tf.cast(张量名，dtype=数据类型)
     
   - 计算张量维度上元素的最小值
   
     tf.reduce_min(张量名)
     
   - 计算张量维度上元素的最大值
   
     tf.reduce_max(张量名)
     
   - 理解axis
   
     对于二维张量
     
     axis=0表示对第一个维度（经度）
     
     axis=1表示对第二个维度（维度）
     
     如果不指定axis表示所有元素参与计算
     
   - 计算张量沿着指定维度的平均值
   
     tf.reduce_mean(张量名，axis=操作轴)
     
   - 计算张量沿着指定维度的和
   
     tf.reduce_sum(张量名，axis=操作轴)
     
   - tf.Variable
   
     tf.Variable()将变量标记为“可训练”，被标记的变量会在反向传播中记录梯度信息。
     
     **神经网络训练中，常用该函数标记待训练参数。**
     
   - Tensorflow中的数学运算
   
     四则运算(必须维度相同)：tf.add(), tf.subtract(), tf.multiply(), tf.divide()
     
     平方、次方与开方：tf.square(张量名), tf.pow(张量名，n次方数), tf.sqrt(张量名)
     
     矩阵乘法：tf.matmul(矩阵1，矩阵2)
     
   - tf.data.Dataset.from_tensor_slices
   
     切分传入张量的第一个维度，生成输入特征/标签对，构建数据集
     
     data=tf.data.Dataset.from_tensor_slices((输入特征，标签))
     
   - tf.GradientTape
   
     with结构记录计算过程，gradient求出张量的梯度
     
     ```python
     with tf.GradientTape() as tape:
         若干个计算过程
     grad = tape.gradient(函数，对谁求导)
     ```
     
   - enumerate
   
     enumerate是python的内建函数，它可遍历每个元素（如列表、元组或字符串），**组合为：索引 元素**，常在for循环中使用。
     
     ```python
     seq = ['one', 'two', 'three']
     for i, element in enumerate(seq):
         print(i, element)
     ```
     
   - tf.one_hot
   
     独热编码（one-hot encoding）：在分类问题中，常用独热码做标签，标记类别为：1表示是，0表示非。
     
     tf.one_hot()函数将待转换数据，转换为one-hot形式的数据输出（第几个位置为1）.
     
     tf.one_hot(待转换数据，depth=几分类)
     
     ```python
     classes = 3
     labels = tf.constant([1,0,2])
     output = tf.one_hot(labels, depth=classes)
     print(output)
     ```
     
     1 => (0, 1, 0）
     
     0 => (1, 0, 0)
     
     2 => (0, 0, 1)
     
   - tf.nn.softmax
   
     柔性最大值，使输出符合概率分布
     
     当n分类的n个输出(y0, y1, y2, ..., yn-1)通过softmax()函数，便符合概率分布了。
     
     ```python
     y = tf.constant([1.01, 2.01, -0.66])
     y_pro = tf.nn.softmax(y)
     print("After softmax, y_pro is:", y_pro)
     ```
     
   - assign_sub
   
     赋值操作，更新参数的值并返回。
     
     调用assgin_sub前，先用tf.Variable定义变量w为可训练（可自更新）。
     
     w.assgin_sub(w要自减的内容)
     
   - tf.argmax
   
     返回张量沿指定维度最大值的索引
     
     tf.argmax(张量名，axis=操作轴)
     
     ```python
     import numpy as np
     test = np.array([[1, 2, 3],[2, 3, 4], [5, 4, 3], [8, 7, 2]])
     print(test)
     print(tf.argmax(test, axis=0)) # 返回每一列（经度）最大值的索引
     print(tf.argmax(test, axis=1)) # 返回每一行（纬度）最大值的索引
     ```
     
   - 
   
     
     
     
     
     