# tensor切片
import keras
from keras.datasets import mnist
# 构建网络需要导入至少三个模块
# 1.序列化模型
# 2.核心层
# 3. 优化器
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 获取每个张量的轴
print(train_images.ndim)
# 获取每个张量的形状
print(train_images.shape)
# 数据类型
print(train_images.dtype)

# 切片沿着轴的方向来切片
    # 1.切去100到200的的图片
print(train_images[100:200].shape)
    # 2.加其它轴
print(train_images[:, 0:14, 0:].shape)
    # 3.张量变形  ,变形的总数应该一样
print(train_images.reshape(47040000, 1).shape)
print(train_images.reshape(60000, 28, 28).shape)


# 做预处理
# 1.对训练数据做归一化处理
train_images = train_images/255
test_images = test_images/255
# 2.修改数据的维度（包括训练数据和测试数据）
train_images = train_images.reshape(train_images.shape[0],-1)
test_images = test_images.reshape(test_images.shape[0],-1)
# 3.对标签值做one-hot编码
train_labels = keras.utils.to_categorical(train_labels,num_classes=10)
test_labels = keras.utils.to_categorical(test_labels,num_classes=10)

# 构建一个网络
network = Sequential()

# 往模型中添加网络
network.add(Dense(512,activation='relu',input_shape=(784,)))
network.add(Dense(256,activation="sigmoid"))
network.add(Dense(128,activation="sigmoid"))
network.add(Dense(10,activation="softmax"))
# 获取网络的结构
print(network.summary())

# 编译模型
network.compile(optimizer="SGD",loss="categorical_crossentropy",metrics=['accuracy'])

# 运行模型
network.fit(train_images,train_labels,batch_size=128,epochs=10)