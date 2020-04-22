import keras
import keras.callbacks
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)

im = plt.imshow(x_train[0],cmap="gray")
plt.show()

# 预处理数据
# 1.归一化
x_train = x_train/255
x_test = x_test/255
# 2.降维
x_train = x_train.reshape(x_train.shape[0],-1)   # 获取训练数据矩阵的第一个值（图片的张数）# 将其他的值相乘后放在后面，因为一个tensor目前最多为4维的（图片总张数，图片的长，宽，通道（RGB或则是灰度）），
x_test = x_test.reshape(x_test.shape[0],-1)
# 对标签做one_hot编码
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras .utils.to_categorical(y_test, 10)



# 定义一个自定回调函数，使用rensotboad
callbacklist = []
c_b = keras.callbacks.TensorBoard(
    log_dir='/home/zoumaoyang/chenyu/keras/graph',
    histogram_freq=0, # 每一轮后激活直方图
    embeddings_freq=0, # 没一轮后嵌套数据
)
callbacklist.append(c_b)

# 构建网络模型
# 1. 创建一个序列化网络
model = Sequential()
# 可以往model里面添加网络    做优化
model.add(Dense(369, activation='relu', input_shape=(784,)))
model.add(Dense(200, activation='relu', ))
model.add(Dense(10, activation='softmax'))
print(model.summary())
model.compile(optimizer=SGD(),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=5,callbacks=callbacklist)
print(model.evaluate(x_test, y_test)[0])