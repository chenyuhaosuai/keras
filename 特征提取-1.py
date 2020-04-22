# 使用VGG16网络的卷积基(不使用数据增强,使用卷积基输出自己数据的numpy数组（包括特征值和标签），优点：速度快，但是可能出现过拟合)
# 适用于自己本身数据量很大

from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import Sequential
from keras import layers
from keras import optimizers
from keras import losses
import matplotlib.pyplot as plt

# 获取训练网络的卷积基
def getconv_base():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3)
                      )
    return conv_base

# 定义数据生成器.
def datagenerator():
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen


#特征提取(将数据生成器中生成的数据传入到卷积基中生成对应的Numpy数据)
def extract_feature(dir, sample_count, datagen, conv_base):
    features = np.zeros(shape=(sample_count, 4, 4, 512))   # 用来存储卷积基最后输出的Numpy的矩阵（特征）
    labels = np.zeros(shape=(sample_count))  # （标签）
    batch_size = 20

    generator = datagen.flow_from_directory(
        directory=dir,
        target_size=(150,150),
        class_mode='binary',
        batch_size=batch_size
    )
    i=0
    for input_image, input_label in generator:
        features_batch =conv_base.predict(input_image)
        features[i * batch_size:(i+1)*batch_size] = features_batch
        labels[i * batch_size:(i+1)*batch_size] = input_label
        i += 1
        if i*batch_size >= sample_count:
            break
    return features,labels

# 拉伸数据
def data_falteen(features):
    reshape_feature=np.reshape(features,(features.shape[0],features.shape[1]*features.shape[2]*features.shape[3]))
    return reshape_feature


if __name__ == '__main__':
    conv_base=getconv_base()
    datagen=datagenerator()
    train_dir = "/home/zoumaoyang/chenyu/keras/data/small_dogandcat/train"
    validation_dir = "/home/zoumaoyang/chenyu/keras/data/small_dogandcat/validation"
    test_dir = "/home/zoumaoyang/chenyu/keras/data/small_dogandcat/test"
    train_features,train_labels = extract_feature(train_dir,2000,datagen,conv_base)
    validation_features ,validation_labels = extract_feature(validation_dir,1000,datagen,conv_base)
    # test_features, test_labels = extract_feature(test_dir, 1000, datagen, conv_base)
    train_features=data_falteen(train_features)
    validation_features=data_falteen(validation_features)
    model = Sequential()
    model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation="sigmoid"))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss="binary_crossentropy",
                  metrics=['acc']
                  )
    history = model.fit(train_features,train_labels,
                        batch_size=20,
                        epochs=100,
                        validation_data=[validation_features,validation_labels]
                        )
    history_dict = history.history
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(train_loss)+1)
    plt.plot(epochs, train_loss,'bo', label='train_loss')
    plt.plot(epochs, val_loss, 'b', label='val_loss')
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    train_acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, 'bo', label='train_acc')
    plt.plot(epochs, val_acc, 'b', label='val_acc')
    plt.xlabel("Epochs")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
