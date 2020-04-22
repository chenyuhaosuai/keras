# 扩展csv_base模型（把自定义的密集层添加到卷积基之后，但是要冻结卷积基的权重）
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from keras import losses

# 获取卷积基
def get_conv_base():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150,150,3)
                      )
    return conv_base

# 构造可以使用数据增强的数据生成器
def get_data_generator():
    data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return data_gen

# 生成定义大小的数据
def getdata(dir,data_gen):
    batch_size = 20
    data_generator = data_gen.flow_from_directory(
        directory= dir,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode="binary"
    )
    return  data_generator

if __name__ =="__main__":
    conv_base=get_conv_base()
    data_gen=get_data_generator()
    train_dir = "/home/zoumaoyang/chenyu/keras/data/small_dogandcat/train"
    validation_dir = "/home/zoumaoyang/chenyu/keras/data/small_dogandcat/validation"
    train_data_generator=getdata(train_dir,data_gen)
    validation_data_generator = getdata(validation_dir, data_gen)
    # 构建网络（在编译和训练之前来冻结）
    conv_base.trainable
    model = Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(1,activation="sigmoid"))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc']
                  )
    history = model.fit_generator(train_data_generator,
                        steps_per_epoch=100,
                        epochs=30,
                        validation_data=validation_data_generator,
                        validation_steps=50
                        )
    history_dict = history.history
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'bo', label='train_loss')
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