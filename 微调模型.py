# 微调模型（与特征提取先对应）
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
from keras import optimizers
from matplotlib import pyplot as plt
from keras import losses

# 获取模型的卷积核
def get_conv_base():
    conv_base = VGG16(
        weights='imagenet',
        input_shape=(150,150,3),
        include_top=False
    )
    return conv_base

# 获取数据生成器
def get_data_gen():
    data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    return data_gen

# 利用数据生成器到指定文件中生成图像
def get_generator(data_gen,dir):
    generator = data_gen.flow_from_directory(directory=dir,
                                             target_size=(150,150),
                                             class_mode="binary",
                                             batch_size=32
                                             )
    return generator

# 构建网络
def get_model(conv_base):
    conv_base.summary()
    conv_base.trainable=False
    # 解冻
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1 ':
            set_trainable = True
        if set_trainable == True:
            layer.trainable = True
        else:
            layer.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-5),
        loss='binary_crossentropy',
        metrics=['acc']
    )
    return model


if __name__ == "__main__":
    conv_base=get_conv_base()
    data_gen=get_data_gen()
    train_dir = "/home/zoumaoyang/chenyu/keras/data/small_dogandcat/train"
    validation_dir = '/home/zoumaoyang/chenyu/keras/data/small_dogandcat/validation'
    train_generator=get_generator(data_gen,train_dir)
    validation_generator=get_generator(data_gen,validation_dir)
    model = get_model(conv_base)
    his = model.fit_generator(generator=train_generator,
                        steps_per_epoch=100,
                        epochs=30,
                        validation_data=validation_generator,
                        validation_steps=50
                        )
    history_dict = his.history
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