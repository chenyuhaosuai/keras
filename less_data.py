import os, shutil
from keras_preprocessing.image import ImageDataGenerator
import keras
from keras import Sequential
from keras import layers
from keras import optimizers
from keras import losses
from keras import callbacks
import matplotlib.pyplot as plt

def dataset():
    original_dataset_dir = '/home/zoumaoyang/chenyu/keras/data/dogandcat/train'
    #创建一个存储少量数据的文件夹
    base_dir = '/home/zoumaoyang/chenyu/keras/data/small_dogandcat'
    os.mkdir(base_dir)

    train_dir = os.path.join(base_dir,'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir,'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir,'test')
    os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir,'cats')
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir,'dogs')
    os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir,"cats")
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir,'dogs')
    os.mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir,'cats')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir,'dogs')
    os.mkdir(test_dogs_dir)

    # 将猫中前一千张复制到train_cats_dir
    fnames =['cat.{}.jpg'.format(i) for i in range(1000)]   #获取一个名字列表
    for fname in fnames:
        src = os.path.join(original_dataset_dir,fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src,dst)

    # 将猫中前1000,1500复制到validation_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir,  fname)
        shutil.copyfile(src,dst)

    # 将猫中1500,2000复制到test_cats_dir中
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    # 将狗中前一千张复制到train_cats_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]  # 获取一个名字列表
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # 将狗中前1000,1500复制到validation_cats_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # 将狗中1500,2000复制到test_cats_dir中
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)
# 数据预处理(在keras中用库ImageDataGenerator)
def process():
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1./255,
                                            rotation_range=40,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(directory="/home/zoumaoyang/chenyu/keras/data/small_dogandcat/train",  #目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用(并且每个文件夹中的数据都是同一个类别的)
                                                        target_size=(150,150),
                                                        batch_size=20,
                                                        class_mode= 'binary'  # 返回标签的格式
                                                        )
    validation_generator = validation_datagen.flow_from_directory(directory="/home/zoumaoyang/chenyu/keras/data/small_dogandcat/validation",#目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用
                                                        target_size=(150,150),
                                                        batch_size=20,
                                                        class_mode= 'binary'  # 返回标签的格式
                                                                  )
    test_generator = test_datagen.flow_from_directory(directory="/home/zoumaoyang/chenyu/keras/data/small_dogandcat/test",  #目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用
                                                        target_size=(150,150),
                                                        batch_size=20,
                                                        class_mode= 'binary'  # 返回标签的格式
                                                      )
    return train_generator,validation_generator,test_generator
# 定义回调函数
def get_callback():
    callback_list = []
    earlr_c = callbacks.EarlyStopping(
        monitor='val_loss',   # 监察点（一般是验证集的数据）
        patience=1    # 监察次数（当监察点的值在两轮里的值没哟改变的的话就停止训练）
    )
    check_point = callbacks.ModelCheckpoint(
        filepath='cy.h5',     #当vol_acc不再变化时，就不再保存文件
        monitor='val_loss',
        save_best_only=True
    )
    # 如果验证损失不再变化，通过回调函数来降低学习率
    lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,  # 触发时将数学率除10
        patience=10   #(不改变次数为10次)
    )

    class self_callback(callbacks.Callback):
        # 定义模型
        # def set_model(self, model):
        #     self.model = model
        def on_epoch_begin(self, epoch, logs=None):
            print("train_start")

        def on_epoch_end(self, epoch, logs=None):
            print("train_end")

    #callback_list.append(earlr_c)
    #callback_list.append(check_point)
    self_callback = self_callback()
    callback_list.append(self_callback)
    return callback_list


# 定义网络
def network(train_generator,validation_generator,test_generator,callback_list):
    model = Sequential()
    model.add(layers.Deconv2D(32,(3,3),activation="relu", input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Deconv2D(64,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Deconv2D(128,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Deconv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512,activation="relu"))
    model.add(layers.Dense(1,activation="sigmoid"))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc']
                  )
    history=model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=50,
        callbacks=callback_list,
        validation_data=validation_generator,
        validation_steps=50
    )
    model.save('cats_and_dogs_small_2.h5')
    return history
def painting(history_dict):
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(train_loss)+1)
    plt.plot(epochs,train_loss,'bo',label="train_loss")
    plt.plot(epochs,val_loss,'b',label="val_loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    train_acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, 'bo', label="train_acc")
    plt.plot(epochs, val_acc, 'b', label="val_acc")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_generator, validation_generator, test_generator = process()
    callback_list=get_callback()
    history = network(train_generator, validation_generator, test_generator,callback_list)
    history_dict=history.history
    painting(history_dict)
