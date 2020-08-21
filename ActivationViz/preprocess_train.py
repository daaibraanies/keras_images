import os,shutil
import keras
import matplotlib.pyplot as plt

def byclass_copy(class_name,n,source,destination):
    """
    Copies range n images of a particular class form
    source folder to destination.
    :param class_name: class name that image belongs to.
    :param n: range of image ids.
    :param source: source directory.
    :param destination: destination directory
    :return:
    """
    fnames = [class_name+'.{}.jpg'.format(i) for i in range(*n)]
    for fname in fnames:
        src = os.path.join(source,fname)
        dst = os.path.join(destination,fname)
        shutil.copyfile(src,dst)

def create_dataset():
    """
    Creates small dataset from the standard cats-vs-dogs if not exist.
    Files must be located in the train_data/train folder.
    :return:
    """
    if os.path.exists('train_data/dataset'):
        return 'train_data/dataset/train',\
                'train_data/dataset/val',\
                'train_data/dataset/test'
    else:
        if os.path.exists('train_data/train'):
            dataset = 'train_data/dataset'
            os.mkdir(dataset)

            train_dir = os.path.join(dataset,'train')
            val_dir = os.path.join(dataset,'val')
            test_dir = os.path.join(dataset,'test')

            os.mkdir(train_dir)
            os.mkdir(val_dir)
            os.mkdir(test_dir)

            train_cats_dir = os.path.join(train_dir,'cats')
            val_cats_dir = os.path.join(val_dir,'cats')
            test_cats_dir = os.path.join(test_dir,'cats')

            train_dogs_dir = os.path.join(train_dir,'dogs')
            val_dogs_dir = os.path.join(val_dir,'dogs')
            test_dogs_dir = os.path.join(test_dir,'dogs')

            os.mkdir(train_cats_dir)
            os.mkdir(val_cats_dir)
            os.mkdir(test_cats_dir)

            os.mkdir(train_dogs_dir)
            os.mkdir(val_dogs_dir)
            os.mkdir(test_dogs_dir)

            byclass_copy('cat',(1000,),'train_data/train',train_cats_dir)
            byclass_copy('cat',(1000,1500),'train_data/train',val_cats_dir)
            byclass_copy('cat',(1500,2000),'train_data/train',test_cats_dir)

            byclass_copy('dog',(1000,),'train_data/train',train_dogs_dir)
            byclass_copy('dog',(1000,1500),'train_data/train',val_dogs_dir)
            byclass_copy('dog',(1500,2000),'train_data/train',test_dogs_dir)

            return train_dir,val_dir,test_dir
        else:
            print('Folder with files has not been found. Looking for: train_data/train')

def create_generator(data_dir,target_size,
                     batch_size,class_mode,scale):
    generator = keras.preprocessing.image.ImageDataGenerator(rescale=scale,
                                                             rotation_range=40,
                                                             width_shift_range=0.2,
                                                             height_shift_range=0.2,
                                                             shear_range=0.2,
                                                             zoom_range=0.2,
                                                             horizontal_flip=True)
    generator = generator.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode
    )
    return generator

def create_simple_nn(dirdict,mname='activiz_model.h5',verbose=True):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
    model.add(keras.layers.MaxPooling2D(2,2))
    model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(51,activation='relu'))
    model.add(keras.layers.Dense(1,activation='sigmoid'))

    if verbose:
        model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=1e-3,amsgrad=True),
                  metrics=['acc'])

    if verbose:
        print("Getting generators ready...")
    train_gen = create_generator(dirdict['train'],(150,150),20,'binary',1./255)
    val_gen = create_generator(dirdict['val'], (150, 150), 20, 'binary', 1. / 255)

    if verbose:
        print("Ready!\nFitting:\n")

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        epochs=30,
        validation_data=val_gen,
        validation_steps=50
    )

    model.save(mname)
    if verbose:
        print('Model saved as {}.'.format(mname))

    if verbose:
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1,len(acc)+1)

        plt.plot(epochs,acc,'bo',label='Training accuracy')
        plt.plot(epochs,val_acc,'b',label='Validation accuracy')
        plt.legend(loc='best')

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training accuracy')
        plt.plot(epochs, val_loss, 'b', label='Validation accuracy')
        plt.legend(loc='best')
        plt.show()

    return model