from ActivationViz.preprocess_train import *
import os
import keras
import numpy as np
import matplotlib.pyplot as plt

if __name__ =='__main__':
    model = None
    if not os.path.exists('activiz_model.h5'):
        print("Model was not found. Creating a new one.")
        train_dir,val_dir,test_dir=create_dataset()
        dirdict = {'train':train_dir,'val':val_dir,'test':test_dir}
        model = create_simple_nn(dirdict)
    else:
        print("A previously trained model has been found. Loading...")
        model = keras.models.load_model('activiz_model.h5')
        model.summary()

    img = keras.preprocessing.image.load_img('cat_img.jpg',target_size=(150,150))
    img_tensor = keras.preprocessing.image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor,axis=0)
    img_tensor /= 255.

    plt.imshow(img_tensor[0])
    plt.show()

    layers_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = keras.models.Model(inputs=model.input,
                                          outputs=layers_outputs)
    activations = activation_model.predict(img_tensor)


    plt.show()


