from ActivationViz.preprocess_train import *
import os
import keras
import numpy as np
import matplotlib.pyplot as plt

def vizualize_layers(layers,activations):
    """
    Vizualize all the channels for all hidden activations.
    :param layers: list of the model's layers names to be vizualized.
    :param activation:  layers' activations.
    :return:
    """
    img_per_row = 16
    for layer_name,layer_activation in zip(layers,activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        ncols = n_features//img_per_row

        display_grid = np.zeros((size*ncols,img_per_row*size))

        for col in range(ncols):
            for row in range(img_per_row):
                channel_image = layer_activation[0,:,:,col*img_per_row+row]
                channel_image  -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image,0,255).astype('uint8')

                display_grid[col*size:(col+1)*size,
                            row*size:(row+1)*size] = channel_image

        scale = 1./255
        plt.figure(figsize=(scale*display_grid.shape[1],
                            scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid,aspect='auto',cmap='viridis',extent=[display_grid.shape[1],0,display_grid.shape[0],0])
        plt.show()




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

    layer_names = [layer.name for layer in model.layers[:8]]
    vizualize_layers(layer_names,activations)



