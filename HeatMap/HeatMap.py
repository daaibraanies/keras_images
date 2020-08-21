import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing import image


def img2nn_fromat(image_file, target_size):
    img = image.load_img(image_file, target_size=(target_size, target_size))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def get_heatmap(model, output_id, layer_name, img):
    map_layer = model.get_layer(layer_name)
    target_output = model.output[:, output_id]

    grads = K.gradients(target_output, map_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, map_layer.output[0]])
    pooled_grads_value, target_conv_layer_value = iterate(img)

    for i in range(pooled_grads.shape[0]):
        target_conv_layer_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(target_conv_layer_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


def save_ingest_heatmap(heatmap, img_file, save_file='img_w_heatmap.jpg'):
    original_img = cv2.imread(img_file)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposure = heatmap * alpha + original_img
    cv2.imwrite(save_file, superimposure)

    print("Ingest heatmap saved as {}.".format(save_file))


if __name__ == "__main__":
    alpha = 0.4
    imgae_file = 'chickadee.jpg'

    model = VGG19(weights='imagenet')
    img = img2nn_fromat(imgae_file, 224)
    prediction = model.predict(img)
    print("Predicted animal: {}".format(decode_predictions(prediction, top=3)[0]))
    chickadee_output_id = np.argmax(prediction[0])
    print("Prediction argmax corresponds to id: {}".format(chickadee_output_id))  # chickadee id = 19

    heatmap = get_heatmap(model, chickadee_output_id, 'block5_conv4', img)
    plt.matshow(heatmap)
    plt.show()
    
    save_ingest_heatmap(heatmap, imgae_file)
