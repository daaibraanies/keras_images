import numpy as np
import scipy
from imageio import imwrite
from keras import backend as K
from keras.applications import inception_v3
from keras.preprocessing.image import image


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_values = outs[0]
    grad_values = outs[1]
    return loss_values, grad_values


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_val, grad_val = eval_loss_and_grads(x)
        if max_loss is not None and loss_val > max_loss:
            break
        print('...Loss value at {} : {}'.format(i, loss_val))
        x += step * grad_val
    return x


def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)


def process_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')  # выглядит как костыль, надо поменять перед заливом
    return x


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    imwrite(fname, pil_img)


layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2,
    'mixed5': 1.5
}

test = K.set_learning_phase(0)
model = inception_v3.InceptionV3(weights='imagenet',
                                 include_top=False)

model.summary()

layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss = loss + (coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling)

dream = model.input
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

fetch_loss_and_grads = K.function([dream], [loss, grads])

###################################################
# Applying filters

step = 0.01
num_octave = 3
octave_scale = 1.4
iterations = 20
max_loss = 10.
image_path = 'aleksa.jpg'
img = process_image(image_path)

original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)

successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape {}'.format(shape))
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    upscale_shrunk_original_image = resize_img(original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_datails = same_size_original - upscale_shrunk_original_image

    img += lost_datails
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
save_img(img, fname='final_dream.png')
