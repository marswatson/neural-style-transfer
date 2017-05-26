import vgg

import tensorflow as tf
import numpy as np

from sys import stderr
from PIL import Image
from functools import reduce

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

def generateImage(network, content, style, iterations,
        content_weight, style_weight,
        learning_rate, beta1, beta2, epsilon, pooling):
    #placehoder shape = (batch_size, image)
    shape = (1,) + content.shape
    style_shape = (1,) + style.shape
    content_features = {}
    style_features = {}

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    style_layers_weights = {}
    for layer in STYLE_LAYERS:
        style_layers_weights[layer] = 1.0

    # normalize style layer weights
    layer_weights_sum = 0
    for layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[layer]
    for layer in STYLE_LAYERS:
        style_layers_weights[layer] /= layer_weights_sum

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        #define the tf op place holder with batchsize 1, and content shape
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=style_shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        style_pre = np.array([vgg.preprocess(style, vgg_mean_pixel)])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            #gram matrix made by vectorised feature map
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        #perform gradient decent on a white noise image to find another image that matches the feature respense of original image
        initial = tf.random_normal(shape) * 0.256
        image = tf.Variable(initial)

        net = vgg.net_preloaded(vgg_weights, image, pooling)

        # content loss
        content_layers_weights = {}
        content_weight_blend = 1.0
        # we only use conv4 content
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1-content_weight_blend

        #compute content loss
        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(content_layers_weights[content_layer]  * (2 * tf.nn.l2_loss(
                    net[content_layer] - content_features[content_layer]) /
                    content_features[content_layer].size))
        content_loss += reduce(tf.add, content_losses)

        # style loss
        style_loss = 0
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
        style_loss +=  reduce(tf.add, style_losses)


        # total variation denoising
        # tv_y_size = _tensor_size(image[:,1:,:,:])
        # tv_x_size = _tensor_size(image[:,:,1:,:])
        # tv_loss = tv_weight * 2 * (
        #         (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
        #             tv_y_size) +
        #         (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
        #             tv_x_size))
        # overall loss
        loss = content_weight*content_loss + style_weight * style_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            for i in range(iterations):
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                train_step.run()

                #check if its last step, if it is, we save the image
                last_step = (i == iterations - 1)
                if last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                    yield (
                        (None if last_step else i),
                        img_out
                    )
