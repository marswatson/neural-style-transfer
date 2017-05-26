import os

import numpy as np
import scipy.misc

from train_image import generateImage
from argparse import ArgumentParser

from PIL import Image

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'avg'

''' build parser '''
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--styles',
            dest='style',help='style image',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    return parser

''' read image '''
def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img

''' save image '''
def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

''' main function '''
def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(VGG_PATH):
        print("Network %s does not exist." % VGG_PATH)
        return

    content_image = imread(options.content)
    style_image = imread(options.style)

    #set the output image shape
    target_shape = content_image.shape

    #resize the style image as content image
    style_scale = STYLE_SCALE
    style_image = scipy.misc.imresize(style_image, style_scale *target_shape[1] / style_image.shape[1])


    for iteration, image in generateImage(
        network=VGG_PATH,
        content=content_image,
        style=style_image,
        iterations=options.iterations,
        content_weight=CONTENT_WEIGHT,
        style_weight=STYLE_WEIGHT,
        #optimization parameters
        learning_rate=LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        pooling=POOLING,
    ):
        output_file = None
        combined_rgb = image
        if iteration is not None:
            if options.checkpoint_output:
                output_file = options.checkpoint_output % iteration
        else:
            output_file = options.output
        if output_file:
            imsave(output_file, combined_rgb)

if __name__ == '__main__':
    main()