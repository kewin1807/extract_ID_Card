import time
import click
import glob
import cv2
import multiprocessing
from mrcnn.config import Config
from imutils import transform
from mrcnn import visualize
import mrcnn.model as modellib
import matplotlib.pyplot as plt
import matplotlib
import skimage.io
import numpy as np
import math
import random
import sys
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# pip install https://github.com/lakshayg/tensorflow-build/releases/download/tf1.14.1-mojave-py3.7/tensorflow-1.14.1-cp37-cp37m-macosx_10_9_x86_64.whl
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

exts = ['*.jpg', '*.png']
MODEL_DIR = "logs"


class CropConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.85

    def __init__(self, name, images_per_gpu, gpus):
        self.NAME = name
        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        self.IMAGES_PER_GPU = images_per_gpu
        self.GPU_COUNT = gpus
        super().__init__()


exts = ['.jpg', '.png', '.jpeg', '.JPG']


def get_images(input):
    files = []
    _, ext = os.path.splitext(input)
    if ext in exts:
        files.append(input)
    else:
        for parent, dirnames, filenames in os.walk(input):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
    print('Find {} images'.format(len(files)))
    return files


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


@click.command()
@click.option('--input', '-i', required=True, help='Input Folder or image')
@click.option('--output', '-o', default='', help='Output folder')
@click.option('--padding', '-p', default=0, type=int)
@click.option('--model_name', '-n', default='maskrcnn')
@click.option('--model_path', '-m', default='logs/card20191004T0826/mask_rcnn_card_0010.h5')
@click.option('--verbose', '-v', default=True, type=bool)
@click.option('--show_size', '-ss', default=800, type=int)
@click.option('--max_size', '-maxs', default=512, type=int)
# @click.option('--min_size', '-mins', default=512, type=int)  # 1024, 512, 448, 384
@click.option('--images_per_gpu', '-ipg', default=multiprocessing.cpu_count(), type=int)
@click.option('--gpus', '-g', default=1, type=int)
@click.option('--quality', '-q', default=80, type=int)
def main(input, output, padding, model_name, model_path, verbose, show_size, max_size, images_per_gpu, gpus, quality):
    image_paths = get_images(input)
    # can not exceed the length of images
    images_per_gpu = min(images_per_gpu, len(image_paths))
    if images_per_gpu < 1:
        return
    config = CropConfig(model_name, max_size, images_per_gpu, gpus)
    # config.IMAGE_MAX_DIM = max_size
    # config.IMAGE_MIN_DIM = min_size
    # config.IMAGE_SHAPE = np.array([config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM,
    #                                config.IMAGE_CHANNEL_COUNT])
    if verbose:
        config.display()
    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(model_path, by_name=True)
    image_paths = list(chunks(image_paths, images_per_gpu))
    need_wait = False
    for chunk_paths in image_paths:
        origin_images = []
        for path in chunk_paths:
            origin_image = cv2.imread(path)
            if origin_image is not None:
                origin_images.append(origin_image)

        if len(origin_images) < 1:
            continue
        basenames = [os.path.basename(path).split(".")[0]
                     for path in chunk_paths]
        # fix model config
        # model.config.BATCH_SIZE = len(chunk_paths)
        if len(chunk_paths) < images_per_gpu:
            for i in range(0, images_per_gpu - len(chunk_paths)):
                origin_images.append(np.zeros((1, 1, 3), np.uint8))
        # process image
        start = time.time()
        result = model.detect(origin_images, verbose=verbose)
        cost_time = (time.time() - start)
        click.secho("cost time: {:.2f}s".format(cost_time), fg="yellow")

        for index, r in enumerate(result):
            origin_image = origin_images[index]
            if origin_image.shape[0] < 10:
                continue
            basename = basenames[index]
            for i in range(0, len(r['scores'])):
                mask = r["masks"][:, :, i]
                mask_image = np.reshape(mask == 1, (mask.shape[0], mask.shape[1], 1)).astype(
                    np.uint8)
                cnts = cv2.findContours(
                    mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

                if len(cnts) != 0:
                    c = max(cnts, key=cv2.contourArea)
                    # try to find quadrilateral
                    # epsilon = 0.01 * cv2.arcLength(c, True)
                    # approx = cv2.approxPolyDP(c, epsilon, True)
                    # if len(approx) == 4:
                    #     box = np.squeeze(approx, axis=1)
                    # else:
                    #     box = cv2.boxPoints(cv2.minAreaRect(c))
                    box = cv2.boxPoints(cv2.minAreaRect(c))

                    # apply the perspective transformation
                    img = transform.four_point_transform(origin_image, box)
                    # fix clockwise
                    if img.shape[0] > img.shape[1]:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    h, w, _ = img.shape
                    if quality == 100:
                        file_name = "{}/{}_{}.png".format(
                            output, basename, i)
                    else:
                        file_name = "{}/{}_{}.jpg".format(
                            output, basename, i)

                    if not output:
                        cv2.namedWindow(file_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(file_name, show_size,
                                         int(show_size * (h / w)))
                        cv2.imshow(file_name, img)
                    else:
                        if quality == 100:
                            cv2.imwrite(file_name, img)
                        else:
                            cv2.imwrite(file_name, img, [
                                        int(cv2.IMWRITE_JPEG_QUALITY), quality])

    if not output:
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
