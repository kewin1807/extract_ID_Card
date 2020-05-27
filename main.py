from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import tensorflow as tf
import math
import cv2
import os.path as osp
import glob
import numpy as np
from shapely.geometry import Polygon
import pyclipper
import time
from model import dbnet
from imutils import transform, utils
import click
from correction.address_correction import AddressCorrection
from correction.fullname_correction import FullnameCorrection
import helper
from mrcnn.config import Config
from imutils import transform
from mrcnn import visualize
import mrcnn.model as modellib
from multiprocessing import Pool


app = Flask(__name__)
pool = None
app.config['DEFAULT_PARSERS'] = [
    'flask.ext.api.parsers.JSONParser',
    'flask.ext.api.parsers.URLEncodedParser',
    'flask.ext.api.parsers.MultiPartParser'
]
CORS(app)


mean = np.array([103.939, 116.779, 123.68])


class CropConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    def __init__(self, name, images_per_gpu, gpus):
        self.NAME = name
        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        self.IMAGES_PER_GPU = images_per_gpu
        self.GPU_COUNT = gpus
        super().__init__()


def load_model():
    sess = tf.InteractiveSession()
    _, predictModel = dbnet()
    model = predictModel
    model.load_weights('models/model.h5', by_name=True, skip_mismatch=True)
    config = CropConfig("card", 1, 1)
    model_crop = modellib.MaskRCNN(
        mode="inference", model_dir="logs", config=config)
    model_crop.load_weights("model_card/mask_rcnn_card_0030.h5", by_name=True)
    graph = tf.get_default_graph()
    return model, graph, sess, model_crop


def resize_image(image, image_short_side=544):
    height, width, _ = image.shape
    if height < width:
        new_height = image_short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = image_short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized_img


def box_score_fast(bitmap, _box):
    # 计算 box 包围的区域的平均得分
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7):
    pred = pred[..., 0]
    bitmap = bitmap[..., 0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    contours = cv2.findContours(
        (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    for contour in contours[:max_candidates]:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=2.0)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:
            continue

        box[:, 0] = np.clip(
            np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height)

        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores


def crop(image):
    global sess, graph
    start_time = time.time()
    im = None
    with sess.as_default():
        with graph.as_default():
            r = model_crop.detect([image], verbose=1)[0]
    process_time = round(time.time() - start_time)
    if r["masks"].shape[2] == 0:
        return im, 0
    mask = r["masks"][:, :, 0]
    mask_image = np.reshape(mask == 1, (mask.shape[0], mask.shape[1], 1)).astype(
        np.uint8)
    cnts = cv2.findContours(
        mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) != 0:
        c = max(cnts, key=cv2.contourArea)
        box = cv2.boxPoints(cv2.minAreaRect(c))
        im = transform.four_point_transform(image, box)
        if im.shape[0] > im.shape[1]:
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return im, process_time


def segmentWord(img):
    src_image = img.copy()
    h, w = img.shape[:2]
    img = resize_image(img)
    img = img.astype(np.float32)
    img -= mean
    image_input = np.expand_dims(img, axis=0)
    start = time.time()
    with sess.as_default():
        with graph.as_default():
            p = modelSegment.predict(image_input)[0]
    bitmap = p > 0.3
    boxes, scores = polygons_from_bitmap(p, bitmap, w, h, box_thresh=0.5)
    predict_time = round(time.time() - start, 2)
    rects = [cv2.minAreaRect(np.array(box)) for box in boxes]
    centers = utils.group_box_row(src_image, rects, boxes)
    return centers


@app.route("/api/detect", methods=["POST"])
def recognize():
    UPLOAD_FOLDER = "uploads"
    img = None
    file = request.files['file']
    filename = secure_filename(file.filename)
    filename = filename + ".png"
    destination = os.path.join(UPLOAD_FOLDER, filename)
    file.save(destination)
    original_image = cv2.imread(destination)

    if helper.elements(np.array(original_image)) == 0:
        return jsonify({"error": {"message": "Không nhận diện được ảnh"}, "data": {}}), 200
    img, process_card = crop(original_image)

    if helper.elements(np.array(img)) == 0:
        return jsonify({"error": {"message": "Không nhận diện được ảnh"}, "data": {}}), 200
    h, w, _ = img.shape
    if h <= 500:
        return jsonify({"error": {"message": "Kích thước ảnh hơi nhỏ"}, "data": {}}), 200
    else:
        rotate_image = cv2.rotate(img, cv2.ROTATE_180)
        if helper.rotate_face(img):
            img = rotate_image
        original_image_crop = img.copy()
        print("shape: ", original_image_crop.shape)

        img = utils.resize(img, height=900)
        src_image = img.copy()
        cv2.imwrite("test.png", img)
        rh = original_image_crop.shape[0] / img.shape[0]
        rw = original_image_crop.shape[1] / img.shape[1]
        centers = segmentWord(img)
        images = []
        model_names = []

        # group image with line
        for key in centers:
            model_name = "vie"
            boxes = np.asarray(centers[key])

            boxes = boxes.reshape(-1, 2)
            minBox = cv2.minAreaRect(np.array(boxes))
            minBox = utils.fix_rect(minBox, 3)
            box = cv2.boxPoints(minBox)
            cnt = np.int0(box)

        # return origin_image
            original_box = np.array([[x * rw, y * rh]
                                     for x, y in box], dtype="float32")
            img_ind = transform.four_point_transform(
                original_image_crop, original_box)

            cv2.drawContours(src_image, [np.array(cnt)], -1, (0, 255, 0), 2)

            if int(key) == len(centers) - 3:
                h, w, _ = img_ind.shape
                img_ind = img_ind[0:h, int(w/6):w]
                # img_ind = cv2.resize(img_ind, None, fx=1.5,
                #                      fy=1.5, interpolation=cv2.INTER_CUBIC)
                model_name = "ID"
            img_ind = cv2.detailEnhance(img_ind, sigma_s=10, sigma_r=0.015)
            images.append(img_ind)
            model_names.append(model_name)
        start = time.time()
        result = pool.map(helper.detect, [(im, model_name) for (
            im, model_name) in zip(images, model_names)])
        print(result)
        cv2.imwrite("test_2.png", src_image)
        # print(result.get(timeout=2))
        # result = list(result)
        raw_data = result
        result = helper.extraction(result)
        cost_time = (time.time() - start)
        result["process_time"] = round(process_card + cost_time, 2)
        return jsonify({"error": {"message": "Thành công"}, "data": result}), 200


global modelSegment, graph, sess, model_crop
modelSegment, graph, sess, model_crop = load_model()


if __name__ == "__main__":
    pool = Pool(processes=os.cpu_count())
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    try:

        app.run(host="0.0.0.0", port=5000)
    except:
        pool.close()
        pool.join()
