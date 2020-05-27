from multiprocessing import Pool
import cv2
import numpy as np
import sys
import time
import os
import tesserocr
from PIL import Image
import re
from correction.address_correction import AddressCorrection
from correction.fullname_correction import FullnameCorrection

modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
TESSDATA_PREFIX = os.environ['TESSDATA_PREFIX'] if 'TESSDATA_PREFIX' in os.environ else os.path.join(
    os.getcwd(), 'tessdata')


def get_apis(model_names):
    apis = {}
    for model_name in model_names:
        apis[model_name] = tesserocr.PyTessBaseAPI(
            path=TESSDATA_PREFIX, lang=model_name, psm=tesserocr.PSM.RAW_LINE)
    return apis


apis = get_apis(['vie', "ID"])
address_correction = AddressCorrection()
fullname_correction = FullnameCorrection()


def clean_text(text):
    text = re.sub(
        '[^0-9a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ.-/ ]', '', text)
    return text


def detect(args):
    img, model_name = args
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB = Image.fromarray(imgRGB)
    api = apis[model_name]
    api.SetImage(imgRGB)
    result = api.GetUTF8Text()
    # result = clean_text(result)
    return result


pool = Pool(os.cpu_count())


def detect_images(imgs, model_names):
    return pool.map(detect, [(im, model_name) for (im, model_name) in zip(imgs, model_names)])


def rotate_face(rotate_frame, conf_threshold=0.8, min_conf=0.3):
    frameHeight = rotate_frame.shape[0]
    frameWidth = rotate_frame.shape[1]
    blob = cv2.dnn.blobFromImage(rotate_frame, 1.0, (300, 300), [
                                 104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    confidences = detections[0, 0, :, 2]
    max_confidence = np.max(confidences)
    print(max_confidence)
    indexes = np.argmax(confidences)
    if max_confidence < conf_threshold:
        return True
    else:
        x1 = int(detections[0, 0, indexes, 3] * frameWidth)
        y1 = int(detections[0, 0, indexes, 4] * frameHeight)
        x2 = int(detections[0, 0, indexes, 5] * frameWidth)
        y2 = int(detections[0, 0, indexes, 6] * frameHeight)
        if x2 < frameWidth / 2:
            return False
        else:
            return True


def find_index_upper(line):
    count = 0
    for index, c in enumerate(line):
        if c.isupper():
            count += 1
        if count == 2:
            return index
    return -1


def elements(array):
    return array.ndim and array.size


def extraction(lines):
    lines.reverse()
    lines = lines[3:]
    objectResult = {}
    currentKey = ""
    currentName = False
    for i in range(len(lines)):
        line = lines[i]

        if i == 0:
            text = re.sub("†", "1", line)
            text = re.sub('[^0-9]+', "", text)
            currentKey = "card_number"
            objectResult[currentKey] = text
        if line.find("Ho") != -1 or line.find("Họ") != -1 or line.find("teh") != -1 or line.find("ten") != -1 or line.find("tên") != -1 or line.find("ov") != -1:
            if not currentName:
                currentKey = "fullname"

                if sum(1 for c in line if c.isupper()) > 4:
                    start_index = find_index_upper(line)
                    currentName = True
                    text = line[start_index-1:]
                    text = re.sub(
                        '[^a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ ]+', '', text)
                    objectResult[currentKey] = text
                else:
                    text = lines[i+1]
                    text = re.sub(
                        '[^a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ ]+', '', text)
                    objectResult[currentKey] = text
                    currentName = True
                    i = i+1

        if i < len(lines) and currentKey == "fullname":
            text = line.replace("J", "1")
            text = text.replace("j", "1")
            date_3 = re.search(r'\d{2}:\d{2}:\d{4}', text)
            date_1 = re.search(r'\d{2}/\d{2}/\d{4}', text)
            date_2 = re.search(r'\d{2}-\d{2}-\d{4}', text)
            date_4 = re.search(r'\d{2}/\d{6}', text)
            date_5 = re.search(r'\d{4}/\d{4}', text)
            if date_1 != None:
                currentKey = "birthday"
                objectResult[currentKey] = date_1.group()
            if date_2 != None:
                currentKey = "birthday"
                objectResult[currentKey] = date_2.group()
            if date_3 != None:
                currentKey = "birthday"
                objectResult[currentKey] = date_3.group()
                objectResult[currentKey] = objectResult[currentKey].replace(
                    ":", "-")
            if date_4 != None:
                currentKey = "birthday"
                objectResult[currentKey] = date_4.group()[:5] + \
                    "/" + date_4.group()[5:]
            if date_5 != None:
                currentKey = "birthday"
                objectResult[currentKey] = date_5.group()[:3] + \
                    "/" + date_5.group()[3:]
        if line.find("Que") != -1 or line.find("Qué") != -1 or line.find("quan") != -1 or line.find("quá") != -1 or line.find("Quê") != -1:
            currentKey = "hometown"
            if sum(1 for c in line if c.isupper()) >= 2:
                start_index = find_index_upper(line)
                text = line[start_index-1:]
                if sum(1 for c in line if c.isupper()) <= 5:
                    text += ", " + lines[i+1]
                text = re.sub(
                    '[^a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ ]+', '', text)
                text = text.strip()
                text = text.replace("Huỹện", "Huyện ")
                objectResult[currentKey] = text
                i += 1
            else:
                # text = lines[i-1]
                text = lines[i+1]
                text = re.sub(
                    '[^a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ ]+', '', text)
                text = text.strip()
                objectResult[currentKey] = text
                i += 1
        if line.find("Nơi") != -1 or line.find("Noi") != -1 or line.find("thường") != -1 or line.find("thương") != -1 or line.find("thưởng") != -1 or line.find("tru") != -1 or line.find("trú") != -1:
            currentKey = "address"
            text = line.replace("ĐKHK", "")
            text = text.replace("DKHK", "")
            if sum(1 for c in line if c.isupper()) >= 2:
                start_index = find_index_upper(text)
                text = text[start_index-1:]
                text += ", " + lines[i+1]
                text = re.sub(
                    '[^a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ ]+', '', text)
                text = text.strip()
                objectResult[currentKey] = text
                break
            else:
                text = lines[i-1]
                text += ", "+lines[i+1]
                text = re.sub(
                    '[^a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ ]+', '', text)
                text = text.strip()
                objectResult[currentKey] = text
                break
    if "address" in objectResult:
        objectResult["address"] = address_correction.correction(
            objectResult["address"].lower())[0]
    if "hometown" in objectResult:
        objectResult["hometown"] = address_correction.correction(
            objectResult["hometown"].lower())[0]
    # if "fullname" in objectResult:
    #     objectResult["fullname"] = fullname_correction.correction(
    #         objectResult["fullname"].lower())[0]
    return objectResult
    # for index, line in enumerate(lines):
