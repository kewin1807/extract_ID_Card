# Import the necessary packages
import numpy as np
import cv2


def translate(image, x, y):
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Return the translated image
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # If the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
    return rotated


def resize(image, width=None, height=None, inter=cv2.INTER_CUBIC):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def group_box_row(img, rects, boxes):
    group = {}
    centers = [get_center(rect) for rect in rects]
    row = 1
    tmp = centers[0][1]
    rected = []
    rected = np.int0(np.array(boxes[0]))
    group[str(row)] = np.int0(np.array(boxes[0]))
    for (center, rect, boxe) in (zip(centers, rects, boxes)):
        distance = abs(center[1] - tmp)
        # box = cv2.boxPoints(rect)
        cnt = np.int0(np.array(boxe))
        if distance <= 20:
            rected = np.append(rected, cnt, axis=0)
            boxes_rect = np.array(rected)

            boxes_rect = boxes_rect.reshape(-1, 2)
            minBox = cv2.minAreaRect(np.array(boxes_rect))
            tmp = get_center(minBox)[1]
            group[str(row)] = np.append(group[str(row)], cnt, axis=0)
        else:
            tmp = center[1]
            row += 1
            group[str(row)] = cnt
            rected = []
            rected = cnt

    return group


def get_center(rect):
    # M = cv2.moments(contour)
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])

    (cX, cY), (w, h), angle = rect  # cv2.minAreaRect(cnt)
    return int(cX), int(cY)


def fix_rect(rect, padding=0):
    center, (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90
    return (center, (w + 2*padding, h+2*padding), angle)
# def check_rotate(rect)
