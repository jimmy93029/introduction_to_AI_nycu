import os
from unittest import result
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    with open(dataPath, 'r') as file:
        line_list = [line.rstrip().split() for line in file]

    line_idx = 0
    while line_idx < len(line_list):
        img_gray = cv2.imread(os.path.join("data/detect", line_list[line_idx][0]),  cv2.IMREAD_GRAYSCALE)
        num_faces = int(line_list[line_idx][1])

        # Crop face region using the ground truth label
        box_list = []
        for i in range(num_faces):
            # get boxes
            x, y = int(line_list[line_idx + 1 + i][0]), int(line_list[line_idx + 1 + i][1])
            w, h = int(line_list[line_idx + 1 + i][2]), int(line_list[line_idx + 1 + i][3])
            img_crop = cv2.resize(img_gray[y:y + h, x:x + w].copy(), (19, 19))

            # classify
            if clf.classify(img_crop) == 1:
                box_list.append(((x, y), (x + w, y + h), 1))
            else:
                box_list.append(((x, y), (x + w, y + h), 0))

        image = cv2.imread(os.path.join("data/detect", line_list[line_idx][0]))
        for left_top, right_bottom, label in box_list:
            if label == 1:
                cv2.rectangle(image, left_top, right_bottom, (0, 255, 0), 2)
            else:
                cv2.rectangle(image, left_top, right_bottom, (255, 0, 0), 2)

        line_idx += num_faces + 1

        # Show the image with face detections
        cv2.imshow('Face Detection Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # End your code (Part 4)
