import os
import cv2
import glob
import numpy as np


def load_data_small():
    """
        This function loads images form the path: 'data/data_small' and return the training
        and testing dataset. The dataset is a list of tuples where the first element is the 
        numpy array of shape (m, n) representing the image the second element is its 
        classification (1 or 0).

        Parameters:
            None

        Returns:
            dataset: The first and second element represents the training and testing dataset respectively
    """

    # Begin your code (Part 1-1)
    # Define paths for face and non-face images
    train_face_path = 'data/data_small/train/face/*.pgm'
    train_nonface_path = 'data/data_small/train/nonface/*.pgm'
    test_face_path = 'data/data_small/test/face/*.pgm'
    test_nonface_path = 'data/data_small/test/nonface/*.pgm'

    # Load training dataset
    train_data = []
    for path, label in [(train_face_path, 1), (train_nonface_path, 0)]:
        for filename in glob.glob(path):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            train_data.append((img, label))

    # Load testing dataset
    test_data = []
    for path, label in [(test_face_path, 1), (test_nonface_path, 0)]:
        for filename in glob.glob(path):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            test_data.append((img, label))

    # End your code (Part 1-1)

    return train_data, test_data


def load_data_FDDB(data_idx="01"):
    """
        This function generates the training and testing dataset  form the path: 'data/data_small'.
        The dataset is a list of tuples where the first element is the numpy array of shape (m, n)
        representing the image the second element is its classification (1 or 0).
        
        In the following, there are 4 main steps:
        1. Read the .txt file
        2. Crop the faces using the ground truth label in the .txt file
        3. Random crop the non-faces region
        4. Split the dataset into training dataset and testing dataset
        
        Parameters:
            data_idx: the data index string of the .txt file

        Returns:
            train_dataset: the training dataset
            test_dataset: the testing dataset
    """

    with open("data/data_FDDB/FDDB-folds/FDDB-fold-{}-ellipseList.txt".format(data_idx)) as file:
        line_list = [line.rstrip() for line in file]

    # Set random seed for reproducing same image croping results
    np.random.seed(0)

    face_dataset, nonface_dataset = [], []
    line_idx = 0

    # Iterate through the .txt file
    # The detail .txt file structure can be seen in the README at https://vis-www.cs.umass.edu/fddb/
    while line_idx < len(line_list):
        img_gray = cv2.imread(os.path.join("data/data_FDDB", line_list[line_idx] + ".jpg"), cv2.IMREAD_GRAYSCALE)
        num_faces = int(line_list[line_idx + 1])

        # Crop face region using the ground truth label
        face_box_list = []
        for i in range(num_faces):
            # Here, each face is denoted by:
            # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
            coord = [int(float(j)) for j in line_list[line_idx + 2 + i].split()]
            x, y = coord[3] - coord[1], coord[4] - coord[0]
            w, h = 2 * coord[1], 2 * coord[0]

            left_top = (max(x, 0), max(y, 0))
            right_bottom = (min(x + w, img_gray.shape[1]), min(y + h, img_gray.shape[0]))
            face_box_list.append([left_top, right_bottom])
            # cv2.rectangle(img_gray, left_top, right_bottom, (0, 255, 0), 2)

            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()
            face_dataset.append((cv2.resize(img_crop, (19, 19)), 1))

        line_idx += num_faces + 2

        # Random crop N non-face region
        # Here we set N equal to the number of faces to generate a balanced dataset
        # Note that we have alreadly save the bounding box of faces into `face_box_list`,
        # you can utilize it for non-face region cropping
        for i in range(num_faces):
            # Begin your code (Part 1-2)
            while True:
                intersects = False
                img_h, img_w = img_gray.shape

                # Randomly generate non-face bounding box
                # Ensure that the non-face bounding box does not intersect with any face bounding box

                # Generate random coordinates for non-face region
                x1 = np.random.randint(0, img_w - 19)  # Left coordinate of the bounding box
                y1 = np.random.randint(0, img_h - 19)  # Top coordinate of the bounding box
                x2 = x1 + 19  # Right coordinate of the bounding box
                y2 = y1 + 19  # Bottom coordinate of the bounding box

                # Check if the non-face bounding box intersects with any face bounding box
                for face_box in face_box_list:
                    if (x1 < face_box[1][0] and x2 > face_box[0][0] and
                            y1 < face_box[1][1] and y2 > face_box[0][1]):
                        intersects = True
                        break

                if not intersects:
                    img_crop = img_gray[y1:y2, x1:x2].copy()
                    break

            # End your code (Part 1-2)
            nonface_dataset.append((cv2.resize(img_crop, (19, 19)), 0))

        # cv2.imshow("windows", img_gray)
        # cv2.waitKey(0)

    # train test split
    num_face_data, num_nonface_data = len(face_dataset), len(nonface_dataset)
    SPLIT_RATIO = 0.7

    train_dataset = face_dataset[:int(SPLIT_RATIO * num_face_data)] + nonface_dataset[
                                                                      :int(SPLIT_RATIO * num_nonface_data)]
    test_dataset = face_dataset[int(SPLIT_RATIO * num_face_data):] + nonface_dataset[
                                                                     int(SPLIT_RATIO * num_nonface_data):]
    return train_dataset, test_dataset


def create_dataset(data_type):
    if data_type == "small":
        return load_data_small()
    else:
        return load_data_FDDB()
