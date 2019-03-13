import os
import numpy as np


def vgg_testing_set():
    num_test_img = 50
    data_path = "/data/disk2/ALL/gsk/VGG_Face2/test/"

    train_path_set = []
    train_label_set = []
    test_path_set = []
    test_label_set = []
    num_class = -1
    for class_id in os.listdir(data_path):
        img_list = []
        class_path = os.path.join(data_path, class_id)
        num_class += 1
        for img_id in os.listdir(class_path):
            img_path = os.path.join(class_path, img_id)
            img_list.append(img_path)
        np.random.shuffle(img_list)
        for i in img_list[0:num_test_img]:
            test_path_set.append(i)
            test_label_set.append(num_class)
        for i in img_list[num_test_img:]:
            train_path_set.append(i)
            train_label_set.append(num_class)
    np.savez('./train_set.npz', path=train_path_set, label=train_label_set)
    np.savez('./test_set.npz', path=test_path_set, label=test_label_set)


def vgg_training_set():
    data_path = "/data/disk2/ALL/gsk/VGG_Face2/train/"

    train_path_set = []
    train_label_set = []
    num_class = -1
    for class_id in os.listdir(data_path):
        class_path = os.path.join(data_path, class_id)
        num_class += 1
        for img_id in os.listdir(class_path):
            img_path = os.path.join(class_path, img_id)
            train_path_set.append(img_path)
            train_label_set.append(num_class)
    np.savez('./transformation_train_set.npz', path=train_path_set, label=train_label_set)


vgg_training_set()
