import json
import math
import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np

joint_data = {}
start = 0
j_data = None

def get_json_data(isTrain=True):
    global j_data
    if isTrain:
        with open("dataset/annot/train.json", "r") as file:
            j_data = None
            j_data = json.load(file)
    else:
        with open("dataset/annot/valid.json", "r") as file:
            j_data = None
            j_data = json.load(file)

def read_saved_annot(isTrain):
    global start, joint_data
    if isTrain:
        if os.path.exists("dataset/joint_train_data.npy"):
            joint_data = np.load("dataset/joint_train_data.npy", allow_pickle=True).item()
            start = joint_data["processing_till"]
        else:
            joint_data = {}
            print("joint_data.npy not found")
    else:
        if os.path.exists("dataset/joint_val_data.npy"):
            joint_data = np.load("dataset/joint_val_data.npy", allow_pickle=True).item()
            start = joint_data["processing_till"]
        else:
            joint_data = {}
            print("joint_data.npy not found")

def save_joint_data(isTrain):
    if isTrain:
        np.save("dataset/joint_train_data.npy", joint_data)
    else:
        np.save("dataset/joint_val_data.npy", joint_data)

def transform(img, target_size, points=None, name='test.jpg', num=-1, save_img=True):
    global joint_data
    if img is None or img.shape[0] <= 0 or img.shape[1] <= 0:
        print('assertion error if img is None or img.shape[0] <= 0 or img.shape[1] <= 0')
        return
    h, w, _ = img.shape
    sf = float(min(target_size) / max(h, w))
    new_size = [math.floor(h * sf), math.floor(w * sf)]
    img = cv2.resize(img, (max(new_size[1], 1), max(new_size[0], 1)), interpolation=cv2.INTER_LINEAR)
    padding = [math.floor((target_size[0] - new_size[0]) / 2), math.floor((target_size[1] - new_size[1]) / 2)]
    new_img = np.zeros([target_size[0], target_size[1], 3])
    new_img[padding[0]:(padding[0] + new_size[0]), padding[1]:(padding[1] + new_size[1])] = img
    if points is not None:
        points[:, 0] = points[:, 0] * sf + padding[1]
        points[:, 1] = points[:, 1] * sf + padding[0]
    if num == -1:
        file_name = name
    else:
        file_name = "{}_{}.jpg".format(name.split(".")[0], num)
    if save_img:
        if cv2.imwrite("dataset/images_256/"+file_name,new_img):
          print("file saved   "+"dataset/images_256/"+file_name)
          joint_data[file_name] = points
        else:
          print("failede to save file "+"dataset/images_256/"+file_name)

def crop_img(img, j, joints_vis, name="test.jpg", num=0):
    j = np.array(j)
    if j.ndim != 2:
        print(" not 2d points")
        return
    h, w, dim = img.shape
    x_min, x_max, y_min, y_max = 2**32 , -2**32, 2**32 , -2*32
    for i,p in enumerate(j):
        if int(joints_vis[i])==1:
            x_min, x_max, y_min, y_max = min(p[0],x_min), max(p[0],x_max), min(p[1],y_min), max(p[1],y_max)
    x_pad, y_pad = math.floor((x_max - x_min) * .10), math.floor((y_max - y_min) * .10)
    ul = np.array([max(0, math.floor(x_min - x_pad)), max(0, math.floor(y_min - y_pad))])
    br = np.array([min(w, math.floor(x_max + x_pad)), min(h, math.floor(y_max + y_pad))])
    new_img = img[ul[1]:br[1], ul[0]:br[0]]
    for i in range(len(j)):
        if int(joints_vis[i])==1:
            j[i][0] -= ul[0]
            j[i][1] -= ul[1]
        else:
            j[i][0] = -1
            j[i][0] = -1
    transform(new_img, [256, 256], points=j, name=name, num=num)

def create_dataset(isTrain=True):
    global start, j_data
    get_json_data(isTrain)
    read_saved_annot(isTrain)
    for k in range(start,len(j_data)):
        data = j_data[k]
        if os.path.exists("dataset/images/" + data['image']):
            img = cv2.imread("dataset/images/" + data['image'])
        else:
            print(str(k) + "    img_not found  ", "dataset/images/" + data['image'])
            joint_data["processing_till"] = k
            if k % 50 == 0 and k != 0:
                save_joint_data(isTrain)
                with open("log/process.txt", "w+") as f:
                    f.write("image processed " + str(k) + "\n")
            continue
        crop_img(img.copy(), data["joints"], data["joints_vis"], data["image"],num=k)
        joint_data["processing_till"] = k
        # os.remove("dataset/images/" + data['img_paths'])
        if k % 50 == 0 and k != 0:
            save_joint_data(isTrain)
            with open("process.txt","w+") as fi:
                fi.write("image processed " + str(k) + "\n")

if __name__ == '__main__':
    create_dataset(isTrain=True)
    print("\n\n\n\n\n\n\n\n\n\n\n\n=========================")
    create_dataset(isTrain=False)
