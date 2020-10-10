import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime

def get_gpu_memory():
    import nvidia_smi
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Used GPU memory: {}%".format((info.used*100)//info.total))
    nvidia_smi.nvmlShutdown()


class Logs():
    def __init__(self, file_name='log/logs.txt'):
        self.file_name = file_name
        self.f = open(self.file_name, 'a+')

    def save(self):
        self.f.close()

    def write(self, line):
        if self.f.closed:
            self.f = open(self.file_name, 'a+')
        self.f.write(str(datetime.datetime.now()) + '::  ' + line + '\n')


def cal_acc(targets, output):
    avg = np.sum(
        np.sqrt(np.square(output[:, :, 0] - targets[:, :, 0]) + np.square(output[:, :, 1] - targets[:, :, 1]))) / \
          targets.shape[1]
    return avg


def get_preds(scores,img_h=256,img_w=256):
    # scores = torch.Tensor.cpu(scores).detach().numpy()
    scores = np.reshape(scores, (scores.shape[0], scores.shape[1], -1))
    max_val, indx = np.max(scores, -1), np.argmax(scores, -1)
    preds = np.zeros((scores.shape[0], scores.shape[1], 2), int)
    preds[:, :, 0] = indx[:, :] // img_h
    preds[:, :, 1] = indx[:, :] % img_w
    return preds


def show_targets(img, targets, points, isShow=False, isSave=True):
    columns = 4
    rows = 5
    pred = get_preds(targets)
    for k in range(len(targets)):
        plt.figure(figsize=(15, 15))
        print(pred[k])
        print(points[k])
        for i in range(0, 17):
            if i == 16:
                plt.subplot(rows, columns, i + 1)
                plt.imshow(img[k])
                plt.scatter(pred[k][:, 0], pred[k][:, 1])
            else:
                plt.subplot(rows, columns, i + 1)
                plt.imshow(targets[k][i])
        if isSave:
            plt.savefig('output/targets_{}.png'.format(str(k)))
        if isShow:
            plt.show()
