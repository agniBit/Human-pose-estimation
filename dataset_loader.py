import os
from torchvision import transforms
import cv2
import numpy as np
import cfg.config as config
import torch
import random
import PIL
import torchvision.transforms.functional as TF

cfg = config.get_cfg_defaults()


class Data_augmentation():
    def __init__(self, train=True ,normalize=False, size=(cfg.img_h, cfg.img_w), resize=False,
                 crop=False, horizontal_flip=True, vertical_flip=False, ColorJitter =True):
        self.train = train
        self.normalize = normalize
        self.size = size
        self.resize = resize
        self.crop = crop
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.ColorJitter = ColorJitter

    def transform(self, image_path, targets=None):
        image = PIL.Image.open(image_path)
        # Resize
        if self.train:
            if self.resize:
                resize = transforms.Resize(self.size)
                image = resize(image)
                targets = resize(targets)

            # Random crop
            if self.crop:
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.size)
                image = TF.crop(image, i, j, h, w)
                targets = TF.crop(targets, i, j, h, w)

            # Random horizontal flipping
            if self.horizontal_flip:
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    targets = TF.hflip(targets)

            # Random vertical flipping
            if self.vertical_flip:
                if random.random() > 0.5:
                    image = TF.vflip(image)
                    targets = TF.vflip(targets)

            if self.ColorJitter:
                Jitter = transforms.ColorJitter(hue=.05, saturation=.05)
                image = Jitter(image)

        # Transform to tensor
        image = TF.to_tensor(image)

        if self.normalize:
            norm = transforms.Normalize(mean=cfg['dataset'].mean, std=cfg['dataset'].std)
            image = norm(image)
        return image, targets


class Data_flow(object):
    """docstring for data_flow"""

    def __init__(self, batch_size, raw_data_filename, img_dir, target_size, num_outputs , train=False):
        super(Data_flow, self).__init__()
        self.data_augmentation = Data_augmentation(train=train)
        self.batch_size = batch_size
        self.file_names = []
        self.joints = []
        self.img_dir = img_dir
        self.c_batch_num = 0
        self.target_size = target_size
        self.num_outputs = num_outputs
        self.gaussian_kernel = None
        self.size = 3
        self.sigma = 2
        self.generate_gaussian_kernel()
        self.get_filenames_joints(raw_data_filename)
        self.data_len = len(self.file_names)

    def update_kernel(self, size, sigma):
        self.size = size
        self.sigma = sigma
        self.generate_gaussian_kernel()

    def generate_gaussian_kernel(self):
        size = self.sigma * self.size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        self.gaussian_kernel = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
        self.gaussian_kernel = torch.from_numpy(self.gaussian_kernel)

    def get_filenames_joints(self, raw_data_filename):
        data = np.load(raw_data_filename, allow_pickle=True).item()
        lst = list(data.keys())
        random.shuffle(lst)
        for file_name in lst:
            if os.path.exists(os.path.join(self.img_dir, file_name)):
                self.file_names.append(file_name)
                self.joints.append(data[file_name])
            else:
                print("file not found ", file_name)

    def generate_target(self, points):
        target = torch.zeros((self.num_outputs, self.target_size[0], self.target_size[1]),
                             dtype=torch.float64)
        tmp_size = (self.sigma * self.size +1) // 2
        for k in range(self.num_outputs):
            pt = [int(points[k][1]), int(points[k][0])]
            if not (0 <= pt[0] <= self.target_size[0] and 0 <= pt[1] <= self.target_size[1]):
                continue
            ul = [pt[0] - tmp_size, pt[1] - tmp_size]
            br = [pt[0] + tmp_size + 1, pt[1] + tmp_size + 1]
            img_ul = [max(0, ul[0]), max(0, ul[1])]
            img_br = [min(self.target_size[0], br[0]), min(self.target_size[1], br[1])]
            ul = [max(0, -ul[0]), max(0, -ul[1])]
            br = [min(img_br[0], self.target_size[0]) - img_ul[0], min(img_br[1], self.target_size[1]) - img_ul[1]]
            target[k][img_ul[0]:img_br[0] - ul[0], img_ul[1]:img_br[1] - ul[1]] = self.gaussian_kernel[
                                                                                  ul[0]:br[0], ul[1]:br[1]]
        return target

    def load_next_batch(self):
        images = []
        targets = []
        i = 0
        for i in range(min(self.batch_size, self.data_len - self.c_batch_num * self.batch_size)):
            image_path = os.path.join(
                    self.img_dir, self.file_names[self.c_batch_num * self.batch_size + i]
                )
            target = self.generate_target(self.joints[self.c_batch_num * self.batch_size + i])
            t_image, t_target = self.data_augmentation.transform(image_path, target)
            images.append(t_image)
            targets.append(t_target)
        if self.c_batch_num == int(self.data_len / self.batch_size):
            self.c_batch_num = 0
        else:
            self.c_batch_num += 1
        return torch.stack(images).float(), torch.stack(targets).float()
