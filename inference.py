import dataset
import cv2
import argparse
import numpy as np
from PIL import Image
import torch
import tools
import PIL
from torchvision import transforms
import matplotlib.pyplot as plt
import levelnet
import dataset_loader
import cfg.config as config
import torch

def infer(img_path,save_output):
    cfg = config.get_cfg_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = levelnet.Model()
    model.load_state_dict(torch.load('backup/best_checkpoints_v5_mse.pth')["state_dict"])
    model.to(device)
    model.eval()
    img = cv2.imread(img_path)
    img = dataset.transform(img, [256, 256], save_img=True , save_file_name = 'out.jpg')
    image = PIL.Image.open('out.jpg')
    trans = transforms.ToTensor()
    img = trans(image)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model.forward(img.to(device))
    i = 0
    pred = tools.get_preds(np.array(output.cpu()), img_h=256, img_w=256)
    plt.figure()
    plt.imshow(img[i].permute(1, 2, 0))
    plt.scatter(pred[i][:, 0], pred[i][:, 1])
    plt.savefig(save_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Human pose estimation")
    parser.add_argument("--input", type=str, default='', help="video source")
    parser.add_argument("--save", type=str, default='', help="save output")
    args = parser.parse_args()
    infer(args.input,args.save)
