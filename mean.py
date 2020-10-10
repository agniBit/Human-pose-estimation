import torch
import dataset_loader
import cfg.config as config
import torch
import numpy as np
cfg = config.get_cfg_defaults()
mean = 0.
std = 0.
train_data_flow = dataset_loader.Data_flow(cfg.TRAIN.batch_size,
                                               cfg.TRAIN.raw_data_file,
                                               cfg.img_dir,
                                               [cfg.img_h, cfg.img_w],
                                               cfg.out_features)
iter_ = int(train_data_flow.data_len/cfg.TRAIN.batch_size)
for i in range(iter_):
  images,_,_ = train_data_flow.load_next_batch()
  print(images[0])
  # images = images.cpu()
  # batch_samples = images.shape[0] # batch size (the last batch can have smaller size!)
  # images = images.view(batch_samples, images.shape[1], -1)
  # mean += images.mean(2).sum(0)
  # std += images.std(2).sum(0)
  # if i%10==0:
  #   print(i)
  break

print("pre", mean,std)
mean /= train_data_flow.data_len
std /= train_data_flow.data_len
print(mean,std)