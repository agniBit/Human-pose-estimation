# Model Config
from yacs.config import CfgNode as CN

_c = CN()
_c.plans = 3
_c.img_h = 256
_c.img_w = 256
_c.out_features = 16
_c.img_dir = "dataset/images_256/"
_c.save_model_to = 'backup/last_checkpoints_v5_mse.pth'
_c.load_model_from = 'backup/last_checkpoints_v5_mse.pth'
_c.best_model_filename = 'backup/last_checkpoints_v5_mse.pth'

_c.TRAIN = CN()
_c.TRAIN.OPTIMIZER = 'adam'
_c.TRAIN.lr = 0.001
_c.TRAIN.loss = 'AdaptiveWingLoss'
_c.TRAIN.step_size = 12500
_c.TRAIN.momentum = 0.9
_c.TRAIN.batch_size = 24
_c.TRAIN.SHUFFLE = True
_c.TRAIN.raw_data_file = "dataset/joint_train_data.npy"

_c.VALID = CN()
_c.VALID.batch_size = 24
_c.VALID.raw_data_file = "dataset/joint_val_data.npy"

_c.dataset = CN()
_c.dataset.mean = [0.485, 0.456, 0.406]
_c.dataset.std = [0.229, 0.224, 0.225]
_c.x = CN()
_c.x.inplans = 3
_c.x.outplans = 2*_c.out_features

_c.x_2d = CN()
_c.x_2d.inplans = _c.x.outplans
_c.x_2d.conv5plans = 16
_c.x_2d.conv3plans = 32
_c.x_2d.conv1plans = 32
_c.x_2d.out = _c.x_2d.conv5plans + _c.x_2d.conv3plans + _c.x_2d.conv1plans
_c.x_2d.reduce_to = 4*_c.out_features

_c.x_4d = CN()
_c.x_4d.inplans = _c.x_2d.reduce_to
_c.x_4d.conv5plans = 32
_c.x_4d.conv3plans = 64
_c.x_4d.conv1plans = 64
_c.x_4d.out = _c.x_4d.conv5plans +_c.x_4d.conv3plans + _c.x_4d.conv1plans
_c.x_4d.reduce_to = 4*2*_c.out_features

_c.x_8d = CN()
_c.x_8d.inplans = _c.x_4d.reduce_to
_c.x_8d.conv5plans = 64
_c.x_8d.conv3plans = 128
_c.x_8d.conv1plans = 128
_c.x_8d.out = _c.x_8d.conv5plans +_c.x_8d.conv3plans + _c.x_8d.conv1plans
_c.x_8d.reduce_to = 4*2*2*_c.out_features

_c.x_16d = CN()
_c.x_16d.inplans = _c.x_8d.reduce_to
_c.x_16d.conv5plans = 128
_c.x_16d.conv3plans = 128
_c.x_16d.conv1plans = 128
_c.x_16d.out = _c.x_16d.conv5plans +_c.x_16d.conv3plans + _c.x_16d.conv1plans
_c.x_16d.reduce_to = 4*2*2*_c.out_features

_c.x_32d = CN()
_c.x_32d.inplans = _c.x_16d.reduce_to
_c.x_32d.conv5plans = 164
_c.x_32d.conv3plans = 164
_c.x_32d.conv1plans = 164
_c.x_32d.out = _c.x_32d.conv5plans +_c.x_32d.conv3plans + _c.x_32d.conv1plans
_c.x_32d.reduce_to = 4*2*2*2*_c.out_features

_c.x_16u = CN()
_c.x_16u.inplans = _c.x_32d.reduce_to
_c.x_16u.conv5plans = 128
_c.x_16u.conv3plans = 128
_c.x_16u.conv1plans = 164
_c.x_16u.out = _c.x_16u.conv5plans +_c.x_16u.conv3plans + _c.x_16u.conv1plans
_c.x_16u.reduce_to = 4*2*2*_c.out_features

_c.x_8u = CN()
_c.x_8u.inplans = _c.x_16d.reduce_to + _c.x_16u.reduce_to
_c.x_8u.conv5plans = 64
_c.x_8u.conv3plans = 128
_c.x_8u.conv1plans = 128
_c.x_8u.out = _c.x_8u.conv5plans + _c.x_8u.conv3plans + _c.x_8u.conv1plans
_c.x_8u.reduce_to = 4*2*2*_c.out_features

_c.x_4u = CN()
_c.x_4u.inplans = _c.x_8d.reduce_to + _c.x_8u.reduce_to
_c.x_4u.conv5plans = 64
_c.x_4u.conv3plans = 64
_c.x_4u.conv1plans = 128
_c.x_4u.out = _c.x_4u.conv5plans +_c.x_4u.conv3plans + _c.x_4u.conv1plans
_c.x_4u.reduce_to = 4*2*_c.out_features

_c.x_2u = CN()
_c.x_2u.inplans = _c.x_4d.reduce_to + _c.x_4u.reduce_to
_c.x_2u.conv5plans = 16
_c.x_2u.conv3plans = 32
_c.x_2u.conv1plans = 32
_c.x_2u.out = _c.x_2u.conv5plans + _c.x_2u.conv3plans + _c.x_2u.conv1plans
_c.x_2u.reduce_to = 4*_c.out_features

_c.x_u = CN()
_c.x_u.inplans = _c.x_2u.reduce_to + _c.x_2d.reduce_to
_c.x_u.reduce_to = 4*_c.out_features

def get_cfg_defaults():
    return _c.clone()


