import argparse
import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import cfg.config as config
import dataset_loader
import levelnet
from tools import *
from train_utils import *


def train(isLoad=True, e_epoch=200):
    cfg = config.get_cfg_defaults()
    log = Logs()
    train_data_flow = dataset_loader.Data_flow(cfg.TRAIN.batch_size,
                                               cfg.TRAIN.raw_data_file,
                                               cfg.img_dir,
                                               [cfg.img_h, cfg.img_w],
                                               cfg.out_features,
                                               train=True)
    val_data_flow = dataset_loader.Data_flow(cfg.VALID.batch_size,
                                             cfg.VALID.raw_data_file,
                                             cfg.img_dir,
                                             [cfg.img_h, cfg.img_w],
                                             cfg.out_features,
                                             train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.write("device " + str(device))
    model = levelnet.Model().to(device)
    if cfg['TRAIN'].OPTIMIZER == 'adam':
        log.write("adam")
        print('adam')
        optimizer = optim.Adam(model.parameters(), lr=cfg['TRAIN'].lr, amsgrad = True)
    elif cfg['TRAIN'].OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg['TRAIN'].lr,
                              momentum=cfg['TRAIN'].momentum).to(device)
        clr = cyclical_lr(cfg['TRAIN'].step_size)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    elif cfg['TRAIN'].OPTIMIZER == 'adaw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg['TRAIN'].lr)
    else:
        assert False, 'provide a valid optimiszer'
    if cfg["TRAIN"].loss == "J_MSE":
        criterion = J_MSELoss()
        log.write("loss : J_MSE")
    elif cfg["TRAIN"].loss == "Custom_MSELoss":
        criterion = Custom_MSELoss()
    elif cfg["TRAIN"].loss == "MSE":
        criterion = nn.MSELoss(size_average=False)
    elif cfg["TRAIN"].loss == 'AdaptiveWingLoss':
        print('adaptive')
        criterion = AdaptiveWingLoss()
    else:
        assert "Not a Valid loss"
    log.write('lr :' + str(cfg['TRAIN'].lr))
    summary(model, (3, 256, 256))
    model_filename = cfg.load_model_from
    if isLoad and os.path.isfile(model_filename):
        checkpoint = torch.load(model_filename)
        c_epoch = checkpoint['epoch']
        total_time = checkpoint['total_time']
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.write(" model successfully loaded from last epoch:  {}".format(c_epoch))
    else:
        print("starting from epoch 1")
        log.write("starting from epoch 1")
        c_epoch = 0
        total_time = 0
        best_val_loss = 2 * 31 - 1

    ##### staer training
    itr_ = train_data_flow.data_len / cfg['TRAIN'].batch_size

    for epoch in range(c_epoch, e_epoch):
        start_time = time.time()
        train_acc_avg = 0
        val_acc_avg = 0
        avg_train_loss = 0
        avg_val_loss = 0

        for i in range(0, math.floor(itr_)):
            input, targets = train_data_flow.load_next_batch()
            if input.shape[0] == 1:
                continue
            output = model.forward(input.to(device))
            loss = criterion(output, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l = loss.item()
            train_acc = cal_acc(get_preds(targets.detach().cpu().numpy()),
                                get_preds(output.detach().cpu().numpy()))
            train_acc_avg += train_acc
            log.write(str(i) + " loss :" + str(l) + " accuracy : " +
                      str(train_acc / targets.shape[0]))
            avg_train_loss += l

            if i % 20 == 0:
                log.save()
                if i == 0:
                    get_gpu_memory()

        ### val batches
        with torch.no_grad():
            for i in range(0, math.floor(val_data_flow.data_len / cfg.VALID.batch_size)):
                input, targets = val_data_flow.load_next_batch()
                if input.shape[0] == 1:
                    continue
                output = model.forward(input.to(device))
                loss = criterion(output, targets.to(device))
                avg_val_loss += loss.item()
                val_acc = cal_acc(get_preds(targets.detach().cpu().numpy()),
                                  get_preds(output.detach().cpu().numpy()))
                val_acc_avg += val_acc
                log.write(str(i) + "val loss:  " + str(avg_val_loss) +
                          "  val_acc:  " + str(val_acc / targets.shape[0]))
                if i % 20 == 0:
                    log.save()

        if best_val_loss > avg_val_loss:
            torch.save({
                'epoch': epoch + 1,
                'model': 'levelnet',
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'total_time': total_time
            }, cfg.best_model_filename)
            log.write('best model save -- ' + str(cfg.best_model_filename))

        best_val_loss = min(best_val_loss, avg_val_loss)
        total_time += time.time() - start_time

        torch.save({
            'epoch': epoch + 1,
            'model': 'levelnet',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'total_time': total_time
        }, cfg.save_model_to)
        log.write('model save -- ' + str(cfg.save_model_to))
        results = 'epochs:  {}   avg_loss:  {}     val_loss:  {} train_accuracy:   {}  val_accuracy: {} time on ' \
                  'epoch:  {} ' \
            .format(epoch, avg_train_loss / train_data_flow.data_len,
                    avg_val_loss / val_data_flow.data_len,
                    train_acc_avg / train_data_flow.data_len,
                    val_acc_avg / val_data_flow.data_len,
                    time.time() - start_time
                    )
        print(results)
        with open("log/train_log.txt", 'a+') as f:
            f.write(results + '\n')

        log.write(results)
        log.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nLoad', action='store_false', default=True,
                        dest='Load', help='load saved model')
    args = vars(parser.parse_args())
    train(args['Load'])
