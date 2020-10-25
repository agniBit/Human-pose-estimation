import tools
import matplotlib.pyplot as plt
import dataset_loader
import cfg.config as config
import torch
cfg = config.get_cfg_defaults()
val_data_flow = dataset_loader.Data_flow(5,
                                         cfg.TRAIN.raw_data_file,
                                         cfg.img_dir,
                                         [cfg.img_h, cfg.img_w],
                                         cfg.out_features,
                                         train=True)
input, targets = val_data_flow.load_next_batch()
tools.show_targets(input.cpu().permute(0, 2, 3, 1).numpy(),targets.cpu().numpy(),tools.get_preds(targets.numpy()))
print(tools.cal_acc(tools.get_preds(targets.numpy()),
                    tools.get_preds(targets.numpy()))+10)

# class struct:
#     def __init__(self):
#         self.x = []
#         self.y = []
#
#
# epoch = struct()
# train_l = struct()
# val_l = struct()
# train_a = struct()
# val_a = struct()
# time = struct()
# with open('/home/agni/Downloads/train_log(1).txt') as f:
#     for i, line in enumerate(f):
#         l = line.split(" ")
#         print(i,line)
#         if i >= 91:
#             epoch.x.append(i)
#             train_l.x.append(i)
#             val_l.x.append(i)
#             train_a.x.append(i)
#             val_a.x.append(i)
#             time.x.append(i)
#             epoch.y.append(int(float(l[2]))*1000)
#             train_l.y.append(int(float(l[7])*10**8))
#             val_l.y.append(int(float(l[14])*10**8))
#             train_a.y.append(float(l[18]))
#             val_a.y.append(float(l[21]))
#             time.y.append(int(float(l[26])))
#
# plt.figure()
# # # plt.plot(epoch.x, epoch.y, label="epoch")
# plt.plot(train_l.y, label="train loss")
# plt.plot(val_l.y, label="val loss")
# # # plt.plot(train_a.x, train_a.y, label="train acc")
# # # plt.plot(val_a.x, val_a.y, label="val accu")
# # # plt.plot(time.x, time.y , label="time / epoch")
# # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.show()
#
# # plt.figure()
# # plt.plot(epoch.y, label="epoch")
# # plt.plot(train_a.y, label="train acc")
# # plt.plot(val_a.y, label="val accu")
# # plt.plot(time.x, time.y , label="time / epoch")
# # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# # plt.show()
#
#
# # plt.figure()
# # plt.plot(epoch.y, label="epoch")
# # # plt.plot(train_a.y, label="train acc")
# # # plt.plot(val_a.y, label="val accu")
# # plt.plot(time.y , label="time / epoch")
# # # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# # plt.show()

#
# import dataset_loader
# import cfg.config as config
#
# cfg = config.get_cfg_defaults()
# train_data_flow = dataset_loader.Data_flow(cfg.TRAIN.batch_size,
#                                            cfg.TRAIN.raw_data_file,
#                                            cfg.img_dir,
#                                            [cfg.img_h, cfg.img_w],
#                                            cfg.out_features)
#
# import tools
# import matplotlib.pyplot as plt
# import torch
#
# for i in range(3):
#     img, targets = train_data_flow.load_next_batch()
#     p_tar = tools.get_preds(targets.numpy())
#     for i in range(targets.shape[0]):
#         plt.figure()
#         plt.scatter(p_tar[i][:, 1], p_tar[i][:, 0], c='blue', label='pred from targest')
#         plt.imshow(img[i].permute(1, 2, 0))
#         print(max(img[i].numpy().flatten()))
#         print(min(img[i].numpy().flatten()))
#         plt.show()
