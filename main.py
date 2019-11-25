# import import_data
import numpy as np
import nibabel as nib
import tensorflow as tf
import imageio
import argparse
import os
import glob
#
# from core.data_processor import SimpleImageProcessor
# from core.data_provider import DataProvider
# from core.trainer_tf import Trainer
from models.model import SimpleTFModel
from nets_tf.unet3d import UNet3D


# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.70)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

file_path = 'D:/DT/BrainMRI/BrainMRI'
save_path = "/data/"
# import_data.import_data(file_path)


orig_image_data = np.load(file='orig_image_data.npy')
brain_image_data = np.load(file='brain_image_data.npy')



# parser = argparse.ArgumentParser()
# parser.add_argument('-ep', '--epochs', type=int, default=500, help='number of epochs')
# parser.add_argument('-bs', '--batch_size', type=int, default=10, help='batch size')
# parser.add_argument('-mbs', '--minibatch_size', type=int, default=2, help='mini-batch size')
# parser.add_argument('-ebs', '--eval_batch_size', type=int, default=1, help='mini-batch size')
# parser.add_argument('-ef', '--eval_frequency', type=int, default=1, help='frequency of evaluation within training')
# parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
# parser.add_argument('-out', '--output_path', type=str, default='results/test1/')
# args = parser.parse_args()
#
# output_path = args.output_path
#
#
# u_net = UNet3D(n_class=5, n_layer=5, root_filters=16, use_bn=True)
#
# model = SimpleTFModel(u_net, org_suffix, lab_suffix, dropout=0, loss_function={'balanced_dice': 1.}, weight_function=None)
# optimizer = tf.keras.optimizers.Adam(args.learning_rate)
#
# trainer = Trainer(model)
#
# # train test
# result = trainer.train(train_provider, validation_provider,
#                        epochs=args.epochs,
#                        batch_size=args.batch_size,
#                        mini_batch_size=args.minibatch_size,
#                        output_path=output_path,
#                        optimizer=optimizer,
#                        eval_frequency=args.eval_frequency,
#                        is_save_train_imgs=False,
#                        is_save_valid_imgs=True,
#                        is_rebuilt_path=True)
#
# # eval test & pre load test
# test_provider = DataProvider(test_set, [org_suffix, lab_suffix],
#                         is_pre_load=False,
#                         processor=processor)
# trainer.restore(output_path + '/ckpt/final')
# eval_dcit = trainer.eval(test_provider, batch_size=args.eval_batch_size)

