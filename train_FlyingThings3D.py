import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.legacy.nn  as L
from LSTM_disp import LSTMdisp
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from parsepfm import *

def data_preparation():
    #TODO: parse the data list
    data_path = '/home/xiao/Desktop/cnn-stereo/data/'
    train_list = data_path + 'FlyingThings3D_release_TRAIN.list'
    test_list =  data_path + 'FlyingThings3D_release_TEST.list'
    batch_size_test = 1
    batch_size_train = 8
    iteration_train = 1

    width = 960
    height = 640

    train_file = []
    test_file  = []

    with open(train_list) as f:
        content = f.readlines()
        train_file = [re.split('\t|\n',x)[0:3] for x in content]
    with open(test_list) as f:
        content = f.readlines()
        test_file = [re.split('\t|\n',x)[0:3] for x in content]

    num_train_sample = len(train_file)
    num_test_sample  = len(test_file)

    #Training process
    for iter in range(iteration_train):
        perm_train = np.random.permutation(num_train_sample)[0:batch_size_train]
        train_left_batch = torch.Tensor(batch_size_train, 3, height,width)
        train_right_batch = torch.Tensor(batch_size_train, 3, height,width)
        # train_disp_batch = torch.tensor(batch_size_train, 1, height,width)

        # Fill in batch
        for idx in perm_train:
            left_img  = cv2.imread(data_path + train_file[idx][0])
            right_img = cv2.imread(data_path + train_file[idx][1])

            plt.imshow(left_img)
            print(left_img.shape)
            # disp_gt   = load_pfm(data_path + train_file[idx][2])
            # disp_gt   = -disp_gt

            #This image is okay for the network, we escape the implementation of crop

            #switch row and col
            # left_img = torch.Tensor(left_img).transpose(1,2)
            # left_img = torch.Tensor(right_img).transpose(1,2)
            # left_img = torch.Tensor(left_img).transpose(1,2)




data_preparation()
