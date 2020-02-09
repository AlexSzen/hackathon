""" Loader for facade images"""


import numpy as np
from torch.utils.data import Dataset
import PIL
import os
import torchvision.transforms as transforms
import torch
import copy
class cdl_dataset(Dataset):

    ''' Dataset for facade images from CMP database for now'''

    def __init__(self, data_path, train_split = 0.8, split = "train"):


        self.split = split

        images_files = os.listdir(data_path)

        self.data_path = data_path

        # Inputs are jpg files
        self.inputs = [file for file in images_files if ".jpg" in file]
        self.inputs.sort()
        # Outputs are png files
        self.outputs = [file for file in images_files if ".png" in file]
        self.outputs.sort()


        # Split
        if split == "train":
            self.inputs = self.inputs[:int(train_split*len(self.inputs))]
            self.outputs = self.outputs[:int(train_split*len(self.outputs))]
        else:
            self.inputs = self.inputs[int(train_split*len(self.inputs)):]
            self.outputs = self.outputs[int(train_split*len(self.outputs)):]


    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        input_file = self.inputs[index]
        output_file = self.outputs[index]

        in_arr = torch.from_numpy(np.asarray(PIL.Image.open(os.path.join(self.data_path, input_file))))
        out_arr_tmp = (np.asarray((PIL.Image.open(os.path.join(self.data_path, output_file) ) ) ))


        # Convert to binary mask for window/not window. windows have label 3 in dataset
        h = np.where(out_arr_tmp == 3)
        out_arr = np.zeros(out_arr_tmp.shape)
        out_arr[h] = 1

        out_arr = torch.from_numpy(out_arr).long()

        in_arr = in_arr.permute(2,0,1).double()/255


        res = 224
        channels = 3
        in_arr_crop = np.zeros((3, res, res))
        out_arr_crop = np.zeros((res, res))
        # Crop a random 224 patch for train
        if self.split =="train":


            if in_arr.shape[2] >= res:
                hcrop_start = np.random.randint(0, high = (in_arr.shape[2] - res) )
                hcrop_end = hcrop_start + res
            else:
                hcrop_start = 0
                hcrop_end = in_arr.shape[2]

            if in_arr.shape[1] >= res:
                vcrop_start = np.random.randint(0, high = (in_arr.shape[1] - res) )
                vcrop_end = vcrop_start + res
            else:
                vcrop_start = 0
                vcrop_end = in_arr.shape[1]


            if in_arr.shape[2] and in_arr.shape[1] >= res:
                in_arr_crop = in_arr[:,vcrop_start : vcrop_end, hcrop_start : hcrop_end]
                out_arr_crop = out_arr[vcrop_start : vcrop_end, hcrop_start : hcrop_end]

            if in_arr.shape[1] >= res and in_arr.shape[2] < res:
                in_arr_crop[:,:,0:in_arr.shape[2]] = in_arr[:,vcrop_start : vcrop_end, hcrop_start : hcrop_end]
                out_arr_crop[:,0:in_arr.shape[2]] = out_arr[vcrop_start : vcrop_end, hcrop_start : hcrop_end]

            if in_arr.shape[1] < res and in_arr.shape[2] >= res:
                in_arr_crop[:,0:in_arr.shape[1],:] = in_arr[:,vcrop_start : vcrop_end, hcrop_start : hcrop_end]
                out_arr_crop[0:in_arr.shape[1],:] = out_arr[vcrop_start : vcrop_end, hcrop_start : hcrop_end]

            if in_arr.shape[1] < res and in_arr.shape[2] < res:
                in_arr_crop[:,0:in_arr.shape[1],0:in_arr.shape[2]] = in_arr[:,vcrop_start : vcrop_end, hcrop_start : hcrop_end]
                out_arr_crop[0:in_arr.shape[1],0:in_arr.shape[2]] = out_arr[vcrop_start : vcrop_end, hcrop_start : hcrop_end]

        # Crop center 256 patch for val
        if self.split == "val":

            if in_arr.shape[2] >= res:
                hcrop_start = int(in_arr.shape[2]/2) - int(res/2)
                hcrop_end = hcrop_start + res

            else:
                hcrop_start = 0
                hcrop_end = int_arr.shape[2]

            if in_arr.shape[1] >= res:
                vcrop_start = int(in_arr.shape[1]/2) - int(res/2)
                vcrop_end = vcrop_start + res
            else:
                vcrop_start = 0
                vcrop_end = in_arr.shape[1]

            in_arr_crop = in_arr[:,vcrop_start : vcrop_end, hcrop_start : hcrop_end]
            out_arr_crop = out_arr[vcrop_start : vcrop_end, hcrop_start : hcrop_end]

        # normalize manually
        in_arr_crop[0,:,:] = (in_arr_crop[0,:,:] - 0.485)/ 0.229
        in_arr_crop[1,:,:] = (in_arr_crop[1,:,:] - 0.456)/ 0.224
        in_arr_crop[2,:,:] = (in_arr_crop[2,:,:] - 0.406)/ 0.225





        return (in_arr_crop), (out_arr_crop)
