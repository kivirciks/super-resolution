import argparse
from PIL import Image, ImageOps
import os
import numpy as np
from math import floor
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
import random

def tensor2img(tensor):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    tensor = np.squeeze(tensor)
    tensor = np.moveaxis(tensor, 0, 2)
    tensor = (tensor * 255)  # + 0.5  # ? add 0.5 to rounding
    tensor = tensor.clip(0, 255).astype(np.uint8)

    img = Image.fromarray(tensor)
    return img

def mkdir(directory, mode=0o777):
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.chmod(directory, mode=mode)

def dir_exists(directory):
    return os.path.exists(directory)

def crop(img_arr, block_size):
    h_b, w_b = block_size
    v_splited = np.vsplit(img_arr, img_arr.shape[0]//h_b)
    h_splited = np.concatenate(
        [np.hsplit(col, img_arr.shape[1]//w_b) for col in v_splited], 0)
    return h_splited

def generate_patches(src_path, files, set_path, crop_size, img_format, max_patches):
    img_path = os.path.join(src_path, files)
    img = Image.open(img_path).convert('RGB')

    name, _ = files.split('.')
        
    filedirb = os.path.join(set_path, 'b')
    if not dir_exists(filedirb):
        mkdir(filedirb)

    img = np.array(img)
    h, w = img.shape[0], img.shape[1]

    if crop_size == None:
        img = np.copy(img)
        img_patches = np.expand_dims(img, 0)
    else:
        rem_h = (h % crop_size[0])
        rem_w = (w % crop_size[1])
        img = img[:h-rem_h, :w-rem_w]
        img_patches = crop(img, crop_size)

    # print('Cropped')

    n = 0

    for i in range(min(len(img_patches), max_patches)):
        img_grey = Image.fromarray(img_patches[i]).convert('RGB')
        
        img_grey.save(
            os.path.join(filedirb, '{}_{}.{}'.format(name, i, img_format))
        )

        n += 1

    return n

def main(target_dataset_folder, dataset_path, crop_size, img_format, max_patches, max_n):
    print('[ Creating Dataset ]')
    print('Crop Size : {}'.format(crop_size))
    print('Target       : {}'.format(target_dataset_folder))
    print('Dataset       : {}'.format(dataset_path))
    print('Format    : {}'.format(img_format))
    print('Max N    : {}'.format(max_n))

    src_path = dataset_path
    if not dir_exists(src_path):
        raise(RuntimeError('Source folder not found, please put your dataset there'))

    set_path = target_dataset_folder

    mkdir(set_path)

    img_files = os.listdir(src_path)

    max = len(img_files)
    bar = tqdm(img_files)
    i = 0
    j = 0
    for files in bar:
        k = generate_patches(src_path, files, set_path,
                             crop_size, img_format, max_patches)

        bar.set_description(desc='itr: %d/%d' % (
            i, max
        ))

        j += k

        if j >= max_n:
            # Stop the process
            print('Dataset count has been fullfuled')
            break

        i += 1

    print('Dataset Created')

main('dataset/train', '../input/div2kembeddedgrey', [128, 128], 'PNG', 15, 10000)
