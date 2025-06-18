import os
import cv2
import random
import numpy as np
from shutil import copyfile, move
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
from datasets import custom_transforms as tr

# Rand rot flip is a data augumentation from transUnet
# It can rot and crop th imanges.
# def random_rot_flip(image, label):
#     k = np.random.randint(0, 4)
#     image = np.rot90(image, k)
#     label = np.rot90(label, k)
#     axis = np.random.randint(0, 2)
#     image = np.flip(image, axis=axis).copy()
#     label = np.flip(label, axis=axis).copy()
#     return image, label

def read_own_data(root_path, split='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, split + '/images')
    gt_root = os.path.join(root_path, split + '/labels')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def own_data_loader(img_path, mask_path):
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path)
    mask = np.array(mask)
    #mask[mask > 0] = 1
    x,y,z = mask.shape
    mask1 = np.zeros((x,y))
    r = mask[:,:,0]
    g = mask[:,:,1]
    mask1[r == 128] = 1
    mask1[g == 128] = 2
    mask1 = Image.fromarray(np.uint8(mask1))
    return img, mask1


class ImageFolder(data.Dataset):

    def __init__(self, args, split='train'):
        self.args = args
        self.root = self.args.root_path
        self.split = split
        self.images, self.labels = read_own_data(self.root, self.split)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            # tr.TransUnetAugmentation(output_size=[512,512]),
            # # This is the same as TransUNet augmentation.
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.img_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.img_size),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def __getitem__(self, index):
        img, mask = own_data_loader(self.images[index], self.labels[index])
        if self.split == "train":
            sample = {'image': img, 'label': mask}
            return self.transform_tr(sample)
        elif self.split == 'test':
            img_name = os.path.split(self.images[index])[1]
            sample = {'image': img, 'label': mask}
            sample_ = self.transform_test(sample)
            sample_['case_name'] = img_name[0:-4]
            return sample_
        # return sample

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)