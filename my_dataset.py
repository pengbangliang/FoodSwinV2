import os
from copy import deepcopy
import torch.utils.data as data
from PIL import Image,ImageFilter
import random
import numpy as np
from torchvision import transforms

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

class VOCSegmentation(data.Dataset):
    def __init__(self, food_root, device='cuda',  transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        # 修改数据集名称 ：FoodSeg103  /  UECCOMPLETE
        root = os.path.join(food_root, "UECCOMPLETE") # 拼接根路径

        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        txt_path = os.path.join(root, "ImageSets", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        if txt_name == 'unlable.txt':
            self.mode = 'u'
            image_dir = os.path.join(root,'Images','unlable')
            self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        else:
            self.mode = 'l'
            image_dir = os.path.join(root, 'Images','img_dir','all') # 原图像

            mask_dir = os.path.join(root,'Images','ann_dir','all') # 分割结果图像

            self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
            self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]

            assert (len(self.images) == len(self.masks))
        self.transforms = transforms


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if self.mode == 'l':
            img = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.masks[index])
            if self.transforms is not None:
                img, target = self.transforms(img, target)
                # target = np.array(target)
                 # target=torch.from_numpy(target).type(torch.LongTensor)
            return img, target
        else:
            img = Image.open(self.images[index]).convert('RGB')
            mask1 = Image.fromarray(np.zeros((img.size[1], img.size[0])))
            mask2 = Image.fromarray(np.zeros((img.size[1], img.size[0])))
            img_w, img_s = deepcopy(img), deepcopy(img)

            img_s = transforms.RandomGrayscale(p=0.2)(img_s)
            img_s = blur(img_s, p=0.5)
            if self.transforms is not None:
                img_w, mask1 = self.transforms(img_w,mask1)
                img_s, mask2 = self.transforms(img_s,mask2)

            
            return [img_w,img_s]

        
        

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))

        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets




def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

