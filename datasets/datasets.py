import os
import numpy as np
import torch
from glob import glob
from torch.utils import data
from PIL import Image
from imageio import imread
import random
import cv2

class MVTec_Dataset(data.Dataset):
    def __init__(self,root,obj,split="train"):
        super().__init__()
        self.root = root
        self.split = split
        self.files = {}
        self.obj = obj  

        fpattern = os.path.join(self.root, f'{self.obj}/{self.split}/*/*.png')
        self.files = sorted(glob(fpattern))
        
        if self.split == "test":
            random.shuffle(self.files) # shuffle clean and artifact images

        # # Get mean of training images
        # train_images = np.asarray(list(map(imread, self.files)))
        # train_images = np.asarray(list(map(self.resize, train_images)))
        # train_images = train_images / 255
        # self.mean = np.mean(train_images)
        # self.stdev = np.std(train_images)
        
        # self.mean = train_images.astype(np.float32).mean(axis=0)
        # self.stdev = train_images.astype(np.float32).std(axis=0)

    def __len__(self):
        """__len__"""
        return len(self.files)


    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        file_paths = self.files

        # Test mode
        if self.split == "test":
            # Image
            fpath = file_paths[index]

            image = np.asarray(imread(fpath))
            image = self.normalize_images(image) #,train=False)

            torch_image = torch.from_numpy(image)
            torch_image = torch_image.type(torch.FloatTensor)  

            if (os.path.basename(os.path.dirname(file_paths[index])) == 'good'):
                binary_label = 0
                # Normal Label for segmentation
                label = np.zeros((256,256), dtype = np.uint8)
                torch_label = torch.from_numpy(label)
                torch_label = torch.unsqueeze(torch_label,dim=0)
            else:
                # Anomaly label for detection
                binary_label = 1
                # Anomaly Labels for segmentation
                common_path = os.path.join(self.root,self.obj)
                base = os.path.basename(os.path.dirname(file_paths[index]))
                file_name = os.path.split(file_paths[index])[1]
                file_num = os.path.splitext(file_name)[0]
                mask_name = file_num+'_mask.png'

                label_path = os.path.join(common_path,'ground_truth',base,mask_name)

                label = np.asarray(imread(label_path))
                label = self.resize(label)
                label = np.asarray(label)
                torch_label = torch.from_numpy(label)
                torch_label = torch.unsqueeze(torch_label,dim=0)
   
            return torch_image, torch_label, binary_label

        # Train mode
        else:
            fpath = file_paths[index]
            image = np.asarray(imread(fpath))

            image = self.normalize_images(image) #,train=True)

            torch_image = torch.from_numpy(image)

            torch_image = torch_image.type(torch.FloatTensor)

            return torch_image

    def normalize_images(self,images): #,train):
            # RGB and Resize Operations
            if images.shape[-1] != 3:
                images = self.gray2rgb(images)
            images = self.resize(images)
            images = np.asarray(images)
            # if train == True:
                # images = (images.astype(np.float32) - self.mean) / 255

                # mean = np.mean(self.mean, axis=tuple(range(self.mean.ndim-1)))
                # images = images.astype(np.float32) - mean 
                # images = images / 255 # between [-1,1]
                
                # # #### DEBUG #####
                # import pdb; pdb.set_trace()

                # images = (images - self.mean)
            images = images / 255 # between [0,1]
                # images = (images - self.mean) / self.stdev
            images = np.transpose(images, [2, 0, 1]) # Images with [C x H x W]

            return images

    def resize(self, image, shape=(256, 256)):
        return np.array(Image.fromarray(image).resize(shape[::-1]))

    def gray2rgb(self, images): # Check tuple logic
        tile_shape = tuple(np.ones(len(images.shape), dtype=int))
        tile_shape += (3,)

        images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
        # print(images.shape)
        return images

class CyCIF_Dataset(data.Dataset):
    def __init__(self,root,split="train"):
        super().__init__()
        self.root = root
        self.split = split
        self.files = {}

        fpattern = os.path.join(self.root, f'{self.split}_split/*/*.png')
        self.files = sorted(glob(fpattern))

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        file_paths = self.files
        
        # Val mode
        if self.split == "val" or self.split == "test":
            # Image
            fpath = file_paths[index]

            image = np.asarray(imread(fpath))
            image = self.normalize_images(image) #,train=False)

            torch_image = torch.from_numpy(image)
            torch_image = torch_image.type(torch.FloatTensor)

            if (os.path.basename(os.path.dirname(file_paths[index])) == 'clean'):
                binary_label = 0
                # Normal Label for segmentation
                label = np.zeros((256,256), dtype = np.uint8)
                torch_label = torch.from_numpy(label)
                torch_label = torch.unsqueeze(torch_label,dim=0)
            else:
                # Anomaly label for detection
                binary_label = 1
                # Anomaly Labels for segmentation
                ann_path = '/n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/'
                full_file = os.path.split(file_paths[index])[1]
                file_name = full_file.split("_")[0]
                channel_num = int(full_file.split("_")[1].split("c")[1])

                label_path = os.path.join(ann_path,f'{file_name}-c{channel_num:04d}.png')
                
                # Scale by upsampling factor = 16 to get coordinates
                label = np.asarray(imread(label_path))
                loc_y = int(full_file.split("_")[2].split("y")[1]) // 16
                loc_x = int(full_file.split("_")[3].split("x")[1]) // 16

                # Original label size is 1024, extract patch, scale to 256
                label_crop = label[loc_y: loc_y + 1024//16, loc_x: loc_x + 1024//16]
                
                label = cv2.resize(label_crop,(label_crop.shape[0]*4,label_crop.shape[1]*4)) # upsample to 256 x 256
                
                # import pdb; pdb.set_trace()
                # import torchvision
                # torchvision.utils.save_image(torch.unsqueeze(torch.from_numpy(label),dim=0),
                #                 "./label_test2.png",
                #                 normalize=True,
                #                 nrow = 1)
                # torchvision.utils.save_image((torch.from_numpy(image)),
                #                 "./img_test2.png",
                #                 normalize=True,
                #                 nrow = 1)

                torch_label = torch.from_numpy(label)
                torch_label = torch.unsqueeze(torch_label,dim=0) 
   
            return torch_image, torch_label, binary_label
        
        # Train mode
        else:
            
            fpath = file_paths[index]
            image = np.asarray(imread(fpath))

            image = self.normalize_images(image) #,train=True)

            torch_image = torch.from_numpy(image)

            torch_image = torch_image.type(torch.FloatTensor)

            return torch_image

    def normalize_images(self,images): #,train):
            # Resize Operation
            images = self.resize(images)
            if len(np.shape(images)) != 3:
                images = np.expand_dims(images,axis=2)
            images = np.asarray(images)

            images = images / 255 # between [0,1]
            images = np.transpose(images, [2, 0, 1]) # Images with [C x H x W]

            return images

    def resize(self, image, shape=(256, 256)):
        return cv2.resize(image,shape)



# Unit Test of Data loader
if __name__ == '__main__':
    from datasets import MVTec_Dataset
    from datasets import CyCIF_Dataset    
    from torch.utils import data
    import torchvision.utils as vutils

    # data_path = "/n/pfister_lab2/Lab/vcg_natural/mvtec/"
    # dataset = MVTec_Dataset(data_path,"bottle",split="test")
    data_path = "/n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/test/data/"
    dataset = CyCIF_Dataset(data_path,split="val")
    bs = 16
    trainloader = data.DataLoader(dataset, batch_size=bs, num_workers=0)

    for i, data_samples in enumerate(trainloader):
        imgs = data_samples

        # import pdb; pdb.set_trace()
        # vutils.save_image(imgs,
        #                   "./imgs_test.png",
        #                   normalize=True,
        #                   nrow = 8)

