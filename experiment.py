import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA, MNIST
from torch.utils.data import DataLoader
from datasets.datasets import MVTec_Dataset
import os 
from sklearn.metrics import roc_auc_score
import numpy as np

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

        self.diff_list = []
        self.det_list = []
        self.seg_list = [] 

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        # real_img, labels = batch
        real_img = batch
        self.curr_device = real_img.device

        # results = self.forward(real_img, labels = labels)
        results = self.forward(real_img)
        # Normalize input to [0,1] - NOT GOOD 
        # results[1] = ( real_img - real_img.min() ) / (real_img.max() - real_img.min() )
        
        # Mean center ground truth images (from input in [0,1])
        # results[1] = real_img - 0.5
        
        # #### DEBUG #####
        # results[1] = (real_img*0.36086705589152146) + 0.5383238062602264
        # if self.current_epoch == 0:
        #     import pdb; pdb.set_trace()
        train_loss = self.model.loss_function(*results,
                                              # results[0], real_img, results[2], results[3],
                                              M_N = self.params['kld_weight'],
                                              # M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        print((f"Model Version: {self.logger.version}"))
        print("Losses: ",{key: val.item() for key, val in train_loss.items()})

        train_path = os.path.join(self.logger.save_dir,self.logger.name,"version_"+str(self.logger.version),"train")
        if os.path.exists(train_path) == False:
            os.makedirs(train_path)
        
        if self.current_epoch % 10 == 0:
            vutils.save_image(results[0],
                            f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/train/"
                            f"output_{self.logger.name}_{self.current_epoch}.png",
                            normalize=True,
                            nrow = 8)
                            #   nrow=12)

            vutils.save_image(real_img,
                            f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/train/"
                            f"input_{self.logger.name}_{self.current_epoch}.png",
                            normalize=True,
                            nrow = 8)
            
            # difference image in training:
            diff_img = torch.abs(results[0] - real_img)
            vutils.save_image(diff_img,
                            f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/train/"
                            f"diff_tr_img_{self.logger.name}_{self.current_epoch}.png",
                            normalize=True,
                            nrow = 8)
        else:
            pass

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):      
        # real_img, labels = batch
        # real_img = batch
        real_img, seg_label, det_label = batch
        self.curr_device = real_img.device

        # results = self.forward(real_img, labels = labels)
        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results,
                                            # M_N = self.params['batch_size']/ self.num_val_imgs,
                                            M_N = self.params['kld_weight'],
                                            # M_N = self.params['batch_size']/ self.num_train_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        diff_img = torch.abs(results[0] - real_img)
        self.diff_list.append(diff_img)
        self.seg_list.append(seg_label)
        self.det_list.append(det_label)
        
        # # #### DEBUG #####
        # print("Val_LOSS",val_loss)
        # import pdb; pdb.set_trace()
        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        
        # Detection Metric
        labels = torch.cat(self.det_list)
        diff_imgs = torch.cat(self.diff_list)
        anomaly_scores = diff_imgs.view(diff_imgs.size(0),-1).mean(1)
        det_auroc = roc_auc_score(labels.cpu().detach(), anomaly_scores.cpu().detach())
        # print(f'Det: {det_auroc:4.2f}')

        # Segmentation Metric
        gt_mask = torch.cat(self.seg_list)
        gt_mask[gt_mask <= 128] = 0
        gt_mask[gt_mask > 128] = 255
        gt_mask[gt_mask == 255] = 1  # 1: anomaly

        anomaly_maps = torch.mean(diff_imgs,dim=1)
        anomaly_maps = torch.unsqueeze(anomaly_maps,dim=1)

        seg_auroc = roc_auc_score(gt_mask.cpu().detach().flatten(),anomaly_maps.cpu().detach().flatten())
        # print(f'Det: {seg_auroc:4.2f}')

        print(f'Det: {det_auroc:4.2f} // Seg: {seg_auroc:4.2f}')
        self.diff_list = []
        self.det_list = []
        self.seg_list = [] 
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        # test_input, test_label = next(iter(self.sample_dataloader))
        # test_input = next(iter(self.sample_dataloader))
        test_input, _, _ = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        # test_label = test_label.to(self.curr_device)
        # recons = self.model.generate(test_input, labels = test_label)

        recons = self.model.generate(test_input)

        if self.current_epoch % 10 == 0:
            vutils.save_image(recons.detach(),
                            f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                            f"recons_{self.logger.name}_{self.current_epoch}.png",
                            normalize=True,
                            nrow = 8)
                            #   nrow=12)

            vutils.save_image(test_input.detach(),
                            f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                            f"real_img_{self.logger.name}_{self.current_epoch}.png",
                            normalize=True,
                            nrow = 8)
                            #   nrow=12)
            
            # difference image in evaluation:
            diff_img = torch.abs(recons - test_input)
            vutils.save_image(diff_img.detach(),
                            f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                            f"diff_val_img_{self.logger.name}_{self.current_epoch}.png",
                            normalize=True,
                            nrow = 8)
        else:
            pass

        # try:
        #     samples = self.model.sample(self.params['batch_size'], #144,
        #                                 self.curr_device) # ,
        #                                 # labels = test_label)
        #     vutils.save_image(samples.cpu().data,
        #                       f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                       f"{self.logger.name}_{self.current_epoch}.png",
        #                       normalize=True,
        #                       nrow = 8)
        #                       nrow=12)
        # except:
        #     pass


        del test_input, recons #, samples


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=True)
        elif self.params['dataset'] == 'mnist':
            dataset = MNIST(root = self.params['data_path'],
                             train = True,
                             transform=transform,
                             download=False)
        elif self.params['dataset'] == 'mvtec':
            dataset = MVTec_Dataset(root = self.params['data_path'],
                             obj = self.params['object'],
                             split = "train")
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)

        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          num_workers = 8,
                          shuffle = True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            self.sample_dataloader =  DataLoader(CelebA(root = self.params['data_path'],
                                                        split = "test",
                                                        transform=transform,
                                                        download=False),
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'mnist':
            self.sample_dataloader =  DataLoader(MNIST(root = self.params['data_path'],
                                                        train = False,
                                                        transform=transform,
                                                        download=False),
                                                 batch_size= self.params['batch_size'],
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'mvtec':
            self.sample_dataloader =  DataLoader(MVTec_Dataset(root = self.params['data_path'],
                                                                obj = self.params['object'],
                                                                split = "test"),
                                                batch_size= self.params['batch_size'],
                                                shuffle = True,
                                                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'mnist':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            SetRange])   
        elif self.params['dataset'] == 'mvtec':
            transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()])                    
        else:
            raise ValueError('Undefined dataset type')
        return transform

