import yaml
import argparse
import numpy as np
import torch
import os
import torchvision.utils as vutils

from models import *
from experiment import VAEXperiment
from torch.utils.data import DataLoader
from datasets.datasets import CyCIF_Dataset, MVTec_Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generic runner for AE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--model_path',  '-m',
                    dest="model_path",
                    help=  'path to the model checkpoint')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

# Load best model
# model_path = '/n/pfister_lab2/Lab/enovikov/unsup-ano-detection/anomaly-project/Working-Models/CyCIF/VAE-L2/version_1/checkpoints/_ckpt_epoch_92.ckpt'
model_path = args.model_path
checkpoint = torch.load(model_path)
# Method to remove string "model." from odict and preserve order
def change_keys(ord_dict):
    for _ in range(len(ord_dict)):
        k, v = ord_dict.popitem(False) # pop beginning to end
        new_k = k.split('model.')[1]

        ord_dict[new_k] = v

# def extract_encoder(ord_dict):
#     rm_modules = ['decoder','final_layer']
#     ord_dict = {key: val for key, val in ord_dict.items() if not any(ele in key for ele in rm_modules)}
#     return ord_dict

state_dict = checkpoint['state_dict']
change_keys(state_dict)

# state_dict = extract_encoder(state_dict)
model.load_state_dict(state_dict)# , strict=False)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initalize dataloader 
test_dataloader =  DataLoader(MVTec_Dataset(root = config['exp_params']['data_path'],
                                    obj = config['exp_params']['object'],
                                    split = "test"),
                                    batch_size= 1,
                                    shuffle = True,
                                    drop_last=True)
# test_dataloader =  DataLoader(CyCIF_Dataset(root = config['exp_params']['data_path'],
#                                     split = "test"),
#                                     batch_size= 1,
#                                     shuffle = True,
#                                     drop_last=True)
num_test_imgs = len(test_dataloader)

# Test the model
with torch.no_grad():
    latent_embedding = []
    for i, (image, seg_label, det_label) in tqdm(enumerate(test_dataloader)):
        image = image.to(device)
        seg_label = seg_label.to(device)
        det_label = det_label.to(device)

        output = model.encode(image)[0]
        # vutils.save_image(output,
        #                 "output_test.png",
        #                 normalize=True,
        #                 nrow = 1)

        # Append detection label to the output latent vector
        det_label = torch.unsqueeze(det_label,dim=0)
        latent_embedding.append(torch.cat((output,det_label.float()),1))

latents = torch.cat(latent_embedding)
latents = latents.cpu().detach().numpy()
out_path = os.path.dirname(os.path.dirname(model_path))
np.save(os.path.join(out_path,'latents.npy'),latents)




