import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

parser = argparse.ArgumentParser(description='AE model runner')
parser.add_argument('--name',  '-n',
                    dest="model_name",
                    help=  'string of model name')
parser.add_argument('--data_path',  '-d',
                    dest="data_path",
                    help=  'path to the latent variables')

args = parser.parse_args()
name = args.model_name
out_path = args.data_path
# Data
# data = np.load('vae_latents.npy')
data = np.load(os.path.join(out_path,'latents.npy'))
latents = data[:,:-1]
det_label = data[:,-1]

# UMAP Object
reducer = umap.UMAP()
scaled_latents = StandardScaler().fit_transform(latents)
embedding = reducer.fit_transform(scaled_latents)
print(embedding.shape)

# Plot
# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=det_label,
#     cmap='plasma'
#     )
plt.scatter(
    embedding[det_label==0, 0],
    embedding[det_label==0, 1],
    c='b',
    label='Normal'
    )
plt.scatter(
    embedding[det_label==1, 0],
    embedding[det_label==1, 1],
    c='y',
    label='Anomaly'
    )
plt.legend(loc="upper left")
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the '+name.upper()+' Latent Space', fontsize=24)
plt.savefig(os.path.join(out_path,name.lower()+'-latent-umap-scaled.png'),bbox_inches='tight')