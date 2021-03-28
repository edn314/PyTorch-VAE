#!/bin/bash
# obj_list="vae_mvtec_carpet.yaml vae_mvtec_tile.yaml vae_mvtec_wood.yaml vae_mvtec.yaml vae_mvtec_cable.yaml"
obj_list="ae_cycif.yaml cae_cycif.yaml ms_ssim_cae_cycif.yaml vae_cycif.yaml"
for obj in $obj_list; do 
    sbatch run_train.sh "$obj"
    echo "Submitted job: $obj" 
done
