#!/bin/bash
obj_list="vae_mvtec_carpet.yaml vae_mvtec_tile.yaml vae_mvtec_wood.yaml vae_mvtec.yaml vae_mvtec_cable.yaml"
for obj in $obj_list; do 
    sbatch run_train.sh "$obj"
    echo "Submitted job: $obj" 
done
