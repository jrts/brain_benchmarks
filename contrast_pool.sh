#!/bin/sh

# abide_schaefer100_1025 adni_schaefer100_1327 ppmi_schaefer100_209 
# taowu_schaefer100_40 neurocon_schaefer100_41 matai_schaefer100_60
# abide_aal116_1025 adni_aal116_1327 ppmi_aal116_209
# taowu_aal116_40 neurocon_aal116_41 matai_aal116_60
# GCN GraphSage GAT BrainNetCNN LiNet PRGNN GatedGCN BrainGNN
# BrainGNN

for model in ContrastPool
do
  for dataset in adni_aal116_1327
  do
    for L in 3
    do
      for hidden_dim in 100
      do
          python main.py --model=$model --config_dir='configs_01' --dataset=$dataset --L=$L --hidden_dim=$hidden_dim --epochs=80 \
                         --gpu_id=0 --batch_size=20 --save_base_url='../results_benchmarks' --threshold='all'
      done
    done
  done
done