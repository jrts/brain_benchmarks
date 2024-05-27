#!/bin/sh

# abide_schaefer100_1025 adni_schaefer100_1327 ppmi_schaefer100_209 
# taowu_schaefer100_40 neurocon_schaefer100_41 matai_schaefer100_60
# abide_aal116_1025 adni_aal116_1327 ppmi_aal116_209
# taowu_aal116_40 neurocon_aal116_41 matai_aal116_60
# GCN GraphSage GAT BrainNetCNN LiNet PRGNN GatedGCN BrainGNN BNTF
# BrainGNN

for model in BNTF
do
  for dataset in abide_aal116_1025 adni_aal116_1327 ppmi_aal116_209 taowu_aal116_40 neurocon_aal116_41 matai_aal116_60
  # for dataset in abide_schaefer100_1025 adni_schaefer100_1327 ppmi_schaefer100_209 taowu_schaefer100_40 neurocon_schaefer100_41 matai_schaefer100_60
  do
    for hidden_dim in 64 100
    do
        python main.py --model=$model --config_dir='configs_01' --dataset=$dataset --hidden_dim=$hidden_dim --epochs=80 \
                       --gpu_id=3 --batch_size=20 --save_base_url='../0_test_brain_gnn_results_benchmarks'
    done
  done
done