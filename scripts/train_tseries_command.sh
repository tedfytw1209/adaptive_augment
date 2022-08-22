GPU=0

function run_reduced_svhn {
    DATASET=reduced_svhn
    MODEL=wresnet28_10
    EPOCH=180
    BATCH=128
    LR=0.05
    WD=0.01
    CUTOUT=0
    TRAIN_PORTION=1
    GF=/mnt/data2/teddy/adaptive_augment/search/reduced_svhn/20220629-000106-reduced_svhn_wresnet40_2_128_160_SLR0.001_SF1_cutout_0_lr0.05_wd0.01/
    H=/mnt/data2/teddy/adaptive_augment/search/reduced_svhn/20220629-000106-reduced_svhn_wresnet40_2_128_160_SLR0.001_SF1_cutout_0_lr0.05_wd0.01/
    SDN=reduced_svhn
    GFN=wresnet40_2
    DELTA=0.4
    TEMP=2
    KOPS=2
}

function run_reduced_cifar10 {
    DATASET=reduced_cifar10
    MODEL=wresnet28_10
    EPOCH=240
    BATCH=128
    LR=0.1
    WD=0.0005
    CUTOUT=16
    TRAIN_PORTION=1
    GF='/mnt/data2/teddy/adaptive_augment/search/reduced_cifar10/20220701-111722-reduced_cifar10_wresnet40_2_128_200_SLR0.001_SF3_cutout_16_lr0.1_wd0.0005/'
    H='/mnt/data2/teddy/adaptive_augment/search/reduced_cifar10/20220701-111722-reduced_cifar10_wresnet40_2_128_200_SLR0.001_SF3_cutout_16_lr0.1_wd0.0005/'
    SDN=reduced_cifar10
    GFN=wresnet40_2
    DELTA=0.3
    TEMP=3
    KOPS=2
}

python ada_aug/train_ts.py --temperature 3 --delta 0.3 --search_dataset ptbxl --gf_model_path  --h_model_path  --gf_model_name resnet_wang --k_ops 3 \
 --report_freq 10 --num_workers 4 --epochs 50 --batch_size 128 --learning_rate 0.01 --dataset ptbxl --model_name resnet_wang --labelgroup subdiagnostic \
 --save ptbxl_resnet_train --gpu 3 --weight_decay 0.01 --train_portion 1 --dataroot /mnt/data2/teddy/ptbxl-dataset --default_split --valselect

CUDA_VISIBLE_DEVICES=1,4 python ada_aug/fold_train_ts.py --temperature 3 --delta 0.3 --search_dataset ptbxl --gf_model_path --h_model_path \
 --gf_model_name resnet_wang --k_ops 1 --report_freq 10 --num_workers 4 --epochs 50 --batch_size 128 --learning_rate 0.01 --dataset ptbxl \
 --model_name resnet_wang --labelgroup superdiagnostic  --save ptbxl_resnet_kfold_diffnw --ray_name ptbsup_resnet_kfold_diffnw_train \
 --gpu 0.25 --cpu 2 --weight_decay 0.01 --train_portion 1 --dataroot /mnt/data2/teddy/ptbxl-dataset --kfold 10 --valselect --diff_aug --not_reweight