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
    GF='/mnt/data2/teddy/adaptive_augment/search/reduced_svhn/20220629-000106-reduced_svhn_wresnet40_2_128_160_SLR0.001_SF1_cutout_0_lr0.05_wd0.01/'
    H='/mnt/data2/teddy/adaptive_augment/search/reduced_svhn/20220629-000106-reduced_svhn_wresnet40_2_128_160_SLR0.001_SF1_cutout_0_lr0.05_wd0.01/'
    SDN=reduced_svhn
    GFN=wresnet28_10
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
    GF='/mnt/data2/teddy/adaptive_augment/search/reduced_cifar10/20220809-101703-reduced_cifar10_wresnet28_10_128_200_SLR0.001_SF10_cutout_16_lr0.1_wd0.0005_lambda0.5/'
    H='/mnt/data2/teddy/adaptive_augment/search/reduced_cifar10/20220809-101703-reduced_cifar10_wresnet28_10_128_200_SLR0.001_SF10_cutout_16_lr0.1_wd0.0005_lambda0.5/'
    SDN=reduced_cifar10
    GFN=wresnet28_10
    DELTA=0.3
    TEMP=3
    KOPS=2
}

if [ $1 = "reduced_cifar10" ]; then
    run_reduced_cifar10
elif [ $1 = "reduced_svhn" ]; then
    run_reduced_svhn
fi

SAVE=${DATASET}_${MODEL}_${BATCH}_${EPOCH}_cutout_${CUTOUT}_lr${LR}_wd${WD}_kops_${KOPS}_TEMP_${TEMP}_${DELTA}
python ada_aug/train_img.py --temperature ${TEMP} --delta ${DELTA} --search_dataset ${SDN} --gf_model_path ${GF} --h_model_path ${H} --gf_model_name ${GFN} --k_ops ${KOPS} --report_freq 10 --num_workers 8 --epochs ${EPOCH} --batch_size ${BATCH} --learning_rate ${LR} --dataset ${DATASET} --model_name ${MODEL} --save ${SAVE} --gpu ${GPU} --weight_decay ${WD} --train_portion 1 --diff_aug
