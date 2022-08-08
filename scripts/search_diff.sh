GPU=1

function run_reduced_svhn {
    DATASET=reduced_svhn
    MODEL=wresnet28_10
    EPOCH=160
    BATCH=128
    LR=0.05
    WD=0.01
    KOPS=2
    TEMP=3
    SLR=0.001
    CUTOUT=0
    SF=10
    LAM=1.0
}

# cifar10
function run_reduced_cifar10 {
    DATASET=reduced_cifar10
    MODEL=wresnet28_10
    EPOCH=200
    BATCH=128
    LR=0.1
    WD=0.0005
    KOPS=2
    TEMP=3
    SLR=0.001
    CUTOUT=16
    SF=10
    LAM=1.0
}

# cifar10
function run_cifar10 {
    DATASET=cifar10
    MODEL=wresnet28_10
    EPOCH=200
    BATCH=128
    LR=0.1
    WD=0.0005
    KOPS=2
    TEMP=3
    SLR=0.001
    CUTOUT=16
    SF=10
    LAM=1.0
}

if [ $1 = "reduced_cifar10" ]; then
    run_reduced_cifar10
elif [ $1 = "cifar10" ]; then
    run_cifar10
elif [ $1 = "reduced_svhn" ]; then
    run_reduced_svhn
fi

SAVE=${DATASET}_${MODEL}_${BATCH}_${EPOCH}_SLR${SLR}_SF${SF}_cutout_${CUTOUT}_lr${LR}_wd${WD}_lambda${LAM}
python ada_aug/search_img.py --k_ops ${KOPS}  --report_freq 10 --num_workers 4 --epochs ${EPOCH} --batch_size ${BATCH} --learning_rate ${LR} --dataset ${DATASET} --model_name ${MODEL} --save ${SAVE} --gpu ${GPU} --weight_decay ${WD} --proj_learning_rate ${SLR} --search_freq ${SF} --cutout --cutout_length ${CUTOUT} --temperature ${TEMP} --diff_aug --lambda_aug ${LAM}
