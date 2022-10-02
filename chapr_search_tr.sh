
CUDA_VISIBLE_DEVICES=3,4 python ada_aug/fold_train_ts.py --temperature 3 --delta 0.3 --search_dataset chapman --gf_model_path search/chapman/20220915-053541-chapmanr_resnet_kfold_all-AdaAugdiff2relativecadarewkeepauto1/ --h_model_path search/chapman/20220915-053541-chapmanr_resnet_kfold_all-AdaAugdiff2relativecadarewkeepauto1/ --gf_model_name resnet_wang --k_ops 1 --report_freq 10 --num_workers 2 --epochs 50 --batch_size 128 --learning_rate 0.01 --dataset chapman  --model_name resnet_wang --labelgroup rhythm --save chapr_resnettr_kfold_all --ray_name chapr_resnettr_kfold_all --gpu 1 --cpu 2 --weight_decay 0.01 --train_portion 1 --dataroot /mnt/data2/teddy/CWDA_research/CWDA/datasets/WFDB_ChapmanShaoxing/ --kfold 10 --valselect --class_embed --class_adapt --keep_aug --keep_len 500







