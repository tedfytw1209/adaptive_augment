
CUDA_VISIBLE_DEVICES=5,6 python ada_aug/fold_search_ts.py --k_ops 1 --report_freq 5 --num_workers 2 --epochs 50 --batch_size 128  --learning_rate 0.01  --dataset ptbxl --model_name resnet_wang --save ptbsubml_resnet_kfold --gpu 0.33 --cpu 2 --ray_name ptbsubml_resnet_kfold --kfold 10 --weight_decay 0.01 --proj_learning_rate 0.0001 --search_freq 10  --temperature 3 --dataroot /mnt/data2/teddy/ptbxl-dataset --train_portion 0.5 --search_size 0.5 --labelgroup subdiagnostic --multilabel --valselect --search_round 4 --class_adapt --class_embed --policy_loss classdiff

CUDA_VISIBLE_DEVICES=5,6 python ada_aug/fold_search_ts.py --k_ops 1 --report_freq 5 --num_workers 2 --epochs 50 --batch_size 128  --learning_rate 0.01  --dataset ptbxl --model_name resnet_wang --save ptbsup_resnet_kfold_ckeep --gpu 0.33 --cpu 2 --ray_name ptbsup_resnet_kfold_ckeep --kfold 10 --weight_decay 0.01 --proj_learning_rate 0.0001 --search_freq 10  --temperature 3 --dataroot /mnt/data2/teddy/ptbxl-dataset --train_portion 0.5 --search_size 0.5 --labelgroup superdiagnostic --valselect --class_adapt --class_embed --keep_aug --keep_len 100 --search_round 4



