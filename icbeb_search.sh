
CUDA_VISIBLE_DEVICES=2,3,6 python ada_aug/fold_search_ts.py --k_ops 1 --report_freq 5 --num_workers 4 --epochs 50 --batch_size 128  --learning_rate 0.01  --dataset icbeb --model_name resnet_wang --save icbeball_resnet_kfold --gpu 1 --cpu 2  --ray_name icbeball_resnet_kfold --kfold 10 --weight_decay 0.01 --proj_learning_rate 0.0001 --search_freq 10  --temperature 3 --dataroot /mnt/data2/teddy/ICBEB-dataset/ --train_portion 0.5 --search_size 0.5 --labelgroup all --valselect --search_round 4

#CUDA_VISIBLE_DEVICES=2,3,4,6 python ada_aug/fold_train_ts.py --temperature 3 --delta 0.3 --search_dataset icbeb --gf_model_path search/icbeb/20221001-140020-icbeball_resnet_kfold-AdaAug/ --h_model_path search/icbeb/20221001-140020-icbeball_resnet_kfold-AdaAug/ --gf_model_name resnet_wang --k_ops 1 --report_freq 10 --num_workers 2 --epochs 50 --batch_size 128 --learning_rate 0.01 --dataset icbeb --model_name resnet_wang --labelgroup all --save icbeball_resnettr_kfold --ray_name icbeball_resnettr_kfold --gpu 1 --cpu 2 --weight_decay 0.01 --train_portion 1 --dataroot /mnt/data2/teddy/ICBEB-dataset/ --kfold 10 --valselect


