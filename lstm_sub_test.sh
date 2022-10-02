
CUDA_VISIBLE_DEVICES=3,4 python ada_aug/fold_search_ts.py --k_ops 1 --report_freq 5 --num_workers 4 --epochs 50 --batch_size 128  --learning_rate 0.003 --grad_clip 1.0 --dataset ptbxl --model_name lstm_ptb --save ptbsub_lstm_kfold --gpu 0.5 --cpu 2  --ray_name ptbsub_lstm_kfold --kfold 10 --weight_decay 0.01 --proj_learning_rate 0.0001 --search_freq 10  --temperature 3 --dataroot /mnt/data2/teddy/ptbxl-dataset --train_portion 0.5 --search_size 0.5 --labelgroup subdiagnostic --valselect --search_round 4


CUDA_VISIBLE_DEVICES=3,4 python ada_aug/fold_search_ts.py --k_ops 1 --report_freq 5 --num_workers 4 --epochs 50 --batch_size 128  --learning_rate 0.003 --grad_clip 1.0 --dataset ptbxl --model_name lstm_ptb --save ptbsub_lstm_kfold_all --gpu 0.5 --cpu 2  --ray_name ptbsub_lstm_kfold_all --kfold 10 --weight_decay 0.01 --proj_learning_rate 0.0001 --search_freq 10  --temperature 3 --dataroot /mnt/data2/teddy/ptbxl-dataset --train_portion 0.5 --search_size 0.5 --labelgroup subdiagnostic --valselect --search_round 4 --keep_aug --keep_len 100 --class_embed --class_adapt --same_train --diff_aug --loss_type relative

#CUDA_VISIBLE_DEVICES=1,2 python ada_aug/fold_search_ts.py --k_ops 1 --report_freq 5 --num_workers 4 --epochs 50 --batch_size 128  --learning_rate 0.003 --grad_clip 1.0 --dataset ptbxl --model_name lstm_ptb --save ptbsub_lstm_kfold_kadapt_all --gpu 0.5 --cpu 2  --ray_name ptbsub_lstm_kfold_kadapt_all --kfold 10 --weight_decay 0.01 --proj_learning_rate 0.0001 --search_freq 10  --temperature 3 --dataroot /mnt/data2/teddy/ptbxl-dataset --train_portion 0.5 --search_size 0.5 --labelgroup subdiagnostic --valselect --search_round 4 --keep_aug --keep_len 100 200 400 --class_embed --class_adapt --same_train --diff_aug --loss_type relative




