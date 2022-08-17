from __future__ import print_function
from __future__ import absolute_import
from multiprocessing import reduction

import os
import sys
import time
import torch
import utils
import logging
import argparse
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
import torch.nn as nn
import torch.utils

from adaptive_augmentor import AdaAug_TS
from networks import get_model_tseries
from networks.projection import Projection_TSeries
from config import get_search_divider
from dataset import get_ts_dataloaders, get_num_class, get_label_name, get_dataset_dimension,get_num_channel
from operation_tseries import TS_OPS_NAMES,ECG_OPS_NAMES,TS_ADD_NAMES,MAG_TEST_NAMES,NOMAG_TEST_NAMES
from step_function import search_train,search_infer
import wandb

parser = argparse.ArgumentParser("ada_aug")
parser.add_argument('--dataroot', type=str, default='./', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
parser.add_argument('--labelgroup', default='')
parser.add_argument('--test_size', type=float, default=0.2, help='test size')
parser.add_argument('--search_size', type=float, default=0.5, help='test size')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.400, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--multilabel', action='store_true', default=False, help='using multilabel default False')
parser.add_argument('--train_portion', type=float, default=1, help='portion of training data')
parser.add_argument('--default_split', action='store_true', help='use dataset deault split')
parser.add_argument('--proj_learning_rate', type=float, default=1e-2, help='learning rate for h')
parser.add_argument('--proj_weight_decay', type=float, default=1e-3, help='weight decay for h]')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--use_cuda', type=bool, default=True, help="use cuda default True")
parser.add_argument('--use_parallel', type=bool, default=False, help="use data parallel default False")
parser.add_argument('--model_name', type=str, default='wresnet40_2', help="mode _name")
parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
parser.add_argument('--k_ops', type=int, default=1, help="number of augmentation applied during training")
parser.add_argument('--temperature', type=float, default=1.0, help="temperature")
parser.add_argument('--search_freq', type=float, default=1, help='exploration frequency')
parser.add_argument('--n_proj_layer', type=int, default=0, help="number of hidden layer in augmentation policy projection")
parser.add_argument('--valselect', action='store_true', default=False, help='use valid select')
parser.add_argument('--augselect', type=str, default='', help="augmentation selection")
parser.add_argument('--diff_aug', action='store_true', default=False, help='use valid select')
parser.add_argument('--not_mix', action='store_true', default=False, help='use valid select')
parser.add_argument('--not_reweight', action='store_true', default=False, help='use valid select')
parser.add_argument('--lambda_aug', type=float, default=1.0, help="augment sample weight")
parser.add_argument('--class_adapt', action='store_true', default=False, help='class adaptive')
parser.add_argument('--relative_loss', action='store_true', default=False, help='use valid select')

args = parser.parse_args()
debug = True if args.save == "debug" else False
if args.k_ops>0:
    Aug_type = 'AdaAug'
else:
    Aug_type = 'NOAUG'
if args.diff_aug:
    description = 'diff2'
else:
    description = ''
if args.class_adapt:
    description += 'cada'
else:
    description += ''
if args.diff_aug and not args.not_reweight:
    description+='rew'
now_str = time.strftime("%Y%m%d-%H%M%S")
args.save = '{}-{}-{}{}'.format(now_str, args.save,Aug_type,description)
if debug:
    args.save = os.path.join('debug', args.save)
else:
    args.save = os.path.join('search', args.dataset, args.save)
utils.create_exp_dir(args.save)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(args.gpu)
    utils.reproducibility(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    #wandb
    experiment_name = f'{Aug_type}{description}lamda{args.lambda_aug}_search{args.augselect}_vselect_{args.dataset}{args.labelgroup}_{args.model_name}_e{args.epochs}_lr{args.learning_rate}'
    run_log = wandb.init(config=args, 
                  project='AdaAug',
                  group=experiment_name,
                  name=f'{now_str}_' + experiment_name,
                  dir='./',
                  job_type="DataAugment",
                  reinit=True)

    #  dataset settings
    n_channel = get_num_channel(args.dataset)
    n_class = get_num_class(args.dataset,args.labelgroup)
    sdiv = get_search_divider(args.model_name)
    class2label = get_label_name(args.dataset, args.dataroot)
    multilabel = args.multilabel
    diff_augment = args.diff_aug
    diff_mix = not args.not_mix
    diff_reweight = not args.not_reweight
    train_queue, valid_queue, search_queue, test_queue, tr_search_queue = get_ts_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        args.dataroot, args.cutout, args.cutout_length,
        split=args.train_portion, split_idx=0, target_lb=-1,
        search=True, search_divider=sdiv,search_size=args.search_size,
        test_size=args.test_size,multilabel=args.multilabel,default_split=args.default_split,
        labelgroup=args.labelgroup)
    
    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'  |total: {len(train_queue.dataset)}')
    logging.info(f'  |train: {len(train_queue)*args.batch_size}')
    logging.info(f'  |valid: {len(valid_queue)*args.batch_size}')
    logging.info(f'  |search: {len(search_queue)*sdiv}')

    #  model settings
    gf_model = get_model_tseries(model_name=args.model_name, num_class=n_class,n_channel=n_channel,
        use_cuda=True, data_parallel=False,dataset=args.dataset)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(gf_model))
    h_input = gf_model.fc.in_features
    if args.class_adapt:
        h_input =h_input + n_class
    h_model = Projection_TSeries(in_features=h_input,
        n_layers=args.n_proj_layer, n_hidden=128, augselect=args.augselect).cuda()

    #  training settings
    '''gf_optimizer = torch.optim.SGD(
        gf_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)'''
    gf_optimizer = torch.optim.AdamW(gf_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) #follow ptbxl batchmark!!!
    '''scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gf_optimizer,
        float(args.epochs), eta_min=args.learning_rate_min)'''
    scheduler = torch.optim.lr_scheduler.OneCycleLR(gf_optimizer, max_lr=args.learning_rate, epochs = args.epochs, steps_per_epoch = len(train_queue)) #follow ptbxl batchmark!!!

    h_optimizer = torch.optim.Adam(
        h_model.parameters(),
        lr=args.proj_learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.proj_weight_decay)

    if not multilabel:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    criterion = criterion.cuda()

    #  AdaAug settings
    after_transforms = train_queue.dataset.after_transforms
    adaaug_config = {'sampling': 'prob',
                    'k_ops': 1, #as paper
                    'delta': 0.0, 
                    'temp': 1.0, 
                    'search_d': get_dataset_dimension(args.dataset),
                    'target_d': get_dataset_dimension(args.dataset),
                    'gf_model_name': args.model_name}

    adaaug = AdaAug_TS(after_transforms=after_transforms,
        n_class=n_class,
        gf_model=gf_model,
        h_model=h_model,
        save_dir=args.save,
        config=adaaug_config,
        multilabel=multilabel,
        augselect=args.augselect,
        class_adaptive=args.class_adapt)
    #for valid data select
    best_val_acc,best_gf,best_h = 0,None,None
    #  Start training
    start_time = time.time()
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        step_dic={'epoch':epoch}
        diff_dic = {'difficult_aug':diff_augment,'reweight':diff_reweight,'lambda_aug':args.lambda_aug, 'class_adaptive':args.class_adapt,
                'relative_loss':args.relative_loss}
        # searching
        train_acc, train_obj, train_dic = search_train(args,train_queue, search_queue, tr_search_queue, gf_model, adaaug,
            criterion, gf_optimizer,scheduler, args.grad_clip, h_optimizer, epoch, args.search_freq, multilabel=multilabel,n_class=n_class,
            **diff_dic)

        # validation
        valid_acc, valid_obj,valid_dic = search_infer(valid_queue, gf_model, criterion, multilabel=multilabel,n_class=n_class,mode='valid')
        logging.info(f'train_acc {train_acc} valid_acc {valid_acc}')
        #scheduler.step()
        #test
        test_acc, test_obj, test_dic  = search_infer(test_queue, gf_model, criterion, multilabel=multilabel,n_class=n_class,mode='test')
        logging.info('test_acc %f', test_acc)
        #val select
        if args.valselect and valid_acc>best_val_acc:
            best_val_acc = valid_acc
            result_valid_dic = {f'result_{k}': valid_dic[k] for k in valid_dic.keys()}
            result_test_dic = {f'result_{k}': test_dic[k] for k in test_dic.keys()}
            valid_dic['best_valid_acc_avg'] = valid_acc
            test_dic['best_test_acc_avg'] = test_acc
            best_gf = gf_model
            best_h = h_model
        elif not args.valselect:
            result_valid_dic = {f'result_{k}': valid_dic[k] for k in valid_dic.keys()}
            result_test_dic = {f'result_{k}': test_dic[k] for k in test_dic.keys()}
            best_gf = gf_model
            best_h = h_model

        utils.save_model(best_gf, os.path.join(args.save, 'gf_weights.pt'))
        utils.save_model(best_h, os.path.join(args.save, 'h_weights.pt'))
        
        step_dic.update(test_dic)
        step_dic.update(train_dic)
        step_dic.update(valid_dic)
        wandb.log(step_dic)

    end_time = time.time()
    elapsed = end_time - start_time

    test_acc, test_obj, test_dic = search_infer(test_queue, gf_model, criterion, multilabel=multilabel,n_class=n_class,mode='test')
    #wandb
    step_dic.update(test_dic)
    step_dic.update(result_valid_dic)
    step_dic.update(result_test_dic)
    wandb.log(step_dic)
    #save&log
    #utils.save_model(gf_model, os.path.join(args.save, 'gf_weights.pt'))
    #utils.save_model(h_model, os.path.join(args.save, 'h_weights.pt'))
    adaaug.save_history(class2label)
    figure = adaaug.plot_history()

    logging.info(f'test_acc {test_acc}')
    logging.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logging.info(f'saved to: {args.save}')


if __name__ == '__main__':
    main()
