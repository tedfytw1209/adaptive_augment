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

from adaptive_augmentor import AdaAug_TS, AdaAugkeep_TS
from networks import get_model_tseries
from networks.projection import Projection_TSeries
from config import get_search_divider
from dataset import get_ts_dataloaders, get_num_class, get_label_name, get_dataset_dimension,get_num_channel
from operation_tseries import TS_OPS_NAMES,ECG_OPS_NAMES,TS_ADD_NAMES,MAG_TEST_NAMES,NOMAG_TEST_NAMES
from step_function import train,infer,search_train,search_infer
from warmup_scheduler import GradualWarmupScheduler
from config import get_warmup_config
import wandb

import ray
import ray.tune as tune
from ray.tune.integration.wandb import WandbTrainableMixin

os.environ['WANDB_START_METHOD'] = 'thread'
RAY_DIR = './ray_results'
parser = argparse.ArgumentParser("ada_aug")
parser.add_argument('--base_path', type=str, default='/mnt/data2/teddy/adaptive_augment/', help='base path of code')
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
###for ray###
parser.add_argument('--ray_dir', type=str, default=RAY_DIR,  help='Ray directory.')
parser.add_argument('--ray_name', type=str, default='ray_experiment')
parser.add_argument('--cpu', type=float, default=4, help='Allocated by Ray')
parser.add_argument('--gpu', type=float, default=0.12, help='Allocated by Ray')
###
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--multilabel', action='store_true', default=False, help='using multilabel default False')
parser.add_argument('--train_portion', type=float, default=1, help='portion of training data')
parser.add_argument('--default_split', action='store_true', help='use dataset deault split')
parser.add_argument('--kfold', type=int, default=0, help='use kfold cross validation')
parser.add_argument('--proj_learning_rate', type=float, default=1e-2, help='learning rate for h')
parser.add_argument('--proj_weight_decay', type=float, default=1e-3, help='weight decay for h]')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--use_cuda', type=bool, default=True, help="use cuda default True")
parser.add_argument('--use_parallel', type=bool, default=False, help="use data parallel default False")
parser.add_argument('--model_name', type=str, default='wresnet40_2', help="mode _name")
parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
parser.add_argument('--search_dataset', type=str, default='./', help='search dataset name')
parser.add_argument('--gf_model_name', type=str, default='./', help='gf_model name')
parser.add_argument('--gf_model_path', type=str, default='./', help='gf_model path')
parser.add_argument('--h_model_path', type=str, default='./', help='h_model path')
parser.add_argument('--k_ops', type=int, default=1, help="number of augmentation applied during training")
parser.add_argument('--delta', type=float, default=0.3, help="degree of perturbation in magnitude")
parser.add_argument('--temperature', type=float, default=1.0, help="temperature")
parser.add_argument('--n_proj_layer', type=int, default=0, help="number of additional hidden layer in augmentation policy projection")
parser.add_argument('--n_proj_hidden', type=int, default=128, help="number of hidden units in augmentation policy projection layers")
parser.add_argument('--restore_path', type=str, default='./', help='restore model path')
parser.add_argument('--restore', action='store_true', default=False, help='restore model default False')
parser.add_argument('--mapselect', action='store_true', default=False, help='use map select for multilabel')
parser.add_argument('--valselect', action='store_true', default=False, help='use valid select')
parser.add_argument('--notwarmup', action='store_true', default=False, help='use valid select')
parser.add_argument('--augselect', type=str, default='', help="augmentation selection")
parser.add_argument('--diff_aug', action='store_true', default=False, help='use valid select')
parser.add_argument('--not_reweight', action='store_true', default=False, help='use valid select')
parser.add_argument('--lambda_aug', type=float, default=1.0, help="augment sample weight")
parser.add_argument('--class_adapt', action='store_true', default=False, help='class adaptive')
parser.add_argument('--class_embed', action='store_true', default=False, help='class embed') #tmp use
parser.add_argument('--noaug_reg', type=str, default='', help='add regular for noaugment ',
        choices=['cadd','add',''])
parser.add_argument('--keep_aug', action='store_true', default=False, help='info keep augment')
parser.add_argument('--keep_mode', type=str, default='auto', help='info keep mode',choices=['auto','adapt','b','p','t'])
parser.add_argument('--adapt_target', type=str, default='len', help='info keep mode',choices=['len','seg'])
parser.add_argument('--keep_seg', type=int, nargs='+', default=[1], help='info keep segment mode')
parser.add_argument('--keep_grid', action='store_true', default=False, help='info keep augment grid')
parser.add_argument('--keep_thres', type=float, default=0.6, help="keep augment weight (lower protect more)")
parser.add_argument('--keep_len', type=int, nargs='+', default=[100], help="info keep seq len")
parser.add_argument('--keep_bound', type=float, default=0.0, help="info keep bound %")
parser.add_argument('--visualize', action='store_true', default=False, help='visualize')

args = parser.parse_args()
debug = True if args.save == "debug" else False
if args.k_ops>0:
    Aug_type = 'AdaAug'
else:
    Aug_type = 'NOAUG'
Aug_type += args.augselect
if args.diff_aug:
    description = 'diff2'
else:
    description = ''
if args.class_adapt:
    description += 'cada'
else:
    description += ''
description += args.noaug_reg
if args.diff_aug and not args.not_reweight:
    description+='rew'
if args.keep_aug:
    keep_seg_str = ''.join([str(i) for i in args.keep_seg])
    description+=f'keep{args.keep_mode}{keep_seg_str}'
now_str = time.strftime("%Y%m%d-%H%M%S")
args.save = '{}-{}-{}{}'.format(now_str, args.save,Aug_type,description)
if debug:
    args.save = os.path.join('debug', args.save)
else:
    args.save = os.path.join('search', args.dataset, args.save)
utils.create_exp_dir(args.save)
for i in range(10):
    utils.create_exp_dir(os.path.join(args.save,f'fold{i}'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
API_KEY = 'cb4c412d9f47cd551e38050ced659b0c58926986'

#ray model
class RayModel(WandbTrainableMixin, tune.Trainable):
    def setup(self, *_args): #use new setup replace _setup
        #self.trainer = TSeriesModelTrainer(self.config)
        os.environ['WANDB_START_METHOD'] = 'thread'
        args = self.config['args']
        #  dataset settings for search
        n_channel = get_num_channel(args.dataset)
        n_class = get_num_class(args.dataset,args.labelgroup)
        sdiv = get_search_divider(args.model_name)
        class2label = get_label_name(args.dataset, args.dataroot,args.labelgroup)
        multilabel = args.multilabel
        diff_augment = args.diff_aug
        diff_reweight = not args.not_reweight
        self.class_noaug, self.noaug_add = False, False
        if args.noaug_reg=='cadd':
            self.class_noaug = True
            self.noaug_add = True
        elif args.noaug_reg=='add':
            self.noaug_add = True
        test_fold_idx = self.config['kfold']
        train_val_test_folds = [[],[],[]] #train,valid,test
        for i in range(10):
            curr_fold = (i+test_fold_idx)%10 +1 #fold is 1~10
            if i==0:
                train_val_test_folds[2].append(curr_fold)
            elif i==9:
                train_val_test_folds[1].append(curr_fold)
            else:
                train_val_test_folds[0].append(curr_fold)
        print('Train/Valid/Test fold split ',train_val_test_folds)
        self.train_queue, self.valid_queue, self.search_queue, self.test_queue, self.tr_search_queue = get_ts_dataloaders(
            args.dataset, args.batch_size, args.num_workers,
            args.dataroot, args.cutout, args.cutout_length,
            split=args.train_portion, split_idx=0, target_lb=-1,
            search=False,test_size=args.test_size,multilabel=args.multilabel,default_split=args.default_split,
            fold_assign=train_val_test_folds,labelgroup=args.labelgroup)
        #  task model settings
        self.task_model = get_model_tseries(model_name=args.model_name,
                            num_class=n_class,n_channel=n_channel,
                            use_cuda=True, data_parallel=False,dataset=args.dataset)
        #follow ptbxl batchmark!!!
        self.optimizer = torch.optim.AdamW(self.task_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.learning_rate,
            epochs = args.epochs, steps_per_epoch = len(self.train_queue)) #follow ptbxl batchmark!!!
        if not args.notwarmup:
            m, e = get_warmup_config(args.dataset)
            self.scheduler = GradualWarmupScheduler( #paper not mention!!!
                self.optimizer,
                multiplier=m,
                total_epoch=e,
                after_scheduler=self.scheduler)
        if not multilabel:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduce='mean')
        self.criterion = self.criterion.cuda()
        #  restore setting
        if args.restore:
            trained_epoch = utils.restore_ckpt(self.task_model, self.optimizer, self.scheduler, args.restore_path, location=0) + 1
            print(f'From epoch {trained_epoch}, Resume')
        else:
            trained_epoch = 0
        #  load trained adaaug sub models
        search_n_class = get_num_class(args.search_dataset,args.labelgroup)
        self.gf_model = get_model_tseries(model_name=args.gf_model_name,
                            num_class=search_n_class,n_channel=n_channel,
                            use_cuda=True, data_parallel=False,dataset=args.dataset)
        h_input = self.gf_model.fc.in_features
        label_num,label_embed =0,0
        if args.class_adapt and args.class_embed:
            label_num = n_class
            label_embed = 32 #tmp use
            h_input = h_input + label_embed
        elif args.class_adapt:
            h_input =h_input + n_class
        #keep aug
        proj_add = 0
        if args.keep_mode=='adapt':
            proj_add = max(len(args.keep_len),len(args.keep_seg)) + 1
        self.h_model = Projection_TSeries(in_features=h_input,label_num=label_num,label_embed=label_embed,
            n_layers=args.n_proj_layer, n_hidden=args.n_proj_hidden, augselect=args.augselect, proj_addition=proj_add).cuda()
        utils.load_model(self.gf_model, os.path.join(self.config['BASE_PATH'],f'{args.gf_model_path}',f'fold{test_fold_idx}', 'gf_weights.pt'), location=0)
        utils.load_model(self.h_model, os.path.join(self.config['BASE_PATH'],f'{args.h_model_path}',f'fold{test_fold_idx}', 'h_weights.pt'), location=0)

        for param in self.gf_model.parameters():
            param.requires_grad = False
        for param in self.h_model.parameters():
            param.requires_grad = False
        after_transforms = self.train_queue.dataset.after_transforms
        adaaug_config = {'sampling': 'prob',
                    'k_ops': args.k_ops,
                    'delta': args.delta,
                    'temp': args.temperature,
                    'search_d': get_dataset_dimension(args.search_dataset),
                    'target_d': get_dataset_dimension(args.dataset),
                    'gf_model_name': args.gf_model_name}
        keepaug_config = {'keep_aug':args.keep_aug,'mode':args.keep_mode,'thres':args.keep_thres,'length':args.keep_len,
            'grid_region':args.keep_grid, 'possible_segment': args.keep_seg, 'info_upper': args.keep_bound, 'adapt_target':args.adapt_target}
        if args.keep_mode=='adapt':
            keepaug_config['mode'] = 'auto'
            self.adaaug = AdaAugkeep_TS(after_transforms=after_transforms,
                n_class=n_class,
                gf_model=self.gf_model,
                h_model=self.h_model,
                save_dir=os.path.join(self.config['BASE_PATH'],self.config['save'],f'fold{test_fold_idx}'),
                visualize=args.visualize,
                config=adaaug_config,
                keepaug_config=keepaug_config,
                multilabel=multilabel,
                augselect=args.augselect,
                class_adaptive=self.class_noaug,
                noaug_add=self.noaug_add)
        else:
            keepaug_config['length'] = keepaug_config['length'][0]
            self.adaaug = AdaAug_TS(after_transforms=after_transforms,
                n_class=n_class,
                gf_model=self.gf_model,
                h_model=self.h_model,
                save_dir=os.path.join(self.config['BASE_PATH'],self.config['save'],f'fold{test_fold_idx}'),
                visualize=args.visualize,
                config=adaaug_config,
                keepaug_config=keepaug_config,
                multilabel=multilabel,
                augselect=args.augselect,
                class_adaptive=self.class_noaug,
                noaug_add=self.noaug_add)
        #to self
        self.n_channel = n_channel
        self.n_class = n_class
        self.sdiv = sdiv
        self.class2label = class2label
        self.multilabel = multilabel
        self.test_fold_idx = test_fold_idx
        self.diff_augment = diff_augment
        self.diff_reweight =diff_reweight
        self.result_valid_dic, self.result_test_dic = {}, {}
        self.best_val_acc = -1
        self.best_task = None
        self.base_path = self.config['BASE_PATH']
        self.mapselect = self.config['mapselect']
        self.pre_train_acc = 0.0
    def step(self):#use step replace _train
        if self._iteration==0:
            wandb.config.update(self.config)
        if self.multilabel:
            if self.mapselect:
                ptype = 'map'
            else:
                ptype = 'auroc'
        else:
            ptype = 'acc'
        print(f'Starting Ray ID {self.trial_id} Iteration: {self._iteration}')
        args = self.config['args']
        step_dic={'epoch':self._iteration}
        diff_dic = {'difficult_aug':self.diff_augment,'reweight':self.diff_reweight,'lambda_aug':args.lambda_aug, 'class_adaptive':args.class_adapt
                }
        # training
        train_acc, train_obj, train_dic = train(args,
            self.train_queue, self.task_model, self.criterion, self.optimizer, self.scheduler, self._iteration, args.grad_clip, self.adaaug, 
                multilabel=self.multilabel,n_class=self.n_class,map_select=self.mapselect,**diff_dic)
        if self.noaug_add:
            class_acc = train_acc / 100.0
            if self.class_noaug:
                class_acc = [train_dic[f'train_{ptype}_c{i}'] / 100.0 for i in range(self.n_class)]
            self.adaaug.update_alpha(class_acc)
        self.pre_train_acc = train_acc / 100.0
        # validation
        valid_acc, valid_obj, _, _, valid_dic = infer(self.valid_queue, self.task_model, self.criterion, multilabel=self.multilabel,
                n_class=self.n_class,mode='valid',map_select=self.mapselect)
        #test
        test_acc, test_obj, test_acc5, _,test_dic  = infer(self.test_queue, self.task_model, self.criterion, multilabel=self.multilabel,
                n_class=self.n_class,mode='test',map_select=self.mapselect)
        #val select
        if args.valselect and valid_acc>self.best_val_acc:
            self.best_val_acc = valid_acc
            self.result_valid_dic = {f'result_{k}': valid_dic[k] for k in valid_dic.keys()}
            self.result_test_dic = {f'result_{k}': test_dic[k] for k in test_dic.keys()}
            valid_dic[f'best_valid_{ptype}_avg'] = valid_acc
            test_dic[f'best_test_{ptype}_avg'] = test_acc
            self.best_task = self.task_model
        elif not args.valselect:
            self.best_task = self.task_model
            self.result_valid_dic = {f'result_{k}': valid_dic[k] for k in valid_dic.keys()}
            self.result_test_dic = {f'result_{k}': test_dic[k] for k in test_dic.keys()}

        #utils.save_model(self.best_task, os.path.join(self.base_path,self.config['save'],f'fold{self.test_fold_idx}', 'h_weights.pt'))
        utils.save_ckpt(self.best_task, self.optimizer, self.scheduler, self._iteration,
            os.path.join(self.base_path,self.config['save'],f'fold{self.test_fold_idx}', 'weights.pt'))
        step_dic.update(test_dic)
        step_dic.update(train_dic)
        step_dic.update(valid_dic)
        wandb.log(step_dic)
        #if last epoch
        if self._iteration==self.config['epochs']-1:
            step_dic.update(self.result_valid_dic)
            step_dic.update(self.result_test_dic)
            #save&log
            self.adaaug.save_history(self.class2label)
            figure = self.adaaug.plot_history()
            wandb.log(step_dic)
        call_back_dic = {'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc}
        return call_back_dic

    def _save(self, checkpoint_dir):
        print(checkpoint_dir)
        path = os.path.join(checkpoint_dir, 'gf_check_weights.pt')
        utils.save_model(self.best_task, path)
        print(path)
        return path

    def _restore(self, checkpoint_path):
        utils.load_model(self.task_model, f'{checkpoint_path}/gf_check_weights.pt', location=0) #0 as default

    def reset_config(self, new_config):
        self.config = new_config
        return True

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    #torch.cuda.set_device(args.gpu)
    utils.reproducibility(args.seed)
    #logging.info('gpu device = %d' % args.gpu)
    #logging.info("args = %s", args)
    date_time_str = now_str
    h_model_dir = args.h_model_path
    h_model_dir = h_model_dir.strip('/').split('/')[-1][:16]
    #wandb
    group_name = f'{Aug_type}_tottrain{args.augselect}_{args.dataset}{args.labelgroup}_{args.model_name}_hmodel{h_model_dir}_e{args.epochs}_lr{args.learning_rate}'
    experiment_name = f'{Aug_type}{args.k_ops}{description}_tottrain{args.augselect}_{args.dataset}{args.labelgroup}_{args.model_name}_{h_model_dir}_e{args.epochs}_lr{args.learning_rate}'
    '''run_log = wandb.init(config=args, 
                  project='AdaAug',
                  group=experiment_name,
                  name=f'{now_str}_' + experiment_name,
                  dir='./',
                  job_type="DataAugment",
                  reinit=True)'''
    #hparams
    hparams = dict(vars(args)) #copy args
    hparams['args'] = args
    if args.kfold==10:
        hparams['kfold'] = tune.grid_search([i for i in range(args.kfold)])
    else:
        hparams['kfold'] = tune.grid_search([args.kfold]) #for some fold
    #wandb
    wandb_config = {
        #'config':FLAGS, 
        'project':'AdaAug',
        'group':f'{now_str}_' + experiment_name,
        #'name':experiment_name,
        'dir':'./',
        'job_type':"DataAugment",
        'reinit':False,
        'api_key':API_KEY
    }
    hparams["log_config"]= False
    hparams['wandb'] = wandb_config
    hparams['BASE_PATH'] = args.base_path

    # if FLAGS.restore:
    #     train_spec["restore"] = FLAGS.restore

    ray.init()
    print(f'Run {args.kfold} folds experiment')
    #tune_scheduler = ASHAScheduler(metric="valid_acc", mode="max",max_t=hparams['num_epochs'],grace_period=10,
    #    reduction_factor=3,brackets=1)1
    tune_scheduler = None
    analysis = tune.run(
        RayModel,
        name=hparams['ray_name'],
        scheduler=tune_scheduler,
        #reuse_actors=True,
        verbose=True,
        metric="valid_acc",
        mode='max',
        checkpoint_score_attr="valid_acc",
        #checkpoint_freq=FLAGS.checkpoint_freq,
        resources_per_trial={"gpu": args.gpu, "cpu": args.cpu},
        stop={"training_iteration": hparams['epochs']},
        config=hparams,
        local_dir=args.ray_dir,
        num_samples=1, #grid search no need
    )
    
    print("Best hyperparameters found were: ")
    print(analysis.best_config)
    print(analysis.best_trial)
    
    wandb.finish()
    


if __name__ == '__main__':
    main()
