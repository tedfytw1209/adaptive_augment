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
from dataset import get_ts_dataloaders, get_num_class, get_label_name, get_dataset_dimension,get_num_channel,Freq_dict,TimeS_dict
from operation_tseries import TS_OPS_NAMES,ECG_OPS_NAMES,TS_ADD_NAMES,MAG_TEST_NAMES,NOMAG_TEST_NAMES
from step_function import train,infer,search_train,search_infer
from warmup_scheduler import GradualWarmupScheduler
from config import get_warmup_config
from class_balanced_loss import ClassBalLoss,ClassDiffLoss,ClassDistLoss,make_class_balance_count,make_class_weights,make_loss,make_class_weights_maxrel \
    ,make_class_weights_samples
import wandb
from utils import plot_conf_wandb
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
parser.add_argument('--search_size', type=float, default=0, help='use search size to reduce train data')
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
### dataset & params
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--multilabel', action='store_true', default=False, help='using multilabel default False')
parser.add_argument('--train_portion', type=float, default=1, help='portion of training data')
parser.add_argument('--default_split', action='store_true', help='use dataset deault split')
parser.add_argument('--kfold', type=int, default=0, help='use kfold cross validation')
#policy
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
#restore
parser.add_argument('--restore_path', type=str, default='./', help='restore model path')
parser.add_argument('--restore', action='store_true', default=False, help='restore model default False')
#select&search
parser.add_argument('--mapselect', action='store_true', default=False, help='use map select for multilabel')
parser.add_argument('--valselect', action='store_true', default=False, help='use valid select')
parser.add_argument('--notwarmup', action='store_true', default=False, help='use valid select')
parser.add_argument('--randaug', action='store_true', default=False, help="mimic randaug training")
parser.add_argument('--augselect', type=str, default='', help="augmentation selection")
parser.add_argument('--alpha', type=float, default=1.0, help="alpha adpat")
parser.add_argument('--train_sampler', type=str, default='', help='for train sampler',
        choices=['weight','wmaxrel',''])
#diff
parser.add_argument('--diff_aug', action='store_true', default=False, help='use valid select')
parser.add_argument('--not_reweight', action='store_true', default=False, help='use valid select')
parser.add_argument('--lambda_aug', type=float, default=1.0, help="augment sample weight")
#class adapt
parser.add_argument('--class_adapt', action='store_true', default=False, help='class adaptive')
parser.add_argument('--class_embed', action='store_true', default=False, help='class embed') #tmp use
parser.add_argument('--feature_mask', type=str, default='', help='add regular for noaugment ',
        choices=['dropout','select','average','classonly',''])
parser.add_argument('--noaug_reg', type=str, default='', help='add regular for noaugment ',
        choices=['cadd','add',''])
parser.add_argument('--balance_loss', type=str, default='', help="loss type for model and policy training to acheive class balance")
#info keep
parser.add_argument('--keep_aug', action='store_true', default=False, help='info keep augment')
parser.add_argument('--keep_mode', type=str, default='auto', help='info keep mode',choices=['auto','adapt','b','p','t','rand'])
parser.add_argument('--keep_prob', type=float, default=1, help='info keep probabilty')
parser.add_argument('--adapt_target', type=str, default='len', help='info keep mode',
        choices=['fea','len','seg','way','keep','ch','recut','repaste','recut','repaste'])
parser.add_argument('--keep_back', type=str, default='', help='info keep how to paste back',
        choices=['fix','rpeak',''])
parser.add_argument('--keep_seg', type=int, nargs='+', default=[1], help='info keep segment mode')
parser.add_argument('--keep_lead', type=int, nargs='+', default=[12], help='leads (channel) keep, 12 means all lead keep')
parser.add_argument('--lead_sel', type=str, default='thres', help='leads select ways',
        choices=['max','prob','thres','group'])
parser.add_argument('--keep_grid', action='store_true', default=False, help='info keep augment grid')
parser.add_argument('--keep_thres', type=float, default=0.6, help="keep augment weight (lower protect more)")
parser.add_argument('--thres_adapt', action='store_false', default=True, help="keep augment thres adapt")
parser.add_argument('--keep_len', type=int, nargs='+', default=[100], help="info keep seq len")
parser.add_argument('--keep_bound', type=float, default=0.0, help="info keep bound %")
parser.add_argument('--teach_rew', action='store_true', default=False, help='teach reweight')
#visulaize&output
parser.add_argument('--visualize', action='store_true', default=False, help='visualize')
parser.add_argument('--output_visual', action='store_true', default=False, help='visualize output and confusion matrix')
parser.add_argument('--output_pred', action='store_true', default=False, help='output predict result and ture target')

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
description += args.noaug_reg
if args.diff_aug and not args.not_reweight:
    description+='rew'
if args.keep_aug:
    keep_ch_str = ''.join([str(i) for i in args.keep_lead])
    if keep_ch_str=='12':
        keep_ch_str=''
    else:
        keep_ch_str='ch'+keep_ch_str
    keep_seg_str = ''.join([str(i) for i in args.keep_seg])
    description+=f'keep{args.keep_mode}{keep_seg_str}{keep_ch_str}'
now_str = time.strftime("%Y%m%d-%H%M%S")
args.save = '{}-{}-{}{}'.format(now_str, args.save,Aug_type,description+args.augselect+args.balance_loss)
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
        #os.environ['WANDB_START_METHOD'] = 'thread' #tmp disable
        args = self.config['args']
        #random seed setting
        utils.reproducibility(args.seed)
        #  dataset settings for search
        n_channel = get_num_channel(args.dataset)
        n_class = get_num_class(args.dataset,args.labelgroup)
        sdiv = get_search_divider(args.model_name)
        class2label = get_label_name(args.dataset, args.dataroot,args.labelgroup)
        self.sfreq = Freq_dict[args.dataset]
        self.time_s = TimeS_dict[args.dataset]
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
        self.train_queue, self.valid_queue, self.search_queue, self.test_queue, self.tr_search_queue, preprocessors = get_ts_dataloaders(
            args.dataset, args.batch_size, args.num_workers,
            args.dataroot, args.cutout, args.cutout_length,
            split=args.train_portion, split_idx=0, target_lb=-1,search_size=args.search_size, #!use search size to reduce training dataset
            search=False,test_size=args.test_size,multilabel=args.multilabel,default_split=args.default_split,
            fold_assign=train_val_test_folds,labelgroup=args.labelgroup,bal_trsampler=args.train_sampler,
            sampler_alpha=args.alpha)
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
        default_criterion = make_loss(multilabel=multilabel)
        train_labels = self.train_queue.dataset.dataset.label
        self.criterion = self.choose_criterion(train_labels,None,multilabel,n_class,
            loss_type=args.balance_loss,default=default_criterion)
        self.criterion = self.criterion.cuda()
        #  restore setting
        if args.restore:
            trained_epoch = utils.restore_ckpt(self.task_model, self.optimizer, self.scheduler,
                os.path.join(self.config['BASE_PATH'],args.restore_path,f'fold{test_fold_idx}', 'weights.pt'), location=0) + 1
            trained_epoch = args.epochs #tmp fix!!!
            print(f'From epoch {trained_epoch}, Resume')
        else:
            trained_epoch = 0
        self.trained_epoch = trained_epoch
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
        #keep aug, need improve!!
        proj_add = 0
        if args.keep_mode=='adapt':
            if args.adapt_target=='len':
                proj_add = len(args.keep_len) + 1
            elif args.adapt_target=='seg':
                proj_add = len(args.keep_seg) + 1
            elif args.adapt_target=='way':
                proj_add = 4 + 1
            elif args.adapt_target=='keep':
                proj_add = 2 + 1
        self.h_model = Projection_TSeries(in_features=h_input,label_num=label_num,label_embed=label_embed,
            n_layers=args.n_proj_layer, n_hidden=args.n_proj_hidden, augselect=args.augselect, proj_addition=proj_add,
            feature_mask=args.feature_mask).cuda()
        utils.load_model(self.gf_model, os.path.join(self.config['BASE_PATH'],f'{args.gf_model_path}',f'fold{test_fold_idx}', 'gf_weights.pt'), location=0)
        utils.load_model(self.h_model, os.path.join(self.config['BASE_PATH'],f'{args.h_model_path}',f'fold{test_fold_idx}', 'h_weights.pt'), location=0)
        for param in self.gf_model.parameters():
            param.requires_grad = False
        for param in self.h_model.parameters():
            param.requires_grad = False
        self.teach_model = None
        if args.teach_rew: #teacher reweight, only use when search&train same
            self.teach_model = self.gf_model
        #AdaAug / Keep
        after_transforms = self.train_queue.dataset.after_transforms
        adaaug_config = {'sampling': 'prob',
                    'k_ops': args.k_ops,
                    'delta': args.delta,
                    'temp': args.temperature,
                    'search_d': get_dataset_dimension(args.search_dataset),
                    'target_d': get_dataset_dimension(args.dataset),
                    'gf_model_name': args.gf_model_name}
        keepaug_config = {'keep_aug':args.keep_aug,'mode':args.keep_mode,'thres':args.keep_thres,'length':args.keep_len,'thres_adapt':args.thres_adapt,
            'grid_region':args.keep_grid, 'possible_segment': args.keep_seg, 'info_upper': args.keep_bound, 'sfreq':self.sfreq,
            'adapt_target':args.adapt_target,'keep_leads':args.keep_lead,'keep_prob':args.keep_prob,'keep_back':args.keep_back,'lead_sel':args.lead_sel}
        trans_config = {'sfreq':self.sfreq}
        if args.keep_mode=='adapt':
            keepaug_config['mode'] = 'auto'
            self.adaaug = AdaAugkeep_TS(after_transforms=after_transforms,
                n_class=n_class,
                gf_model=self.gf_model,
                h_model=self.h_model,
                save_dir=os.path.join(self.config['BASE_PATH'],self.config['save'],f'fold{test_fold_idx}'),
                #visualize=args.visualize,
                config=adaaug_config,
                keepaug_config=keepaug_config,
                multilabel=multilabel,
                augselect=args.augselect,
                class_adaptive=self.class_noaug,
                noaug_add=self.noaug_add,
                transfrom_dic=trans_config,
                preprocessors=preprocessors)
        else:
            keepaug_config['length'] = keepaug_config['length'][0]
            self.adaaug = AdaAug_TS(after_transforms=after_transforms,
                n_class=n_class,
                gf_model=self.gf_model,
                h_model=self.h_model,
                save_dir=os.path.join(self.config['BASE_PATH'],self.config['save'],f'fold{test_fold_idx}'),
                #visualize=args.visualize,
                config=adaaug_config,
                keepaug_config=keepaug_config,
                multilabel=multilabel,
                augselect=args.augselect,
                class_adaptive=self.class_noaug,
                noaug_add=self.noaug_add,
                transfrom_dic=trans_config,
                preprocessors=preprocessors)
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
        self.result_table_dic = {}
        self.policy_apply = not args.randaug
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
        Curr_epoch = self.trained_epoch + self._iteration
        args = self.config['args']
        step_dic={'epoch':Curr_epoch}
        diff_dic = {'difficult_aug':self.diff_augment,'reweight':self.diff_reweight,'lambda_aug':args.lambda_aug, 'class_adaptive':args.class_adapt
                ,'visualize':args.visualize,'teach_rew':self.teach_model,'policy_apply':self.policy_apply,'noaug_reg':args.noaug_reg}
        if Curr_epoch>self.config['epochs']:
            all_epochs = self.config['epochs']-1
            print(f'Trained epochs {Curr_epoch} Iteration: {self._iteration} already reach {all_epochs}, Skip step')
            call_back_dic = {'train_acc': 0, 'valid_acc': 0, 'test_acc': 0}
            return call_back_dic
        elif Curr_epoch==self.config['epochs']:
            print('Evaluating Train/Valid/Test dataset')
            training = False
        else:
            training = True
        # training or evaluate training data
        train_acc, train_obj, train_dic, train_table = train(args,
                self.train_queue, self.task_model, self.criterion, self.optimizer, self.scheduler, Curr_epoch, args.grad_clip, self.adaaug, 
                multilabel=self.multilabel,n_class=self.n_class,map_select=self.mapselect,training=training,**diff_dic)
        if self.noaug_add:
            class_acc = train_acc / 100.0
            if self.class_noaug:
                class_acc = [train_dic[f'train_{ptype}_c{i}'] / 100.0 for i in range(self.n_class)]
            self.adaaug.update_alpha(class_acc)
        self.pre_train_acc = train_acc / 100.0
        # validation
        valid_acc, valid_obj, _, _, valid_dic, valid_table = infer(self.valid_queue, self.task_model, self.criterion, multilabel=self.multilabel,
                n_class=self.n_class,mode='valid',map_select=self.mapselect)
        #test
        test_acc, test_obj, test_acc5, _,test_dic, test_table  = infer(self.test_queue, self.task_model, self.criterion, multilabel=self.multilabel,
                n_class=self.n_class,mode='test',map_select=self.mapselect)
        #val select 10/31 debug
        if args.valselect and valid_acc>self.best_val_acc:
            self.best_val_acc = valid_acc
            self.result_valid_dic = {f'result_{k}': valid_dic[k] for k in valid_dic.keys()}
            self.result_test_dic = {f'result_{k}': test_dic[k] for k in test_dic.keys()}
            valid_dic[f'best_valid_{ptype}_avg'] = valid_acc
            test_dic[f'best_test_{ptype}_avg'] = test_acc
            self.result_table_dic.update(train_table)
            self.result_table_dic.update(valid_table)
            self.result_table_dic.update(test_table)
            self.best_task = self.task_model
            if 'debug' not in self.config['save'] and not args.restore:
                utils.save_ckpt(self.best_task, self.optimizer, self.scheduler, Curr_epoch,
                    os.path.join(self.base_path,self.config['save'],f'fold{self.test_fold_idx}', 'weights.pt'))
            else:
                print('Debuging: not save at', self.config['save'])
        elif not args.valselect:
            self.best_task = self.task_model
            self.result_valid_dic = {f'result_{k}': valid_dic[k] for k in valid_dic.keys()}
            self.result_test_dic = {f'result_{k}': test_dic[k] for k in test_dic.keys()}
            self.result_table_dic.update(train_table)
            self.result_table_dic.update(valid_table)
            self.result_table_dic.update(test_table)
            if 'debug' not in self.config['save'] and not args.restore:
                utils.save_ckpt(self.best_task, self.optimizer, self.scheduler, Curr_epoch,
                    os.path.join(self.base_path,self.config['save'],f'fold{self.test_fold_idx}', 'weights.pt'))
            else:
                print('Debuging: not save at', self.config['save'])
        step_dic.update(test_dic)
        step_dic.update(train_dic)
        step_dic.update(valid_dic)
        wandb.log(step_dic)
        #if last epoch
        if Curr_epoch==self.config['epochs']-1 or Curr_epoch==self.config['epochs']:
            step_dic.update(self.result_valid_dic)
            step_dic.update(self.result_test_dic)
            #output pred
            if args.output_pred:
                utils.save_pred(self.result_table_dic['valid_target'],self.result_table_dic['valid_predict'],
                        os.path.join(self.base_path,self.config['save'],f'fold{self.test_fold_idx}', 'valid_prediction.csv'))
                utils.save_pred(self.result_table_dic['test_target'],self.result_table_dic['test_predict'],
                        os.path.join(self.base_path,self.config['save'],f'fold{self.test_fold_idx}', 'test_prediction.csv'))
            #save&log
            wandb.log(step_dic)
            if args.output_visual:
                tables_dic = {}
                tables_dic['train_confusion']=plot_conf_wandb(self.result_table_dic['train_confusion'],title='train_confusion')
                tables_dic['valid_confusion']=plot_conf_wandb(self.result_table_dic['valid_confusion'],title='valid_confusion')
                tables_dic['test_confusion']=plot_conf_wandb(self.result_table_dic['test_confusion'],title='test_confusion')
                #! maybe will bug
                tables_dic['train_output']=plot_conf_wandb(self.result_table_dic['train_output'],title='train_output')
                tables_dic['valid_output']=plot_conf_wandb(self.result_table_dic['valid_output'],title='valid_output')
                tables_dic['test_output']=plot_conf_wandb(self.result_table_dic['test_output'],title='test_output')
                wandb.log(tables_dic)
            if Curr_epoch==self.config['epochs']-1:
                self.adaaug.save_history(self.class2label)
                figure = self.adaaug.plot_history()
            wandb.finish()
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
    
    def choose_criterion(self,train_labels,search_labels,multilabel,n_class,loss_type='',mix_type='',default=None):
        if loss_type=='classbal': 
            out_criterion = make_class_balance_count(train_labels,search_labels,multilabel,n_class)
        elif loss_type=='classdiff':
            self.class_difficulty = np.ones(n_class)
            gamma = None
            if multilabel:
                gamma = 2.0
            out_criterion = ClassDiffLoss(class_difficulty=self.class_difficulty,focal_gamma=gamma).cuda() #default now
        elif loss_type=='classweight':
            class_weights = make_class_weights(train_labels,n_class,search_labels)
            class_weights = torch.from_numpy(class_weights).float()
            out_criterion = make_loss(multilabel=multilabel,weight=class_weights).cuda()
        elif loss_type=='classwmaxrel':
            class_weights = make_class_weights_maxrel(train_labels,n_class,search_labels)
            class_weights = torch.from_numpy(class_weights).float()
            out_criterion = make_loss(multilabel=multilabel,weight=class_weights).cuda()
        elif mix_type=='loss':
            if not multilabel:
                out_criterion = nn.CrossEntropyLoss(reduction='none').cuda()
            else:
                out_criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()
        else:
            out_criterion = default
        return out_criterion

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
    group_name = f'{Aug_type}_train{args.augselect}_{args.dataset}{args.labelgroup}_{args.model_name}_hmodel{h_model_dir}'
    if args.search_size>0: #reduce train
        reduce_train = 1.0 - args.search_size
        data_add = '_r' + str(round(reduce_train,3))
    else:
        data_add = ''
    Mode = 'train'
    experiment_name = f'{Aug_type}{args.k_ops}{description}_{Mode}{args.augselect}_{args.dataset}{args.labelgroup}{data_add}_{args.model_name}_{h_model_dir}'
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
    
    wandb.finish()
    


if __name__ == '__main__':
    main()
