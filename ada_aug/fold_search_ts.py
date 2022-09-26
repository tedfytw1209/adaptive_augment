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
import torch.optim as optim

from adaptive_augmentor import AdaAug_TS,AdaAugkeep_TS
from networks import get_model_tseries
from networks.projection import Projection_TSeries
from config import get_search_divider
from dataset import get_ts_dataloaders, get_num_class, get_label_name, get_dataset_dimension,get_num_channel
from operation_tseries import TS_OPS_NAMES,ECG_OPS_NAMES,TS_ADD_NAMES,MAG_TEST_NAMES,NOMAG_TEST_NAMES
from step_function import train,infer,search_train,search_infer
from non_saturating_loss import NonSaturatingLoss
from class_balanced_loss import ClassBalLoss,ClassDiffLoss
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
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--multilabel', action='store_true', default=False, help='using multilabel default False')
parser.add_argument('--train_portion', type=float, default=1, help='portion of training data')
parser.add_argument('--default_split', action='store_true', help='use dataset deault split')
parser.add_argument('--kfold', type=int, default=-1, help='use kfold cross validation')
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
parser.add_argument('--search_round', type=int, default=1, help='exploration frequency') #search_round
parser.add_argument('--n_proj_layer', type=int, default=0, help="number of hidden layer in augmentation policy projection")
parser.add_argument('--n_proj_hidden', type=int, default=128, help="number of hidden units in augmentation policy projection layers")
parser.add_argument('--mapselect', action='store_true', default=False, help='use map select for multilabel')
parser.add_argument('--valselect', action='store_true', default=False, help='use valid select')
parser.add_argument('--augselect', type=str, default='', help="augmentation selection")
parser.add_argument('--diff_aug', action='store_true', default=False, help='use valid select')
parser.add_argument('--same_train', action='store_true', default=False, help='use valid select')
parser.add_argument('--not_mix', action='store_true', default=False, help='use valid select')
parser.add_argument('--not_reweight', action='store_true', default=False, help='use diff reweight')
parser.add_argument('--sim_rew', action='store_true', default=False, help='use sim reweight')
parser.add_argument('--pwarmup', type=int, default=0, help="warmup epoch for policy")
parser.add_argument('--lambda_aug', type=float, default=1.0, help="augment sample weight (difficult)")
parser.add_argument('--lambda_sim', type=float, default=1.0, help="augment sample weight (simular)")
parser.add_argument('--lambda_noaug', type=float, default=0, help="no augment regular weight")
parser.add_argument('--class_adapt', action='store_true', default=False, help='class adaptive')
parser.add_argument('--class_embed', action='store_true', default=False, help='class embed') #tmp use
parser.add_argument('--noaug_reg', type=str, default='', help='add regular for noaugment ',
        choices=['cadd','add',''])
parser.add_argument('--loss_type', type=str, default='minus', help="loss type for difficult policy training",
        choices=['minus','relative','relativediff','adv'])
parser.add_argument('--policy_loss', type=str, default='', help="loss type for simular policy training")
parser.add_argument('--keep_aug', action='store_true', default=False, help='info keep augment')
parser.add_argument('--keep_mode', type=str, default='auto', help='info keep mode',choices=['auto','adapt','b','p','t'])
parser.add_argument('--keep_seg', type=int, nargs='+', default=[1], help='info keep segment mode')
parser.add_argument('--keep_grid', action='store_true', default=False, help='info keep augment grid')
parser.add_argument('--keep_thres', type=float, default=0.6, help="keep augment weight (lower protect more)")
parser.add_argument('--keep_len', type=int, nargs='+', default=[100], help="info keep seq len")
parser.add_argument('--keep_bound', type=float, default=0.0, help="info keep bound %")
parser.add_argument('--teach_aug', action='store_true', default=False, help='teacher augment')
parser.add_argument('--ema_rate', type=float, default=0.999, help="teacher ema rate")
parser.add_argument('--visualize', action='store_true', default=False, help='visualize')

args = parser.parse_args()
debug = True if args.save == "debug" else False
if args.k_ops>0:
    Aug_type = 'AdaAug'
else:
    Aug_type = 'NOAUG'
if args.diff_aug:
    description = 'diff2'
    description += args.loss_type
else:
    description = ''
if args.class_adapt:
    description += f'cada{args.policy_loss}'
else:
    description += ''
if args.diff_aug and not args.not_reweight:
    description+='rew'
if args.keep_aug:
    keep_seg_str = ''.join([str(i) for i in args.keep_seg])
    description+=f'keep{args.keep_mode}{keep_seg_str}'
if args.teach_aug:
    description+=f'teach{args.ema_rate}'
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
        utils.reproducibility(args.seed) #for reproduce
        #  dataset settings for search
        n_channel = get_num_channel(args.dataset)
        n_class = get_num_class(args.dataset,args.labelgroup)
        sdiv = get_search_divider(args.model_name)
        class2label = get_label_name(args.dataset, args.dataroot)
        multilabel = args.multilabel
        diff_augment = args.diff_aug
        diff_mix = not args.not_mix
        diff_reweight = not args.not_reweight
        self.class_noaug, self.noaug_add = False, False
        if args.noaug_reg=='cadd':
            self.class_noaug = True
            self.noaug_add = True
        elif args.noaug_reg=='add':
            self.noaug_add = True
        test_fold_idx = self.config['kfold']
        train_val_test_folds = []
        if test_fold_idx>=0 and test_fold_idx<=9:
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
        elif args.default_split:
            print('Default Train 1~8 /Valid 9/Test 10 fold split')
        else:
            print('Warning: Random splitting train/test')
        self.train_queue, self.valid_queue, self.search_queue, self.test_queue, self.tr_search_queue = get_ts_dataloaders(
            args.dataset, args.batch_size, args.num_workers,
            args.dataroot, args.cutout, args.cutout_length,
            split=args.train_portion, split_idx=0, target_lb=-1,
            search=True, search_divider=sdiv,search_size=args.search_size,
            test_size=args.test_size,multilabel=args.multilabel,default_split=args.default_split,
            fold_assign=train_val_test_folds,labelgroup=args.labelgroup)
        #  model settings
        self.gf_model = get_model_tseries(model_name=args.model_name, num_class=n_class,n_channel=n_channel,
            use_cuda=True, data_parallel=False,dataset=args.dataset)
        #EMA if needed
        self.ema_model = None
        if args.teach_aug:
            print('Using EMA teacher model')
            avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
                args.ema_rate * averaged_model_parameter + (1 - args.ema_rate) * model_parameter
            self.ema_model = optim.swa_utils.AveragedModel(self.gf_model, avg_fn=avg_fn)
            for ema_p in self.ema_model.parameters():
                ema_p.requires_grad_(False)
            self.ema_model.train()
        #h model
        h_input = self.gf_model.fc.in_features
        label_num, label_embed = 0,0
        if args.class_adapt and args.class_embed:
            label_num = n_class
            label_embed = 32 #tmp use
            h_input = h_input + label_embed
        elif args.class_adapt:
            h_input =h_input + n_class
        #keep aug
        proj_add = 0
        if args.keep_mode=='adapt':
            proj_add = len(args.keep_len) + 1
        self.h_model = Projection_TSeries(in_features=h_input,label_num=label_num,label_embed=label_embed,
            n_layers=args.n_proj_layer, n_hidden=args.n_proj_hidden, augselect=args.augselect, proj_addition=proj_add).cuda()
        #  training settings
        self.gf_optimizer = torch.optim.AdamW(self.gf_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) #follow ptbxl batchmark
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.gf_optimizer, max_lr=args.learning_rate, 
            epochs = args.epochs, steps_per_epoch = len(self.train_queue)) #follow ptbxl batchmark
        self.h_optimizer = torch.optim.Adam(
            self.h_model.parameters(),
            lr=args.proj_learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.proj_weight_decay)
        
        if not multilabel:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.criterion = self.criterion.cuda()
        self.adv_criterion = None
        if args.loss_type=='adv':
            self.adv_criterion = NonSaturatingLoss(epsilon=0.1).cuda()
        self.sim_criterion = None
        if args.policy_loss=='classbal':
            search_labels = self.search_queue.dataset.dataset.label
            if not multilabel:
                search_labels_count = [np.count_nonzero(search_labels == i) for i in range(n_class)] #formulticlass
            else:
                search_labels_count = np.sum(search_labels,axis=0)
            sim_type = 'softmax'
            if multilabel:
                sim_type = 'focal'
            self.sim_criterion = ClassBalLoss(search_labels_count,len(search_labels_count),loss_type=sim_type).cuda()
        elif args.policy_loss=='classdiff':
            self.class_difficulty = np.ones(n_class)
            gamma = None
            if multilabel:
                gamma = 2.0
            self.sim_criterion = ClassDiffLoss(class_difficulty=self.class_difficulty,focal_gamma=gamma).cuda() #default now

        #  AdaAug settings for search
        after_transforms = self.train_queue.dataset.after_transforms
        adaaug_config = {'sampling': 'prob',
                    'k_ops': self.config['k_ops'], #as paper
                    'delta': 0.0, 
                    'temp': 1.0, 
                    'search_d': get_dataset_dimension(args.dataset),
                    'target_d': get_dataset_dimension(args.dataset),
                    'gf_model_name': args.model_name}
        keepaug_config = {'keep_aug':args.keep_aug,'mode':args.keep_mode,'thres':args.keep_thres,'length':args.keep_len,
            'grid_region':args.keep_grid, 'possible_segment': args.keep_seg, 'info_upper': args.keep_bound}
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
                class_adaptive=args.class_adapt,
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
                class_adaptive=args.class_adapt,
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
        self.best_gf,self.best_h = None,None
        self.base_path = self.config['BASE_PATH']
        self.mapselect = self.config['mapselect']
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
        lr = self.scheduler.get_last_lr()[0]
        step_dic={'epoch':self._iteration}
        diff_dic = {'difficult_aug':self.diff_augment,'same_train':args.same_train,'reweight':self.diff_reweight,'lambda_aug':args.lambda_aug,
                'lambda_sim':args.lambda_sim,'class_adaptive':args.class_adapt,'lambda_noaug':args.lambda_noaug,
                'loss_type':args.loss_type, 'adv_criterion': self.adv_criterion, 'teacher_model':self.ema_model, 'sim_criterion':self.sim_criterion,
                'sim_reweight':args.sim_rew,'warmup_epoch': args.pwarmup}
        # searching
        train_acc, train_obj, train_dic = search_train(args,self.train_queue, self.search_queue, self.tr_search_queue, self.gf_model, self.adaaug,
            self.criterion, self.gf_optimizer,self.scheduler, args.grad_clip, self.h_optimizer, self._iteration, args.search_freq, 
            search_round=args.search_round,multilabel=self.multilabel,n_class=self.n_class,map_select=self.mapselect, **diff_dic)
        if self.noaug_add:
            class_acc = train_acc / 100.0
            self.adaaug.update_alpha(class_acc)
        # validation
        valid_acc, valid_obj,valid_dic = search_infer(self.valid_queue, self.gf_model, self.criterion, 
            multilabel=self.multilabel,n_class=self.n_class,mode='valid',map_select=self.mapselect)
        if args.policy_loss=='classdiff':
            class_acc = [valid_dic[f'valid_{ptype}_c{i}'] / 100.0 for i in range(self.n_class)]
            print(class_acc)
            self.class_difficulty = 1 - np.array(class_acc)
            self.sim_criterion.update_weight(self.class_difficulty)
        #scheduler.step()
        #test
        test_acc, test_obj, test_dic  = search_infer(self.test_queue, self.gf_model, self.criterion, 
            multilabel=self.multilabel,n_class=self.n_class,mode='test',map_select=self.mapselect)
        #val select
        if args.valselect and valid_acc>self.best_val_acc:
            self.best_val_acc = valid_acc
            self.result_valid_dic = {f'result_{k}': valid_dic[k] for k in valid_dic.keys()}
            self.result_test_dic = {f'result_{k}': test_dic[k] for k in test_dic.keys()}
            valid_dic[f'best_valid_{ptype}_avg'] = valid_acc
            test_dic[f'best_test_{ptype}_avg'] = test_acc
            self.best_gf = self.gf_model
            self.best_h = self.h_model
        elif not args.valselect:
            self.best_gf = self.gf_model
            self.best_h = self.h_model
            self.result_valid_dic = {f'result_{k}': valid_dic[k] for k in valid_dic.keys()}
            self.result_test_dic = {f'result_{k}': test_dic[k] for k in test_dic.keys()}
        if self.test_fold_idx>=0:
            gf_path = os.path.join(f'fold{self.test_fold_idx}', 'gf_weights.pt')
            h_path = os.path.join(f'fold{self.test_fold_idx}', 'h_weights.pt')
        else:
            gf_path = 'gf_weights.pt'
            h_path = 'h_weights.pt'
        utils.save_model(self.best_gf, os.path.join(self.base_path,self.config['save'],gf_path))
        utils.save_model(self.best_h, os.path.join(self.base_path,self.config['save'],h_path))
        
        step_dic.update(test_dic)
        step_dic.update(train_dic)
        step_dic.update(valid_dic)
        wandb.log(step_dic)
        #if last epoch
        if self._iteration==self.config['epochs']-1:
            step_dic.update(self.result_valid_dic)
            step_dic.update(self.result_test_dic)
            #save&log
            wandb.log(step_dic)
            self.adaaug.save_history(self.class2label)
            figure = self.adaaug.plot_history()
            
        call_back_dic = {'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc}
        return call_back_dic

    def _save(self, checkpoint_dir):
        print(checkpoint_dir)
        path = os.path.join(checkpoint_dir, 'gf_check_weights.pt')
        utils.save_model(self.best_gf, path)
        print(path)
        return path

    def _restore(self, checkpoint_path):
        utils.load_model(self.gf_model, f'{checkpoint_path}/gf_check_weights.pt', location=0) #0 as default

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
    logging.info("args = %s", args)
    #wandb
    experiment_name = f'{Aug_type}{description}lamda{args.lambda_aug}_search{args.augselect}_vselect_{args.dataset}{args.labelgroup}_{args.model_name}_e{args.epochs}_lr{args.learning_rate}'
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
    hparams['BASE_PATH'] = args.base_path
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
