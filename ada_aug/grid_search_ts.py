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
from step_function import train,infer,search_train,search_infer
from non_saturating_loss import NonSaturatingLoss
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
parser.add_argument('--search_freq', type=float, nargs='+', default=1, help='exploration frequency')
parser.add_argument('--search_round', type=int, nargs='+', default=1, help='exploration frequency') #search_round
parser.add_argument('--n_proj_layer', type=int, default=0, help="number of hidden layer in augmentation policy projection")
parser.add_argument('--valselect', action='store_true', default=False, help='use valid select')
parser.add_argument('--augselect', type=str, default='', help="augmentation selection")
parser.add_argument('--diff_aug', action='store_true', default=False, help='use valid select')
parser.add_argument('--not_mix', action='store_true', default=False, help='use valid select')
parser.add_argument('--not_reweight', action='store_true', default=False, help='use valid select')
parser.add_argument('--lambda_aug', type=float, nargs='+', default=1.0, help="augment sample weight")
parser.add_argument('--class_adapt', action='store_true', default=False, help='class adaptive')
parser.add_argument('--class_embed', action='store_true', default=False, help='class embed') #tmp use
parser.add_argument('--loss_type', type=str, default='minus', nargs='+', help="loss type for difficult policy training", choices=['minus','relative','adv'])
parser.add_argument('--keep_aug', action='store_true', default=False, help='info keep augment')
parser.add_argument('--keep_mode', type=str, default='auto', help='info keep mode',choices=['auto','b','p','t'])
parser.add_argument('--keep_thres', type=float, nargs='+', default=[0.6], help="augment sample weight")
parser.add_argument('--keep_len', type=int, nargs='+', default=[100], help="info keep seq len")

args = parser.parse_args()
debug = True if args.save == "debug" else False
if args.k_ops>0:
    Aug_type = 'AdaAug'
else:
    Aug_type = 'NOAUG'

description='grid'
if args.keep_aug:
    description+=f'keep{args.keep_mode}'
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
        args = argparse.Namespace(**self.config)
        #  dataset settings for search
        n_channel = get_num_channel(self.config['dataset'])
        n_class = get_num_class(self.config['dataset'],self.config['labelgroup'])
        sdiv = get_search_divider(self.config['model_name'])
        class2label = get_label_name(self.config['dataset'], self.config['dataroot'])
        multilabel = self.config['multilabel']
        diff_augment = self.config['diff_aug']
        diff_mix = not self.config['not_mix']
        diff_reweight = not self.config['not_reweight']
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
        elif self.config['default_split']:
            print('Default Train 1~8 /Valid 9/Test 10 fold split')
        else:
            print('Warning: Random splitting train/test')
        self.train_queue, self.valid_queue, self.search_queue, self.test_queue, self.tr_search_queue = get_ts_dataloaders(
            self.config['dataset'], self.config['batch_size'], self.config['num_workers'],
            self.config['dataroot'], self.config['cutout'], self.config['cutout_length'],
            split=self.config['train_portion'], split_idx=0, target_lb=-1,
            search=True, search_divider=sdiv,search_size=self.config['search_size'],
            test_size=self.config['test_size'],multilabel=self.config['multilabel'],default_split=self.config['default_split'],
            fold_assign=train_val_test_folds,labelgroup=self.config['labelgroup'])
        #  model settings
        self.gf_model = get_model_tseries(model_name=self.config['model_name'], num_class=n_class,n_channel=n_channel,
            use_cuda=True, data_parallel=False,dataset=self.config['dataset'])
        h_input = self.gf_model.fc.in_features
        label_num, label_embed = 0,0
        if self.config['class_adapt'] and self.config['class_embed']:
            label_num = n_class
            label_embed = 32 #tmp use
            h_input = h_input + label_embed
        elif self.config['class_adapt']:
            h_input =h_input + n_class
        
        self.h_model = Projection_TSeries(in_features=h_input,label_num=label_num,label_embed=label_embed,
            n_layers=self.config['n_proj_layer'], n_hidden=128, augselect=self.config['augselect']).cuda()
        #  training settings
        self.gf_optimizer = torch.optim.AdamW(self.gf_model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay']) #follow ptbxl batchmark!!!
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.gf_optimizer, max_lr=self.config['learning_rate'], 
            epochs = self.config['epochs'], steps_per_epoch = len(self.train_queue)) #follow ptbxl batchmark!!!
        self.h_optimizer = torch.optim.Adam(
            self.h_model.parameters(),
            lr=self.config['proj_learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=self.config['proj_weight_decay'])
        
        if not multilabel:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.criterion = self.criterion.cuda()
        self.adv_criterion = None
        if self.config['loss_type']=='adv':
            self.adv_criterion = NonSaturatingLoss(epsilon=0.1)

        #  AdaAug settings for search
        after_transforms = self.train_queue.dataset.after_transforms
        adaaug_config = {'sampling': 'prob',
                    'k_ops': self.config['k_ops'], #as paper
                    'delta': 0.0, 
                    'temp': 1.0, 
                    'search_d': get_dataset_dimension(self.config['dataset']),
                    'target_d': get_dataset_dimension(self.config['dataset']),
                    'gf_model_name': self.config['model_name']}
        keepaug_config = {'keep_aug':self.config['keep_aug'],'mode':self.config['keep_mode'],'thres':self.config['keep_thres'],'length':self.config['keep_len']}
        self.adaaug = AdaAug_TS(after_transforms=after_transforms,
            n_class=n_class,
            gf_model=self.gf_model,
            h_model=self.h_model,
            save_dir=os.path.join(self.config['BASE_PATH'],self.config['save'],f'fold{test_fold_idx}'),
            config=adaaug_config,
            keepaug_config=keepaug_config,
            multilabel=multilabel,
            augselect=self.config['augselect'],
            class_adaptive=self.config['class_adapt'])
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
    def step(self):#use step replace _train
        if self._iteration==0:
            wandb.config.update(self.config)
        print(f'Starting Ray ID {self.trial_id} Iteration: {self._iteration}')
        args = argparse.Namespace(**self.config)
        lr = self.scheduler.get_last_lr()[0]
        step_dic={'epoch':self._iteration}
        diff_dic = {'difficult_aug':self.diff_augment,'reweight':self.diff_reweight,'lambda_aug':self.config['lambda_aug'], 'class_adaptive':self.config['class_adapt'],
                'loss_type':self.config['loss_type'], 'adv_criterion': self.adv_criterion}
        # searching
        train_acc, train_obj, train_dic = search_train(args,self.train_queue, self.search_queue, self.tr_search_queue, self.gf_model, self.adaaug,
            self.criterion, self.gf_optimizer,self.scheduler, self.config['grad_clip'], self.h_optimizer, self._iteration, self.config['search_freq'], 
            search_round=self.config['search_round'],multilabel=self.multilabel,n_class=self.n_class, **diff_dic)

        # validation
        valid_acc, valid_obj,valid_dic = search_infer(self.valid_queue, self.gf_model, self.criterion, multilabel=self.multilabel,n_class=self.n_class,mode='valid')
        #scheduler.step()
        #test
        test_acc, test_obj, test_dic  = search_infer(self.test_queue, self.gf_model, self.criterion, multilabel=self.multilabel,n_class=self.n_class,mode='test')
        #val select
        if self.config['valselect'] and valid_acc>self.best_val_acc:
            self.best_val_acc = valid_acc
            self.result_valid_dic = {f'result_{k}': valid_dic[k] for k in valid_dic.keys()}
            self.result_test_dic = {f'result_{k}': test_dic[k] for k in test_dic.keys()}
            valid_dic['best_valid_acc_avg'] = valid_acc
            test_dic['best_test_acc_avg'] = test_acc
            self.best_gf = self.gf_model
            self.best_h = self.h_model
        elif not self.config['valselect']:
            self.best_gf = self.gf_model
            self.best_h = self.h_model
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
            self.adaaug.save_history(self.class2label)
            figure = self.adaaug.plot_history()
            wandb.log(step_dic)
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
    '''
    hparams = {
        'args':args,
        'dataset':args.dataset, 'batch_size':args.batch_size, 'num_epochs':args.epochs,
        'multilabel': args.multilabel,
        'gradient_clipping_by_global_norm': args.grad_clip,
        'lr': args.learning_rate,
        'wd': args.weight_decay,
        'momentum': args.momentum,
        'temperature': args.temperature,
        'k_ops': args.k_ops,
        'train_portion': args.train_portion, ## for text data controlling ratio of training data
        'default_split': args.default_split,
        'labelgroup': args.labelgroup,
        'augselect': args.augselect,
        'diff_aug': args.diff_aug,
        'not_reweight': args.not_reweight,
        'lambda_aug': args.lambda_aug,
        'class_adapt': args.class_adapt,
        'class_embed': args.class_embed,
        'loss_type': args.loss_type,
        'keep_aug': args.keep_aug,
        'keep_mode': args.keep_mode,
        'keep_thres': args.keep_thres,
        #'kfold': tune.grid_search([i for i in range(args.kfold)]),
        'save': args.save,
        'ray_name': args.ray_name,
        'BASE_PATH': args.base_path,
    }'''
    hparams['args'] = args
    hparams['BASE_PATH'] = args.base_path
    if args.kfold==10:
        hparams['kfold'] = tune.grid_search([i for i in range(args.kfold)])
    else:
        hparams['kfold'] = tune.grid_search([args.kfold]) #for some fold
    #for grid search
    print(hparams)
    hparams['search_freq'] = tune.grid_search(hparams['search_freq'])
    hparams['search_round'] = tune.grid_search(hparams['search_round'])
    hparams['loss_type'] = tune.grid_search(hparams['loss_type'])
    hparams['lambda_aug'] = tune.grid_search(hparams['lambda_aug'])
    hparams['keep_thres'] = tune.grid_search(hparams['keep_thres'])
    hparams['keep_len'] = tune.grid_search(hparams['keep_len'])
    if args.not_reweight:
        hparams['not_reweight'] = tune.grid_search([True,False])
    if args.class_adapt:
        hparams['class_adapt'] = tune.grid_search([True,False])
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
