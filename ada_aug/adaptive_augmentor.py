import random
from unicodedata import name
from cv2 import magnitude
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from operation_tseries import apply_augment,TS_ADD_NAMES,ECG_OPS_NAMES,KeepAugment,AdaKeepAugment
import operation
from networks import get_model
from utils import PolicyHistory,PolicyHistoryKeep
from config import OPS_NAMES,TS_OPS_NAMES

default_config = {'sampling': 'prob',
                    'k_ops': 1,
                    'delta': 0,
                    'temp': 1.0,
                    'search_d': 32,
                    'target_d': 32}
defaultkeep_config = {'keep_aug':False,'mode':'auto','thres':0.6,'length':100}

def perturb_param(param, delta):
    if delta <= 0:
        return param
        
    amt = random.uniform(0, delta)
    if random.random() < 0.5:
        return torch.tensor(max(0, param-amt))
    else:
        return torch.tensor(min(1, param+amt))

def cuc_meanstd(values,idxs):
    mean_v = values[idxs].mean(0).detach().cpu().tolist()
    std_v = values[idxs].std(0).detach().cpu().tolist()
    return mean_v, std_v

def stop_gradient(trans_image, magnitude):
    images = trans_image
    adds = 0

    images = images - magnitude
    adds = adds + magnitude
    images = images.detach() + adds
    return images
def stop_gradient_keep(trans_image, magnitude, keep_thre):
    images = trans_image
    adds = 0

    images = images - magnitude - keep_thre
    adds = adds + magnitude + keep_thre
    images = images.detach() + adds
    return images

def Normal_augment(t_series, model=None,selective='paste', apply_func=None, seq_len=None, **kwargs):
    trans_t_series=[]
    for i, (t_s,each_seq_len) in enumerate(zip(t_series,seq_len)):
        t_s = t_s.detach().cpu()
        trans_t_s = apply_func(t_s,i=i,seq_len=each_seq_len,**kwargs)
        trans_t_series.append(trans_t_s)
    aug_t_s = torch.stack(trans_t_series, dim=0)
    return aug_t_s
def Normal_search(t_series, model=None,selective='paste', apply_func=None,ops_names=None, seq_len=None, **kwargs):
    trans_t_series=[]
    for i, (t_s,each_seq_len) in enumerate(zip(t_series,seq_len)):
        t_s = t_s.detach().cpu()
        #e_len = seq_len[i]
        # Prepare transformed image for mixing
        for k, ops_name in enumerate(ops_names):
            trans_t_s = apply_func(t_s,i=i,k=k,ops_name=ops_name,seq_len=each_seq_len,**kwargs)
            trans_t_series.append(trans_t_s)
            #trans_seqlen_list.append(e_len)
    return torch.stack(trans_t_series, dim=0) #, torch.stack(trans_seqlen_list, dim=0) #(b*k_ops, seq, ch)

class AdaAug(nn.Module):
    def __init__(self, after_transforms, n_class, gf_model, h_model, save_dir=None, 
                    config=default_config):
        super(AdaAug, self).__init__()
        self.ops_names = OPS_NAMES
        self.n_ops = len(self.ops_names)
        self.after_transforms = after_transforms
        self.save_dir = save_dir
        self.gf_model = gf_model
        self.h_model = h_model
        self.n_class = n_class
        self.resize = config['search_d'] != config['target_d']
        self.search_d = config['search_d']
        self.k_ops = config['k_ops']
        self.sampling = config['sampling']
        self.temp = config['temp']
        self.delta = config['delta']
        self.history = PolicyHistory(self.ops_names, self.save_dir, self.n_class)

    def save_history(self, class2label=None):
        self.history.save(class2label)

    def plot_history(self):
        return self.history.plot()
    
    def predict_aug_params(self, images, mode):
        self.gf_model.eval()
        if mode == 'exploit':
            self.h_model.eval()
            T = self.temp
        elif mode == 'explore':
            self.h_model.train()
            T = 1.0
        a_params = self.h_model(self.gf_model.f(images.cuda()))
        magnitudes, weights = torch.split(a_params, self.n_ops, dim=1)
        magnitudes = torch.sigmoid(magnitudes)
        weights = torch.nn.functional.softmax(weights/T, dim=-1)
        return magnitudes, weights

    def add_history(self, images, targets):
        magnitudes, weights = self.predict_aug_params(images, 'exploit')
        for k in range(self.n_class):
            idxs = (targets == k).nonzero().squeeze()
            mean_lambda = magnitudes[idxs].mean(0).detach().cpu().tolist()
            mean_p = weights[idxs].mean(0).detach().cpu().tolist()
            std_lambda = magnitudes[idxs].std(0).detach().cpu().tolist()
            std_p = weights[idxs].std(0).detach().cpu().tolist()
            self.history.add(k, mean_lambda, mean_p, std_lambda, std_p)

    def get_aug_valid_imgs(self, images, magnitudes):
        """Return the mixed latent feature

        Args:
            images ([Tensor]): [description]
            magnitudes ([Tensor]): [description]
        Returns:
            [Tensor]: a set of augmented validation images
        """
        trans_image_list = []
        for i, image in enumerate(images):
            pil_img = transforms.ToPILImage()(image)
            # Prepare transformed image for mixing
            for k, ops_name in enumerate(self.ops_names):
                trans_image = operation.apply_augment(pil_img, ops_name, magnitudes[i][k])
                trans_image = self.after_transforms(trans_image)
                trans_image = stop_gradient(trans_image.cuda(), magnitudes[i][k])
                trans_image_list.append(trans_image)
        return torch.stack(trans_image_list, dim=0)

    def explore(self, images):
        """Return the mixed latent feature

        Args:
            images ([Tensor]): [description]
        Returns:
            [Tensor]: return a batch of mixed features
        """
        magnitudes, weights = self.predict_aug_params(images, 'explore')
        a_imgs = self.get_aug_valid_imgs(images, magnitudes)
        a_features = self.gf_model.f(a_imgs)
        ba_features = a_features.reshape(len(images), self.n_ops, -1)
        
        mixed_features = [w.matmul(feat) for w, feat in zip(weights, ba_features)]
        mixed_features = torch.stack(mixed_features, dim=0)
        return mixed_features

    def get_training_aug_images(self, images, magnitudes, weights):
        # visualization
        if self.k_ops > 0:
            trans_images = []
            if self.sampling == 'prob':
                idx_matrix = torch.multinomial(weights, self.k_ops)
            elif self.sampling == 'max':
                idx_matrix = torch.topk(weights, self.k_ops, dim=1)[1]

            for i, image in enumerate(images):
                pil_image = transforms.ToPILImage()(image)
                for idx in idx_matrix[i]:
                    m_pi = perturb_param(magnitudes[i][idx], self.delta)
                    pil_image = operation.apply_augment(pil_image, self.ops_names[idx], m_pi)
                trans_images.append(self.after_transforms(pil_image))
        else:
            trans_images = []
            for i, image in enumerate(images):
                pil_image = transforms.ToPILImage()(image)
                trans_image = self.after_transforms(pil_image)
                trans_images.append(trans_image)
        
        aug_imgs = torch.stack(trans_images, dim=0).cuda()
        return aug_imgs

    def exploit(self, images):
        resize_imgs = F.interpolate(images, size=self.search_d) if self.resize else images
        magnitudes, weights = self.predict_aug_params(resize_imgs, 'exploit')
        aug_imgs = self.get_training_aug_images(images, magnitudes, weights)
        return aug_imgs

    def forward(self, images, mode):
        if mode == 'explore':
            #  return a set of mixed augmented features
            return self.explore(images)
        elif mode == 'exploit':
            #  return a set of augmented images
            return self.exploit(images)
        elif mode == 'inference':
            return images

class AdaAug_TS(AdaAug):
    def __init__(self, after_transforms, n_class, gf_model, h_model, save_dir=None, visualize=False,
                    config=default_config,keepaug_config=default_config, multilabel=False, augselect='',class_adaptive=False,
                    noaug_add=False,transfrom_dic={}):
        super(AdaAug_TS, self).__init__(after_transforms, n_class, gf_model, h_model, save_dir, config)
        #other already define in AdaAug
        self.ops_names = TS_OPS_NAMES.copy()
        if 'tsadd' in augselect:
            self.ops_names = self.ops_names + TS_ADD_NAMES.copy()
        if 'ecg' in augselect:
            self.ops_names = self.ops_names + ECG_OPS_NAMES.copy()
        print('AdaAug Using ',self.ops_names)
        self.n_ops = len(self.ops_names)
        self.transfrom_dic = transfrom_dic
        self.history = PolicyHistory(self.ops_names, self.save_dir, self.n_class)
        self.config = config
        self.use_keepaug = keepaug_config['keep_aug']
        if self.use_keepaug:
            self.Augment_wrapper = KeepAugment(save_dir=save_dir,**keepaug_config)
            self.Search_wrapper = self.Augment_wrapper.Augment_search
        else:
            self.Augment_wrapper = Normal_augment
            self.Search_wrapper = Normal_search
        self.multilabel = multilabel
        self.class_adaptive = class_adaptive
        self.visualize = visualize
        self.noaug_add = noaug_add
        self.alpha = torch.tensor([0.5]).view(1,-1).cuda()
        self.noaug_max = 0.5
        self.noaug_tensor = self.noaug_max * F.one_hot(torch.tensor([0]), num_classes=self.n_ops).float()

    def predict_aug_params(self, X, seq_len, mode,y=None):
        self.gf_model.eval()
        if mode == 'exploit':
            self.h_model.eval()
            T = self.temp
        elif mode == 'explore':
            if hasattr(self.gf_model, 'lstm'):
                self.gf_model.lstm.train() #!!!maybe for bn is better
            self.h_model.train()
            T = 1.0
        a_params = self.h_model(self.gf_model.extract_features(X.cuda(),seq_len),y=y)
        magnitudes, weights = torch.split(a_params, self.n_ops, dim=1)
        magnitudes = torch.sigmoid(magnitudes)
        weights = torch.nn.functional.softmax(weights/T, dim=-1)
        if self.noaug_add: #add noaug reweights
            if self.class_adaptive: #alpha: (1,n_class), y: (batch_szie,n_class)=>(batch_size,1)
                batch_alpha = torch.sum(self.alpha * y,dim=-1,keepdim=True) / torch.sum(y,dim=-1,keepdim=True)
            else:
                batch_alpha = self.alpha.view(-1)
            weights = batch_alpha * weights + (1.0-batch_alpha) * (self.noaug_tensor.cuda() + weights/2)
        return magnitudes, weights

    def add_history(self, images, seq_len, targets,y=None):
        magnitudes, weights = self.predict_aug_params(images, seq_len, 'exploit',y=y)
        for k in range(self.n_class):
            if self.multilabel:
                idxs = (targets[:,k] == 1).nonzero().squeeze()
            else:
                idxs = (targets == k).nonzero().squeeze()
            mean_lambda = magnitudes[idxs].mean(0).detach().cpu().tolist()
            mean_p = weights[idxs].mean(0).detach().cpu().tolist()
            std_lambda = magnitudes[idxs].std(0).detach().cpu().tolist()
            std_p = weights[idxs].std(0).detach().cpu().tolist()
            self.history.add(k, mean_lambda, mean_p, std_lambda, std_p)

    def get_aug_valid_img(self, image, magnitudes,i=None,k=None,ops_name=None, seq_len=None):
        trans_image = apply_augment(image, ops_name, magnitudes[i][k].detach().cpu().numpy(),seq_len=seq_len,**self.transfrom_dic)
        trans_image = self.after_transforms(trans_image)
        trans_image = stop_gradient(trans_image.cuda(), magnitudes[i][k])
        return trans_image
    def get_aug_valid_imgs(self, images, magnitudes, seq_len=None):
        """Return the mixed latent feature

        Args:
            images ([Tensor]): [description]
            magnitudes ([Tensor]): [description]
        Returns:
            [Tensor]: a set of augmented validation images
        """
        #trans_seqlen_list = []
        '''trans_image_list = []
        for i, image in enumerate(images):
            pil_img = image.detach().cpu()
            #e_len = seq_len[i]
            # Prepare transformed image for mixing
            for k, ops_name in enumerate(self.ops_names):
                trans_image = apply_augment(pil_img, ops_name, magnitudes[i][k].detach().cpu().numpy())
                trans_image = self.after_transforms(trans_image)
                trans_image = stop_gradient(trans_image.cuda(), magnitudes[i][k])
                trans_image_list.append(trans_image)
                #trans_seqlen_list.append(e_len)'''
        aug_imgs = self.Search_wrapper(images, model=self.gf_model,apply_func=self.get_aug_valid_img,magnitudes=magnitudes,ops_names=self.ops_names,selective='paste',seq_len=seq_len)
        #return torch.stack(trans_image_list, dim=0) #, torch.stack(trans_seqlen_list, dim=0) #(b*k_ops, seq, ch)
        return aug_imgs

    def explore(self, images, seq_len, mix_feature=True,y=None):
        """Return the mixed latent feature if mix_feature==True
        !!!can't use for dynamic len now
        Args:
            images ([Tensor]): [description]
        Returns:
            [Tensor]: return a batch of mixed features
        """
        magnitudes, weights = self.predict_aug_params(images, seq_len,'explore',y=y)
        a_imgs = self.get_aug_valid_imgs(images, magnitudes,seq_len=seq_len)
        #a_imgs = self.Augment_wrapper(images, model=self.gf_model,apply_func=self.get_aug_valid_imgs,magnitudes=magnitudes,selective='paste')
        #a_features = self.gf_model.extract_features(a_imgs, a_seq_len)
        a_features = self.gf_model.extract_features(a_imgs)
        ba_features = a_features.reshape(len(images), self.n_ops, -1) # batch, n_ops, n_hidden
        if mix_feature:
            mixed_features = [w.matmul(feat) for w, feat in zip(weights, ba_features)]
            mixed_features = torch.stack(mixed_features, dim=0)
            return mixed_features, [weights]
        else:
            return ba_features, [weights]
    def get_training_aug_image(self, image, magnitudes, idx_matrix,i=None, seq_len=None):
        if i!=None:
            idx_list = idx_matrix[i]
            magnitude_i = magnitudes[i]
        else:
            idx_list,magnitude_i = idx_matrix,magnitudes
        for idx in idx_list:
            m_pi = perturb_param(magnitude_i[idx], self.delta).detach().cpu().numpy()
            image = apply_augment(image, self.ops_names[idx], m_pi,seq_len=seq_len,**self.transfrom_dic)
        return self.after_transforms(image)
    def get_training_aug_images(self, images, magnitudes, weights, seq_len=None):
        # visualization
        if self.k_ops > 0:
            trans_images = []
            if self.sampling == 'prob':
                idx_matrix = torch.multinomial(weights, self.k_ops)
            elif self.sampling == 'max':
                idx_matrix = torch.topk(weights, self.k_ops, dim=1)[1] #where op index the highest weight

            '''for i, image in enumerate(images):
                pil_image = image.detach().cpu()
                for idx in idx_matrix[i]:
                    m_pi = perturb_param(magnitudes[i][idx], self.delta).detach().cpu().numpy()
                    pil_image = apply_augment(pil_image, self.ops_names[idx], m_pi)

                trans_images.append(self.after_transforms(pil_image))'''
            aug_imgs = self.Augment_wrapper(images, model=self.gf_model,apply_func=self.get_training_aug_image,
                    magnitudes=magnitudes,idx_matrix=idx_matrix,selective='paste',seq_len=seq_len)
        else:
            trans_images = []
            for i, image in enumerate(images):
                pil_image = image.detach().cpu()
                trans_image = self.after_transforms(pil_image)
                trans_images.append(trans_image)
            aug_imgs = torch.stack(trans_images, dim=0).cuda()
        
        #aug_imgs = torch.stack(trans_images, dim=0).cuda()
        return aug_imgs.cuda() #(b, seq, ch)

    def exploit(self, images, seq_len,y=None):
        if self.resize and 'lstm' not in self.config['gf_model_name']:
            resize_imgs = F.interpolate(images, size=self.search_d)
        else:
            resize_imgs = images
        #resize_imgs = F.interpolate(images, size=self.search_d) if self.resize else images
        magnitudes, weights = self.predict_aug_params(resize_imgs, seq_len, 'exploit',y=y)
        aug_imgs = self.get_training_aug_images(images, magnitudes, weights,seq_len=seq_len)
        if self.visualize:
            print('Visualize for Debug')
            self.print_imgs(imgs=images,title='id')
            self.print_imgs(imgs=aug_imgs,title='aug')
            exit()
        return aug_imgs

    def visualize_result(self, images, seq_len,y=None):
        if self.resize and 'lstm' not in self.config['gf_model_name']:
            resize_imgs = F.interpolate(images, size=self.search_d)
        else:
            resize_imgs = images
        #resize_imgs = F.interpolate(images, size=self.search_d) if self.resize else images
        magnitudes, weights = self.predict_aug_params(resize_imgs, seq_len, 'exploit',y=y)
        aug_imgs = self.get_training_aug_images(images, magnitudes, weights,seq_len=seq_len)
        if self.use_keepaug:
            slc_out = self.Augment_wrapper.visualize_slc(images, model=self.gf_model)
        print('Visualize for Debug')
        self.print_imgs(imgs=images,title='id')
        self.print_imgs(imgs=aug_imgs,title='aug')
        

    def forward(self, images, seq_len, mode, mix_feature=True,y=None):
        if mode == 'explore':
            #  return a set of mixed augmented features, mix_feature is for experiment
            return self.explore(images,seq_len,mix_feature=mix_feature,y=y)
        elif mode == 'exploit':
            #  return a set of augmented images
            return self.exploit(images,seq_len,y=y)
        elif mode == 'inference':
            return images
    
    def print_imgs(self,imgs,title):
        imgs = imgs.cpu().detach().numpy()
        t = np.linspace(0, 10, 1000)
        for idx,img in enumerate(imgs):
            plt.clf()
            channel_num = img.shape[-1]
            for i in  range(channel_num):
                plt.plot(t, img[:,i])
            if title:
                plt.title(title)
            plt.savefig(f'{self.save_dir}/img{idx}_{title}.png')
    
    def update_alpha(self,class_acc):
        self.alpha = torch.tensor(class_acc).view(1,-1).cuda()

class AdaAugkeep_TS(AdaAug):
    def __init__(self, after_transforms, n_class, gf_model, h_model, save_dir=None, visualize=False,
                    config=default_config,keepaug_config=default_config, multilabel=False, augselect='',class_adaptive=False,ind_mix=False,
                    noaug_add=False,transfrom_dic={}):
        super(AdaAugkeep_TS, self).__init__(after_transforms, n_class, gf_model, h_model, save_dir, config)
        #other already define in AdaAug
        self.ops_names = TS_OPS_NAMES.copy()
        if 'tsadd' in augselect:
            self.ops_names = self.ops_names + TS_ADD_NAMES.copy()
        if 'ecg' in augselect:
            self.ops_names = self.ops_names + ECG_OPS_NAMES.copy()
        self.possible_segment = keepaug_config.get('possible_segment',[1])
        self.keep_lens = keepaug_config['length']
        if keepaug_config['adapt_target'] == 'len': # adapt len
            self.adapt_len = len(self.keep_lens)
            self.adapt_params = self.keep_lens
        elif keepaug_config['adapt_target'] == 'seg': #adapt segment
            self.adapt_len = len(self.possible_segment)
            self.adapt_params = self.possible_segment
        elif keepaug_config['adapt_target'] == 'way': #adapt segment
            self.adapt_len = 4 #!!! const now
            self.adapt_params = [('cut',False),('cut',True),('paste',False),('paste',True)] #(selective,reverse)
        else:
            print('KeepAdapt need multiple lens or segment to learn')
            exit()
        self.thres_adapt = keepaug_config.get('thres_adapt',True)
        self.ind_mix = ind_mix
        print('AdaAug Using ',self.ops_names)
        print('Adapt target ',self.adapt_params)
        print('KeepAug params using ',self.adapt_params)
        print('KeepAug lens using ',self.keep_lens)
        print('KeepAug segments using ',self.possible_segment)
        print('Keep thres adapt: ',self.thres_adapt)
        self.n_ops = len(self.ops_names)
        self.transfrom_dic = transfrom_dic
        self.n_keeplens = len(self.keep_lens)
        self.history = PolicyHistoryKeep(self.ops_names,self.adapt_params, self.save_dir, self.n_class)
        self.config = config
        self.use_keepaug = keepaug_config['keep_aug']
        self.adapt_target = keepaug_config['adapt_target']
        self.keepaug_config = keepaug_config
        if self.use_keepaug:
            self.Augment_wrapper = AdaKeepAugment(save_dir=save_dir,**keepaug_config)
            self.Search_wrapper = self.Augment_wrapper.Augment_search
            if ind_mix:
                self.Search_wrapper = self.Augment_wrapper.Augment_search_ind
        else:
            print('AdaAug Keep always use keep augment')
            exit()
        self.multilabel = multilabel
        self.class_adaptive = class_adaptive
        self.visualize = visualize
        self.noaug_add = noaug_add
        self.alpha = 0.5
        self.noaug_max = 0.5
        self.noaug_tensor = torch.zeros((1,self.n_ops)).float()
        self.noaug_tensor[0:0] = self.noaug_max #noaug

    def predict_aug_params(self, X, seq_len, mode,y=None):
        self.gf_model.eval()
        if mode == 'exploit':
            self.h_model.eval()
            T = self.temp
        elif mode == 'explore':
            if hasattr(self.gf_model, 'lstm'):
                self.gf_model.lstm.train() #!!!maybe for bn is better
            self.h_model.train()
            T = 1.0
        a_params = self.h_model(self.gf_model.extract_features(X.cuda(),seq_len),y=y)
        #mags: mag for ops, weights: weight for ops, keeplen_ws: keeplen weights, keep_thres: keep threshold
        magnitudes, weights, keeplen_ws, keep_thres = torch.split(a_params, [self.n_ops, self.n_ops, self.adapt_len, 1], dim=1)
        magnitudes = torch.sigmoid(magnitudes)
        weights = torch.nn.functional.softmax(weights/T, dim=-1)
        keeplen_ws = torch.nn.functional.softmax(keeplen_ws/T, dim=-1)
        if self.thres_adapt:
            keep_thres = torch.sigmoid(keep_thres)
        else: #fix thres break gradient
            keep_thres = torch.full(keep_thres.shape,self.keepaug_config['thres'],device=keep_thres.device)
        if self.noaug_add: #add noaug reweights
            weights = self.alpha * weights + (1-self.alpha) * self.noaug_tensor
        return magnitudes, weights, keeplen_ws, keep_thres

    def add_history(self, images, seq_len, targets,y=None):
        #need change
        magnitudes, weights, keeplen_ws, keep_thres = self.predict_aug_params(images, seq_len, 'exploit',y=y)
        for k in range(self.n_class):
            if self.multilabel:
                idxs = (targets[:,k] == 1).nonzero().squeeze()
            else:
                idxs = (targets == k).nonzero().squeeze()
            mean_lambda,std_lambda = cuc_meanstd(magnitudes,idxs)
            mean_p,std_p = cuc_meanstd(weights,idxs)
            mean_len, std_len = cuc_meanstd(keeplen_ws,idxs)
            mean_thre, std_thre = cuc_meanstd(keep_thres,idxs)
            self.history.add(k, mean_lambda, mean_p, mean_len, mean_thre, std_lambda, std_p, std_len, std_thre)

    def get_aug_valid_img(self, image, magnitudes,keep_thres,i=None,k=None,ops_name=None, seq_len=None):
        trans_image = apply_augment(image, ops_name, magnitudes[i][k].detach().cpu().numpy(),seq_len=seq_len,**self.transfrom_dic)
        trans_image = self.after_transforms(trans_image)
        #trans_image = stop_gradient_keep(trans_image.cuda(), magnitudes[i][k], keep_thres[i]) #add keep thres
        return trans_image
    def get_aug_valid_imgs(self, images, magnitudes,weights, keeplen_ws, keep_thres, seq_len=None):
        """Return the mixed latent feature
        Args:
            images ([Tensor]): [description]
            magnitudes ([Tensor]): [description]
            keep_thres ([Tensor]): [description]
        Returns:
            [Tensor]: a set of augmented validation images
        """
        #get fixed value, if search ops, fix idx is keeplen_ws after select. if search lens, fix idx is keeplen_ws after select.
        if self.ind_mix:
            stage_name = self.Augment_wrapper.all_stages[self.Augment_wrapper.stage]
            idx_matrix,len_idx = self.select_params(weights,keeplen_ws)
            if stage_name=='trans':
                fix_idx = len_idx
            elif stage_name=='keep':
                fix_idx = idx_matrix
            aug_imgs = self.Search_wrapper(images, model=self.gf_model,apply_func=self.get_aug_valid_img, keep_thres=keep_thres,
                magnitudes=magnitudes,ops_names=self.ops_names,fix_idx=fix_idx,selective='paste',seq_len=seq_len)
        else:
            aug_imgs = self.Search_wrapper(images, model=self.gf_model,apply_func=self.get_aug_valid_img, keep_thres=keep_thres,
                magnitudes=magnitudes,ops_names=self.ops_names,selective='paste',seq_len=seq_len)
        #(b*lens*ops,seq,ch) or 
        return aug_imgs
    def explore(self, images, seq_len, mix_feature=True,y=None):
        """Return the mixed latent feature !!!can't use for dynamic len now
        Args:
            images ([Tensor]): [description]
        Returns:
            [Tensor]: return a batch of mixed features
        """
        magnitudes, weights, keeplen_ws, keep_thres = self.predict_aug_params(images, seq_len,'explore',y=y)
        a_imgs = self.get_aug_valid_imgs(images, magnitudes,weights, keeplen_ws, keep_thres,seq_len=seq_len) #(b*lens*ops,seq,ch)
        a_features = self.gf_model.extract_features(a_imgs) #(b*keep_len*n_ops, n_hidden)
        stage_name = self.Augment_wrapper.all_stages[self.Augment_wrapper.stage] #
        if self.ind_mix:
            if stage_name=='trans':
                ba_features = a_features.reshape(len(images), self.n_ops, -1)# batch, n_ops, n_hidden
                out_w = weights
            elif stage_name=='keep':
                ba_features = a_features.reshape(len(images), self.adapt_len, -1)# batch, keep_lens, n_hidden
                out_w = keeplen_ws
            self.Augment_wrapper.change_stage() #finish
            if mix_feature:
                mixed_features = [w.matmul(feat) for w, feat in zip(out_w, ba_features)]
                mixed_features = torch.stack(mixed_features, dim=0)
                return mixed_features, [out_w]
            else:
                return ba_features, [out_w]
        else:
            ba_features = a_features.reshape(len(images), self.n_ops, self.adapt_len, -1).permute(0,2,1,3) # batch, n_ops,keep_lens, n_hidden
            if mix_feature:
                mixed_features = [w.matmul(feat) for w, feat in zip(weights, ba_features)] #[(keep_lens, n_hidden)]
                mixed_features = [len_w.matmul(feat) for len_w,feat in zip(keeplen_ws,mixed_features)] #[(n_hidden)]
                mixed_features = torch.stack(mixed_features, dim=0)
                return mixed_features , [weights, keeplen_ws]
            else:
                return ba_features, [weights, keeplen_ws]
    def select_params(self,weights,keeplen_ws):
        if self.sampling == 'prob':
            idx_matrix = torch.multinomial(weights, self.k_ops)
            len_idx = torch.multinomial(keeplen_ws, 1).view(-1).detach().cpu().numpy()
        elif self.sampling == 'max':
            idx_matrix = torch.topk(weights, self.k_ops, dim=1)[1] #where op index the highest weight
            len_idx = torch.topk(keeplen_ws, 1, dim=1)[1].view(-1).detach().cpu().numpy()
        return idx_matrix,len_idx
    def get_training_aug_image(self, image, magnitudes, idx_matrix,i=None, seq_len=None):
        if i!=None:
            idx_list = idx_matrix[i]
            magnitude_i = magnitudes[i]
        else:
            idx_list,magnitude_i = idx_matrix,magnitudes
        for idx in idx_list:
            m_pi = perturb_param(magnitude_i[idx], self.delta).detach().cpu().numpy()
            image = apply_augment(image, self.ops_names[idx], m_pi,seq_len=seq_len,**self.transfrom_dic)
        return self.after_transforms(image)
    def get_training_aug_images(self, images, magnitudes, weights, keeplen_ws, keep_thres, seq_len=None):
        # visualization
        if self.k_ops > 0:
            trans_images = []
            idx_matrix,len_idx = self.select_params(weights,keeplen_ws)
            keep_thres = keep_thres.view(-1).detach().cpu().numpy()
            aug_imgs = self.Augment_wrapper(images, model=self.gf_model,apply_func=self.get_training_aug_image,magnitudes=magnitudes,
            idx_matrix=idx_matrix,len_idx=len_idx,keep_thres=keep_thres,selective='paste',seq_len=seq_len)
        else:
            trans_images = []
            for i, image in enumerate(images):
                pil_image = image.detach().cpu()
                trans_image = self.after_transforms(pil_image)
                trans_images.append(trans_image)
            aug_imgs = torch.stack(trans_images, dim=0).cuda()
        
        #aug_imgs = torch.stack(trans_images, dim=0).cuda()
        return aug_imgs.cuda() #(b, seq, ch)
    def exploit(self, images, seq_len,y=None):
        if self.resize and 'lstm' not in self.config['gf_model_name']:
            resize_imgs = F.interpolate(images, size=self.search_d)
        else:
            resize_imgs = images
        #resize_imgs = F.interpolate(images, size=self.search_d) if self.resize else images
        magnitudes, weights, keeplen_ws, keep_thres = self.predict_aug_params(resize_imgs, seq_len, 'exploit',y=y)
        #print(magnitudes.shape, weights.shape, keeplen_ws.shape, keep_thres.shape)
        aug_imgs = self.get_training_aug_images(images, magnitudes, weights, keeplen_ws, keep_thres,seq_len=seq_len)
        if self.visualize:
            print('Visualize for Debug')
            self.print_imgs(imgs=images,title='id')
            self.print_imgs(imgs=aug_imgs,title='aug')
            exit()
        return aug_imgs
    def visualize_result(self, images, seq_len,y=None):
        if self.resize and 'lstm' not in self.config['gf_model_name']:
            resize_imgs = F.interpolate(images, size=self.search_d)
        else:
            resize_imgs = images
        #resize_imgs = F.interpolate(images, size=self.search_d) if self.resize else images
        magnitudes, weights, keeplen_ws, keep_thres = self.predict_aug_params(resize_imgs, seq_len, 'exploit',y=y)
        #print(magnitudes.shape, weights.shape, keeplen_ws.shape, keep_thres.shape)
        aug_imgs = self.get_training_aug_images(images, magnitudes, weights, keeplen_ws, keep_thres,seq_len=seq_len)
        print('Visualize for Debug')
        self.print_imgs(imgs=images,title='id')
        self.print_imgs(imgs=aug_imgs,title='aug')
        if self.use_keepaug:
            self.Augment_wrapper.visualize_slc(images, model=self.gf_model)

    def forward(self, images, seq_len, mode, mix_feature=True,y=None):
        if mode == 'explore':
            #  return a set of mixed augmented features, mix_feature is for experiment
            return self.explore(images,seq_len,mix_feature=mix_feature,y=y)
        elif mode == 'exploit':
            #  return a set of augmented images
            return self.exploit(images,seq_len,y=y)
        elif mode == 'inference':
            return images
    
    def print_imgs(self,imgs,title):
        imgs = imgs.cpu().detach().numpy()
        t = np.linspace(0, 10, 1000)
        for idx,img in enumerate(imgs):
            plt.clf()
            channel_num = img.shape[-1]
            for i in  range(channel_num):
                plt.plot(t, img[:,i])
            if title:
                plt.title(title)
            plt.savefig(f'{self.save_dir}/img{idx}_{title}.png')