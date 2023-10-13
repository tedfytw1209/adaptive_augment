from operation_tseries import *
from datasets import EDFX,PTBXL,Chapman,WISDM

if __name__ == '__main__':
    print('Test all operations')
    
    #t = np.linspace(0, 10, 1000)
    #x = np.vstack([np.cos(t),np.sin(t),np.random.normal(0, 0.3, 1000)]).T
    Freq_dict = {
    'edfx' : 100,
    'ptbxl' : 100,
    'wisdm' : 20,
    'chapman' : 500,
    }
    TimeS_dict = {
    'edfx' : 30,
    'ptbxl' : 10,
    'wisdm' : 10,
    'chapman' : 10,
    }
    #get sample from dataset
    '''folds = [i for i in range(1,11)]
    fold_9_list = [folds[:8],[folds[8]],[folds[9]]]
    print(fold_9_list)
    dataset = PTBXL(dataset_path='../Dataset/ptbxl-dataset',mode=fold_9_list[0],labelgroup='form',multilabel=False)
    print(dataset[0])
    print(dataset[0][0].shape)
    sample = dataset[0]
    x = sample[0]
    label = sample[2]
    '''
    #get sample from .npy
    #fpath = "../intro_case/img5_14_data.npy"
    fpath = "../data_img40_17_identity.npy"
    data = np.load(fpath, mmap_mode=None, allow_pickle=True)
    x = data
    label = 14
    print(x.shape)
    print(x)
    print(x.mean(0))
    t = np.linspace(0, TimeS_dict['ptbxl'], 1000)
    print(t.shape)
    print(x.shape)
    test_ops = ['Scaling']
    '''rng = check_random_state(None)
    rd_start = rng.uniform(0, 2*np.pi, size=(1, 1))
    rd_hz = 1
    tot_s = 10
    rd_T = tot_s / rd_hz
    factor = np.linspace(rd_start,rd_start + (2*np.pi * rd_T),1000,axis=-1).reshape(1000,1) #(bs,len) ?
    print(factor.shape)
    sin_wave = 2 * np.sin(factor)
    plot_line(t,sin_wave)'''
    #4.8~5.8s, lead
    Select_ch = 1 #lead 2
    sel_start = int(0.8 * 100)
    sel_end = int(1.8 * 100)
    #sel_start = int(4.8 * 100)
    #sel_end = int(5.8 * 100)
    plot_line(t,x,title='img5_14_identity',ch=Select_ch,start=sel_start,end=sel_end,save_path='../intro_case/tmp')
    for name in test_ops:
        for m in [0.2,0.3,0.4]:
            for i in range(20):
                x_tensor = torch.from_numpy(x).float().clone()
                trans_aug = TransfromAugment([name],m=m,n=1,p=1,aug_dict=ALL_DICT)
                x_aug = trans_aug(x_tensor).numpy()
                #tmp change
                factor = np.ones(x_aug.shape)
                factor[120:160,:] = 0.4
                x_aug = np.multiply(x_aug, factor)
                print(x_aug.mean(0))
                print(x_aug.shape)
                plot_line(t,x_aug,f'{name}_m{m}_t{i}',ch=Select_ch,start=sel_start,end=sel_end,save_path='../intro_case/tmp')
    #beat aug
    '''plot_line(t,x,title='identity')
    for each_mode in ['b','p','t']:
        for name in test_ops:
            for m in [0,0.1,0.5,0.98]:
                print(each_mode,'='*10,name,'='*10,m)
                info_aug = BeatAugment([name],m=m,p=1.0,mode=each_mode)
                x_aug = info_aug(x_tensor).numpy()
                print(x_aug.shape)
                plot_line(t,x_aug,f'{name}_mode:{each_mode}_m:{m}')'''
    #keep aug
    '''plot_line(t,x,title='identity')
    x_tensor = torch.unsqueeze(x,dim=0)
    for each_mode in ['b','p','t']:
        for name in test_ops:
            for m in [0.5,0.98]:
                print(each_mode,'='*10,name,'='*10,m)
                info_aug = KeepAugment(transfrom=TransfromAugment([name],m=m,p=1.0),mode=each_mode,length=100,default_select='paste')
                print(x_tensor.shape)
                x_aug = info_aug(x_tensor)
                print(x_tensor.shape)
                x_aug = torch.squeeze(x_aug,dim=0).numpy()
                print(x_aug.shape)
                plot_line(t,x_aug,f'{name}_mode:{each_mode}_m:{m}')'''
    #randaug
    '''randaug = RandAugment(1,0,rd_seed=42)
    name = 'random_time_mask'
    for i in range(3):
        #print('='*10,name,'='*10)
        x_aug = randaug(x_tensor).numpy()
        print(x_aug.shape)
        plot_line(t,x_aug)'''
    #ECG part
    '''print('ECG Augmentation')
    for name in ECG_OPS_NAMES:
        print('='*10,name,'='*10)
        x_aug = apply_augment(x_tensor,name,0.5).numpy()
        print(x_aug.shape)
        plot_line(t,x_aug)'''
    '''name = 'QRS_resample'
    for i in range(3):
        print('='*10,name,'='*10)
        x_aug = apply_augment(x_tensor,name,0.5,rd_seed=None).numpy()
        print(x_aug.shape)
        plot_line(t,x_aug)'''