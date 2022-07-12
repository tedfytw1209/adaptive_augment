import numpy as np
import torch
import torchvision
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit
from torch.utils.data import Sampler, Subset, SubsetRandomSampler
from torchvision import transforms
from datasets import EDFX,PTBXL,Chapman,WISDM
import random


_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
_SVHN_MEAN, _SVHN_STD = (0.43090966, 0.4302428, 0.44634357), (0.19652855, 0.19832038, 0.19942076)


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class SubsetSampler(Sampler):
    """Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class AugmentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, pre_transforms, after_transforms, valid_transforms, search, train):
        super(AugmentDataset, self).__init__()
        self.dataset = dataset
        self.pre_transforms = pre_transforms
        self.after_transforms = after_transforms
        self.valid_transforms = valid_transforms
        self.search = search
        self.train = train

    def __getitem__(self, index):
        if self.search:
            raw_image, target = self.dataset.__getitem__(index)
            raw_image = self.pre_transforms(raw_image)
            image = transforms.ToTensor()(raw_image)
            return image, target
        else:
            img, target = self.dataset.__getitem__(index)
            if self.train:
                img = self.pre_transforms(img)
                img = self.after_transforms(img)
            else:
                if self.valid_transforms is not None:
                    img = self.valid_transforms(img)
            return img, target

    def __len__(self):
        return self.dataset.__len__()

class AugmentDataset_TS(torch.utils.data.Dataset):
    def __init__(self, dataset, pre_transforms, after_transforms, valid_transforms, search, train):
        super(AugmentDataset_TS, self).__init__()
        self.dataset = dataset
        self.pre_transforms = pre_transforms
        self.after_transforms = after_transforms
        self.valid_transforms = valid_transforms
        self.search = search
        self.train = train

    def __getitem__(self, index):
        if self.search:
            raw_image,seq_len, target = self.dataset.__getitem__(index)
            raw_image = self.pre_transforms(raw_image)
            image = transforms.ToTensor()(raw_image)
            return image, seq_len, target
        else:
            img,seq_len, target = self.dataset.__getitem__(index)
            if self.train:
                img = self.pre_transforms(img)
                img = self.after_transforms(img)
            else:
                if self.valid_transforms is not None:
                    img = self.valid_transforms(img)
            return img, seq_len, target

    def __len__(self):
        return self.dataset.__len__()

def get_num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'svhn': 10,
        'reduced_svhn': 10,
        'edfx': 5,
        'ptbxl_diagnostic': 44,
        'ptbxl_subdiagnostic': 23,
        'wisdm': 18,
        'chapman': 12,
    }[dataset]


def get_num_channel(dataset):
    return {
        'cifar10': 3,
        'reduced_cifar10': 3,
        'svhn': 3,
        'reduced_svhn': 3,
        'edfx': 2,
        'ptbxl_diagnostic': 12,
        'ptbxl_subdiagnostic': 12,
        'wisdm': 3,
        'chapman': 12,
    }[dataset]


def get_dataloaders(dataset, batch, num_workers, dataroot, cutout,
                    cutout_length, split=0.5, split_idx=0, target_lb=-1,
                    search=True, search_divider=1, search_size=0):
    '''
    If search is True, dataloader will give batches of image without after_transforms,
    the transform will be done by augment agent
    If search is False, used in benchmark training
    '''
    if 'cifar10' in dataset:
        transform_train_pre = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        transform_train_after = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    elif 'svhn' in dataset:
        transform_train_pre = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
        ])
        transform_train_after = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_SVHN_MEAN, _SVHN_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_SVHN_MEAN, _SVHN_STD),
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    if cutout and cutout_length != 0:
        transform_train_after.transforms.append(CutoutDefault(cutout_length))

    if dataset == 'cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=None)
        search_dataset = None
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=None)
    elif dataset == 'reduced_cifar10': #!!!paper can use reduced search dataset for search (update policy network gradient)!!!
        search_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=None)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=45744, random_state=0)
        sss = sss.split(list(range(len(search_dataset))), search_dataset.targets)
        train_idx, valid_idx = next(sss)
        targets = [search_dataset.targets[idx] for idx in train_idx]
        total_trainset = Subset(search_dataset, train_idx)
        total_trainset.targets = targets
        targets = [search_dataset.targets[idx] for idx in valid_idx]
        search_dataset = Subset(search_dataset, valid_idx)
        search_dataset.targets = targets
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=None)
    elif dataset == 'svhn':
        total_trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=None)
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=None)
        search_dataset = None
    elif dataset == 'reduced_svhn':
        search_dataset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=None)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=73257-1000, random_state=0)  # 1000 + 1000 trainset
        sss = sss.split(list(range(len(search_dataset))), search_dataset.labels)
        train_idx, search_idx = next(sss)
        targets = [search_dataset.labels[idx] for idx in train_idx]
        total_trainset = Subset(search_dataset, train_idx)
        total_trainset.labels = targets
        total_trainset.targets = targets
        targets = [search_dataset.labels[idx] for idx in search_idx]
        search_dataset = Subset(search_dataset, search_idx)
        search_dataset.labels = targets
        search_dataset.targets = targets
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=None)
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    train_sampler = None
    if split < 1.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=1-split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        print(len(valid_idx))

        if target_lb >= 0:
            train_idx = [i for i in train_idx if total_trainset.targets[i] == target_lb]
            valid_idx = [i for i in valid_idx if total_trainset.targets[i] == target_lb]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    test_sampler =  None

    train_data = AugmentDataset(total_trainset, transform_train_pre, transform_train_after, transform_test, search=search, train=True)
    if search and search_dataset is not None:
        search_data = AugmentDataset(search_dataset, transform_train_pre, transform_train_after, transform_test, search=True, train=False)
    valid_data = AugmentDataset(total_trainset, transform_train_pre, transform_train_after, transform_test, search=False, train=False)
    test_data = AugmentDataset(testset, transform_train_pre, transform_train_after, transform_test, search=False, train=False)

    if train_sampler is None:
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch, shuffle=True,
            drop_last=True, pin_memory=True,
            num_workers=num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch, shuffle=False,
            sampler=train_sampler, drop_last=False,
            pin_memory=True, num_workers=num_workers)

    validloader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch,
        sampler=valid_sampler, drop_last=False,
        pin_memory=True, num_workers=num_workers)

    if search and search_dataset is not None:
        searchloader = torch.utils.data.DataLoader(
            search_data, batch_size=search_divider,
            shuffle=True, drop_last=True, pin_memory=True,
            num_workers=num_workers)
    else:
        searchloader = None

    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch,
        sampler=test_sampler, drop_last=False,
        pin_memory=True, num_workers=num_workers)

    print(f'Dataset: {dataset}')
    print(f'  |total: {len(train_data)}')
    print(f'  |train: {len(trainloader)*batch}')
    print(f'  |valid: {len(validloader)*batch}')
    print(f'  |test: {len(testloader)*batch}')
    if search and search_dataset is not None:
        print(f'  |search: {len(searchloader)*search_divider}')
    return trainloader, validloader, searchloader, testloader

def get_ts_dataloaders(dataset_name, batch, num_workers, dataroot, cutout,
                    cutout_length, split=0.5, split_idx=0, target_lb=-1,
                    search=True, search_divider=1, search_size=0, test_size=0.2, multilabel=False,
                    default_split=False):
    '''
    If search is True, dataloader will give batches of image without after_transforms,
    the transform will be done by augment agent
    If search is False, used in benchmark training
    search_size = % of data only for search
    '''
    if 'cifar10' in dataset_name:
        transform_train_pre = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        transform_train_after = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    else:   #default for time series dataset_name
        transform_train_pre = transforms.Compose([
        ])
        transform_train_after = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(_SVHN_MEAN, _SVHN_STD), assume already normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(_SVHN_MEAN, _SVHN_STD), assume already normalize
        ])

    #if cutout and cutout_length != 0:
    #    transform_train_after.transforms.append(CutoutDefault(cutout_length))
    valid_size = split
    subtrain_ratio = 1 - search_size
    '''if dataset_name == 'cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=None)
        search_dataset = None
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=None)
    elif dataset_name == 'reduced_cifar10':
        search_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=None)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=45744, random_state=0)
        sss = sss.split(list(range(len(search_dataset))), search_dataset.targets)
        train_idx, valid_idx = next(sss)
        targets = [search_dataset.targets[idx] for idx in train_idx]
        total_trainset = Subset(search_dataset, train_idx)
        total_trainset.targets = targets
        targets = [search_dataset.targets[idx] for idx in valid_idx]
        search_dataset = Subset(search_dataset, valid_idx)
        search_dataset.targets = targets
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=None)'''
    if dataset_name == 'ptbxl':
        dataset_func = PTBXL
    elif dataset_name == 'wisdm':
        dataset_func = WISDM
    elif dataset_name == 'edfx':
        dataset_func = EDFX
    elif dataset_name == 'chapman':
        dataset_func = Chapman
    else:
        ValueError(f'Invalid dataset name={dataset}')
    
    if not default_split or dataset_name=='chapman': #chapman didn't have default split now!!!
        dataset = dataset_func(dataroot,multilabel=multilabel)
        total = len(dataset)
        random.seed(0) #!!!
        rd_idxs = [i for i in range(total)]
        random.shuffle(rd_idxs)
        testset = Subset(dataset,rd_idxs[:int(total*test_size)])
        search_trainset = Subset(dataset,rd_idxs[int(total*test_size):])
    else:
        if dataset_name == 'edfx': #edfx have special split method
            dataset = dataset_func(dataroot,multilabel=multilabel)
            splits_proportions = dataset.CV_split_indices() #(k, ratio, sub_tr_idx, valid_idx, test_idx) * 5
            split_info = splits_proportions[0]
            sub_tr_idx, valid_idx, test_idx = split_info[2],split_info[3],split_info[4]
            testset = Subset(dataset,test_idx)
            search_trainset = Subset(dataset,sub_tr_idx+valid_idx)
        else:
            search_trainset = dataset_func(dataroot,mode='tottrain',multilabel=multilabel)
            testset = dataset_func(dataroot,mode='test',multilabel=multilabel)
    #make search validation set
    total = len(search_trainset)
    random.seed(0) #!!!
    rd_idxs = [i for i in range(total)]
    random.shuffle(rd_idxs)
    if search_size > 0:
        search_dataset = Subset(search_trainset,rd_idxs[:int(total*(search_size))])
        tot_train_idx = rd_idxs[int(total*(search_size)):]
        total_trainset = Subset(search_trainset,tot_train_idx)
    else:
        search_dataset = None
        total_trainset = search_trainset
            

    train_sampler = None
    if split < 1.0: 
        if not multilabel: #multilabel can't
            sss = StratifiedShuffleSplit(n_splits=5, test_size=1-split, random_state=0)
            sss = sss.split(list(range(len(total_trainset))), total_trainset.label)
        else:
            sss = ShuffleSplit(n_splits=5, test_size=1-split, random_state=0)
            sss = sss.split(list(range(len(total_trainset))))
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)
        print(len(valid_idx))
        if target_lb >= 0:
            train_idx = [i for i in train_idx if total_trainset.label[i] == target_lb]
            valid_idx = [i for i in valid_idx if total_trainset.label[i] == target_lb]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])
        
    test_sampler =  None

    train_data = AugmentDataset(total_trainset, transform_train_pre, transform_train_after, transform_test, search=search, train=True)
    if search and search_dataset is not None:
        search_data = AugmentDataset(search_dataset, transform_train_pre, transform_train_after, transform_test, search=True, train=False)
    valid_data = AugmentDataset(total_trainset, transform_train_pre, transform_train_after, transform_test, search=False, train=False)
    test_data = AugmentDataset(testset, transform_train_pre, transform_train_after, transform_test, search=False, train=False)

    if train_sampler is None:
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch, shuffle=True,
            drop_last=True, pin_memory=True,
            num_workers=num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch, shuffle=False,
            sampler=train_sampler, drop_last=False,
            pin_memory=True, num_workers=num_workers)

    validloader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch,
        sampler=valid_sampler, drop_last=False,
        pin_memory=True, num_workers=num_workers)

    if search and search_dataset is not None:
        searchloader = torch.utils.data.DataLoader(
            search_data, batch_size=search_divider,
            shuffle=True, drop_last=True, pin_memory=True,
            num_workers=num_workers)
    else:
        searchloader = None

    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch,
        sampler=test_sampler, drop_last=False,
        pin_memory=True, num_workers=num_workers)

    print(f'Dataset: {dataset_name}')
    print(f'  |multilabel: {multilabel}')
    print(f'  |total: {len(train_data)}')
    print(f'  |train: {len(trainloader)*batch}')
    print(f'  |valid: {len(validloader)*batch}')
    print(f'  |test: {len(testloader)*batch}')
    if search and search_dataset is not None:
        print(f'  |search: {len(searchloader)*search_divider}')
    return trainloader, validloader, searchloader, testloader

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
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

def get_dataset_dimension(dset):
    return {'cifar10': 32,
            'reduced_cifar10': 32,
            'svhn': 32,
            'reduced_svhn': 32,
            'edfx':3000,
            'ptbxl':1000,
            'wisdm':200,
            'chapman':5000,
            }[dset]


def get_label_name(dset, dataroot):
    if 'cifar10' in dset:
        meta = unpickle(f'{dataroot}/cifar-10-batches-py/batches.meta')
        classes = [t.decode('utf8') for t in meta[b'label_names']]
    elif 'svhn' in dset:
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        class_idxs = np.arange(0, get_num_class(dset))
        classes = [str(i) for i in class_idxs]
    return classes
