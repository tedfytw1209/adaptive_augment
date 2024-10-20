from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import time
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils

from adaptive_augmentor import AdaAug
from networks import get_model
from networks.projection import Projection
from config import get_search_divider
from dataset import get_dataloaders, get_num_class, get_label_name, get_dataset_dimension

parser = argparse.ArgumentParser("ada_aug")
parser.add_argument('--dataroot', type=str, default='./', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
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
parser.add_argument('--train_portion', type=float, default=1, help='portion of training data')
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

args = parser.parse_args()
debug = True if args.save == "debug" else False
args.save = '{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.save)
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

    #  dataset settings
    n_class = get_num_class(args.dataset)
    sdiv = get_search_divider(args.model_name)
    class2label = get_label_name(args.dataset, args.dataroot)

    train_queue, valid_queue, search_queue, test_queue = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        args.dataroot, args.cutout, args.cutout_length,
        split=args.train_portion, split_idx=0, target_lb=-1,
        search=True, search_divider=sdiv)

    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'  |total: {len(train_queue.dataset)}')
    logging.info(f'  |train: {len(train_queue)*args.batch_size}')
    logging.info(f'  |valid: {len(valid_queue)*args.batch_size}')
    logging.info(f'  |search: {len(search_queue)*sdiv}')

    #  model settings
    gf_model = get_model(model_name=args.model_name, num_class=n_class,
        use_cuda=True, data_parallel=False)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(gf_model))

    h_model = Projection(in_features=gf_model.fc.in_features,
        n_layers=args.n_proj_layer, n_hidden=128).cuda()

    #  training settings
    gf_optimizer = torch.optim.SGD(
        gf_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gf_optimizer,
        float(args.epochs), eta_min=args.learning_rate_min)

    h_optimizer = torch.optim.Adam(
        h_model.parameters(),
        lr=args.proj_learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.proj_weight_decay)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    #  AdaAug settings
    after_transforms = train_queue.dataset.after_transforms
    adaaug_config = {'sampling': 'prob',
                    'k_ops': 1, #as paper
                    'delta': 0.0, 
                    'temp': 1.0, 
                    'search_d': get_dataset_dimension(args.dataset),
                    'target_d': get_dataset_dimension(args.dataset)}

    adaaug = AdaAug(after_transforms=after_transforms,
        n_class=n_class,
        gf_model=gf_model,
        h_model=h_model,
        save_dir=args.save,
        config=adaaug_config)

    #  Start training
    start_time = time.time()
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # searching
        train_acc, train_obj = train(train_queue, search_queue, gf_model, adaaug,
            criterion, gf_optimizer, args.grad_clip, h_optimizer, epoch, args.search_freq)

        # validation
        valid_acc, valid_obj = infer(valid_queue, gf_model, criterion)

        logging.info(f'train_acc {train_acc} valid_acc {valid_acc}')
        scheduler.step()

        utils.save_model(gf_model, os.path.join(args.save, 'gf_weights.pt'))
        utils.save_model(h_model, os.path.join(args.save, 'h_weights.pt'))

    end_time = time.time()
    elapsed = end_time - start_time

    test_acc, test_obj = infer(test_queue, gf_model, criterion)
    utils.save_model(gf_model, os.path.join(args.save, 'gf_weights.pt'))
    utils.save_model(h_model, os.path.join(args.save, 'h_weights.pt'))
    adaaug.save_history(class2label)
    figure = adaaug.plot_history()

    logging.info(f'test_acc {test_acc}')
    logging.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logging.info(f'saved to: {args.save}')

def train(train_queue, search_queue, gf_model, adaaug, criterion, gf_optimizer,
            grad_clip, h_optimizer, epoch, search_freq):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)

        # exploitation
        timer = time.time()
        aug_images = adaaug(input, mode='exploit')
        gf_model.train()
        gf_optimizer.zero_grad()
        logits = gf_model(aug_images)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(gf_model.parameters(), grad_clip)
        gf_optimizer.step()

        #  stats
        prec1, prec5 = utils.accuracy(logits.detach(), target.detach(), topk=(1, 5))
        n = target.size(0)
        objs.update(loss.detach().item(), n)
        top1.update(prec1.detach().item(), n)
        top5.update(prec5.detach().item(), n)
        exploitation_time = time.time() - timer

        # exploration
        timer = time.time()
        if step % search_freq == 0:
            input_search, target_search = next(iter(search_queue))
            target_search = target_search.cuda(non_blocking=True)

            h_optimizer.zero_grad()
            mixed_features = adaaug(input_search, mode='explore')
            logits = gf_model.g(mixed_features)
            loss = criterion(logits, target_search)
            loss.backward()
            h_optimizer.step()
            exploration_time = time.time() - timer

            #  log policy
            adaaug.add_history(input_search, target_search)

        global_step = epoch * len(train_queue) + step
        if global_step % args.report_freq == 0:
            logging.info('  |train %03d %e %f %f | %.3f + %.3f s', global_step,
                objs.avg, top1.avg, top5.avg, exploitation_time, exploration_time)

    return top1.avg, objs.avg


def infer(valid_queue, gf_model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    gf_model.eval()

    with torch.no_grad():
        for input, target in valid_queue:
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = gf_model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.detach().item(), n)
            top1.update(prec1.detach().item(), n)
            top5.update(prec5.detach().item(), n)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
