import os
import math
import argparse
import random
import logging
import json
from time import time
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from utils.progress_bar import progress_bar
from data import create_dataloader, create_dataset
from models import create_model


def init_dist(backend='nccl', **kwargs):
    # initialization for distributed training
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def create_loaders(opt, logger, rank):
    # create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_epochs = int(opt['train']['nepochs'])
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs planned: {:d}'.format(
                    total_epochs))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    return train_loader, val_loader, train_sampler


def train_main(opt, train_loader, val_loader, train_sampler, logger, resume_state=None, tb_logger=None, rank=-1):
    # create model
    model = create_model(opt)

    try:
        total_nfe = model.netG.module.conv_trunk.nfe
        nfe = True
        try:
            nfe_count = json.load(open(os.path.join(opt['path']['log'], "nfe_count.json")))
            print("resuming NFE count from {}".format(os.path.join(opt['path']['log'], "nfe_count.json")))
        except FileNotFoundError:
            print("no previous NFE count file found, starting from scratch")
            nfe_count = []
    except AttributeError:
        nfe = False
        total_nfe = None
        nfe_count = None

    best_niqe = 1e10
    best_psnr = 0
    patience = 0

    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}., psnr: {}, niqe: {}, patience: {}'.format(
            resume_state['epoch'], resume_state['iter'], resume_state['psnr'], resume_state['niqe'],
            resume_state['patience']))

        start_epoch = resume_state['epoch'] + 1
        current_step = resume_state['iter']
        best_psnr = resume_state.get('psnr', 0)
        best_niqe = resume_state.get('niqe', 1e10)
        patience = resume_state.get('patience', 0)
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    try:
        if opt['train']['G_pretraining'] >= 1:
            pretraining_epochs = opt['train']['G_pretraining']
        else:
            pretraining_epochs = 0
    except (KeyError, TypeError):
        pretraining_epochs = 0
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    total_epochs = int(opt['train']['nepochs'])
    lr_decay = opt['train']['lr_decay']
    min_lr = opt['train']['min_lr']
    pretraining = False
    all_results = []
    start_time = time()
    for epoch in range(start_epoch, total_epochs):

        if pretraining_epochs > 0:
            if epoch == 0:
                pretraining = True
                logger.info('Starting pretraining.')
            if epoch == pretraining_epochs:
                pretraining = False
                logger.info('Pretraining done, adding feature and discriminator loss.')

        if opt['dist']:
            train_sampler.set_epoch(epoch)

        if nfe:
            epoch_nfe = []

        for batch_num, train_data in enumerate(train_loader):
            # try:
                current_step += 1

                # training
                model.feed_data(train_data)
                model.optimize_parameters(current_step, pretraining=pretraining)

                if nfe:
                    last_nfe = model.netG.module.conv_trunk.nfe - total_nfe
                    total_nfe = model.netG.module.conv_trunk.nfe
                    epoch_nfe.append(last_nfe)

                progress_bar(batch_num, len(train_loader), msg=None)

            # except RuntimeError:
            #     continue
        if nfe:
            nfe_count.append(epoch_nfe)
        # log
        if epoch % opt['logger']['print_freq'] == 0:
            logs = model.get_current_log()
            message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                epoch, current_step, model.get_current_learning_rate())
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    if rank <= 0:
                        tb_logger.add_scalar(k, v, current_step)
            if rank <= 0:
                logger.info(message)

        # batched validation

        if nfe:
            epoch_nfe = []

        if epoch % opt['train']['val_freq'] == 0 and rank <= 0 and epoch >= pretraining_epochs - 1:
            avg_psnr = 0.0
            avg_niqe = 0.0
            idx = 0
            for batch_num, val_data in enumerate(val_loader):
                # try:
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()
                    if nfe:
                        last_nfe = model.netG.module.conv_trunk.nfe - total_nfe
                        total_nfe = model.netG.module.conv_trunk.nfe
                        epoch_nfe.append(last_nfe)

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    # ground truth image
                    # gt_img = util.tensor2img(visuals['GT'])  # uint8

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}.png'.format(img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    item_psnr = util.tensor_psnr(model.real_H, model.fake_H)
                    if math.isfinite(item_psnr):
                        avg_psnr += item_psnr
                        idx += 1

                    # calculate NIQE
                    if opt['niqe']:
                        item_niqe = util.tensor_niqe(model.fake_H)
                        # item_niqe = 0
                        if math.isfinite(item_niqe):
                            avg_niqe += item_niqe

                    progress_bar(batch_num, len(val_loader), msg=None)
            if nfe:
                nfe_count.append(epoch_nfe)
                json.dump(nfe_count, open(os.path.join(opt['path']['log'], 'nfe_count.json'), 'w'), indent=2)

            avg_psnr = avg_psnr / idx
            avg_niqe = avg_niqe / idx
            all_results.append((time()-start_time, avg_psnr, avg_niqe))

            # save models and training states
            if rank <= 0 and (avg_psnr > best_psnr or avg_niqe < best_niqe - 10e-6):
                logger.info('Saving models and training states.')
                model.save(epoch)
                model.save_training_state(epoch, current_step, best_psnr, best_niqe, patience)

            else:
                patience += 1
                if patience == opt['train']['epoch_patience']:
                    model.update_learning_rate(lr_decay)
                    print("no improvement, final patience, updating learning rate to {}".format(model.get_current_learning_rate()))
                    patience = 0
                else:
                    print("no improvement, patience {} out of {}".format(patience, opt['train']['epoch_patience']))
                if model.get_current_learning_rate() < min_lr:
                    break

            if avg_niqe < best_niqe:
                best_niqe = avg_niqe

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr

            # log
            logger.info('# Validation # PSNR: {:.4e} # NIQE: {:.4e}'.format(avg_psnr, avg_niqe))
            logger_val = logging.getLogger('val')  # validation logger
            logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} niqe: {:.4e} (best: {:.4e}/{:.4e})'.format(
                epoch, current_step, avg_psnr, avg_niqe, best_psnr, best_niqe))
            # tensorboard logger
            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                tb_logger.add_scalar('psnr', avg_psnr, current_step)

        print('\n')

    if rank <= 0:
        # save results
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        json.dump(all_results, open(os.path.join(opt['path']['log'], 'validation_results.json'), 'w'), indent=2)
        if nfe:
            nfe_count.append(epoch_nfe)
            json.dump(nfe_count, open(os.path.join(opt['path']['log'], 'nfe_count.json'), 'w'), indent=2)

        # clear validation logger
        logger_val.handlers.clear()

        # print out graph of val psnr with time
        fig, ax = plt.subplots()
        y = list(zip(*all_results))
        runtime, dev_psnr, dev_niqe = y[0], y[1], y[2]
        ax.plot(runtime, dev_psnr, color='blue', label='Validation PSNR')

        ax.set(xlabel='Time (s)', ylabel='Dev. PSNR.')
        ax.legend(loc='upper right')
        ax.grid()

        plt.savefig(os.path.join(opt['path']['log'], "psnr_evolution.png"))


def get_resume_state(opt):
    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['epoch'])  # check resume options
    else:
        resume_state = None
    return resume_state


def setup_logging(opt, resume_state, rank):
    tb_logger = None
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')
    return logger, tb_logger


def train_harness(opt, rank, main_loop=train_main):
    resume_state = get_resume_state(opt)

    # mkdir and loggers
    logger, tb_logger = setup_logging(opt, resume_state, rank)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # loaders
    train_loader, val_loader, train_sampler = create_loaders(opt, logger, rank)

    # training
    main_loop(opt, train_loader, val_loader, train_sampler, logger, resume_state, tb_logger, rank)

    logger.handlers.clear()


def naming_convention(dataset_name, diff, time_dependent, adjoint, nb):
    return "_{}_{}{}{}_{}".format(dataset_name, diff, "_time" if time_dependent and diff else "", "_adjoint" if adjoint and diff else "", nb)


if __name__ == '__main__':

    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-dl', '--diff-list', nargs='+', default=[])
    parser.add_argument('-td', '--time-dep-list', nargs='+', default=[])
    parser.add_argument('-ad', '--adjoint-list', nargs='+', default=[])
    args = parser.parse_args()
    raw_opt = option.load_yaml(args.opt)

    diff_list = args.diff_list if len(args.diff_list) > 0 else [raw_opt['network_G']['diff']]
    time_dep_list = [eval(value) for value in args.time_dep_list] if len(args.time_dep_list) > 0 \
        else [raw_opt['network_G']['time_dependent']]
    adjoint_list = [eval(value) for value in args.adjoint_list] if len(args.adjoint_list) > 0\
        else [raw_opt['network_G']['adjoint']]
    original_name = raw_opt['name']

    for diff in diff_list:
        # time_dependency and adjoint/discrete parameters only make sense for a set of diff values
        grid_testing = diff not in ["None", False, "checkpointed"]
        for time_dependent in time_dep_list if grid_testing else [False]:
            for adjoint in adjoint_list if grid_testing else [False]:
                dataset_name = raw_opt['datasets']['train']['name']
                nb = raw_opt['network_G']['nb']
                raw_opt['network_G']['diff'] = diff
                raw_opt['network_G']['time_dependent'] = time_dependent
                raw_opt['network_G']['adjoint'] = adjoint
                raw_opt['name'] = original_name + \
                                  naming_convention(dataset_name, diff, time_dependent, adjoint, nb)
                parsed_opt = option.parse_raw(raw_opt, is_train=True)

                # distributed training settings
                if args.launcher == 'none':  # disabled distributed training
                    parsed_opt['dist'] = False
                    rank = -1
                    print('Disabled distributed training.')
                else:
                    parsed_opt['dist'] = True
                    init_dist()
                    world_size = torch.distributed.get_world_size()
                    rank = torch.distributed.get_rank()

                train_harness(parsed_opt, rank)


