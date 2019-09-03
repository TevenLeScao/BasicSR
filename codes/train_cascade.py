import argparse
import os
import math
import logging
import json

import torch
import torch.distributed as dist

import options.options as option
from utils import util
from utils.progress_bar import progress_bar
from data import create_dataloader, create_dataset
from models import create_model
from train import init_dist, training_harness


def train_psnr(opt, train_loader, val_loader, train_sampler, logger, resume_state=None, tb_logger=None, rank=-1):
    # create model
    model = create_model(opt)

    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    total_epochs = int(opt['train']['nepochs'])
    best_psnr = 0
    patience = 0
    all_results = []
    for epoch in range(start_epoch, total_epochs):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for batch_num, train_data in enumerate(train_loader):
            current_step += 1
            # update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step, pretraining=True, discriminator=False)

            progress_bar(batch_num, len(train_loader), msg=None)

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
        if epoch % opt['train']['val_freq'] == 0 and rank <= 0:
            avg_psnr = 0.0
            idx = 0
            for batch_num, val_data in enumerate(val_loader):
                img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                img_dir = os.path.join(opt['path']['val_images'], img_name)
                util.mkdir(img_dir)

                model.feed_data(val_data)
                model.test()

                # calculate PSNR
                item_psnr = util.tensor_psnr(model.real_H, model.fake_H)
                if math.isfinite(item_psnr):
                    avg_psnr += item_psnr
                    idx += 1

                progress_bar(batch_num, len(val_loader), msg=None)

            avg_psnr = avg_psnr / idx
            all_results.append(avg_psnr)

            if avg_psnr < best_psnr:
                patience += 1
                if patience == opt['train']['epoch_patience']:
                    break

            else:
                best_psnr = avg_psnr
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save('latest')

            # log
            logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
            logger_val = logging.getLogger('val')  # validation logger
            logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} (best: {:.4e})'.format(
                epoch, current_step, avg_psnr, best_psnr))
            # tensorboard logger
            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                tb_logger.add_scalar('psnr', avg_psnr, current_step)

    if rank <= 0:
        logger.info('End of training.')
        json.dump(all_results, open(os.path.join(opt['path']['log'], 'validation_results.json'), 'w'), indent=2)


if __name__ == '__main__':

    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    raw_opt = option.load_yaml(args.opt)
    raw_opt['train']['G_pretraining'] = raw_opt['train']['nepochs']

    for cascade_width in [2, 4, 8, 16, 32, 64]:
        raw_opt['network_G']['nf'] = cascade_width
        raw_opt['name'] = raw_opt['name'] + '_cascade_{}'.format(cascade_width)
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

        training_harness(parsed_opt, rank, training_function=train_psnr)
