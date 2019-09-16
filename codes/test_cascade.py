import os.path as osp
import logging
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from test import setup_logging, create_loaders, create_model


def cascade_test_main(opt, logger, model, test_loader):
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    results = []

    try:
        total_nfe = model.netG.module.conv_trunk.nfe
    except AttributeError:
        total_nfe = None

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        model.feed_data(data, need_GT=need_GT)
        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test()
        visuals = model.get_current_visuals(need_GT=need_GT)
        sr_img = util.tensor2img(visuals['SR'])  # uint8

        # calculate PSNR and SSIM
        if need_GT:
            gt_img = util.tensor2img(visuals['GT'])
            gt_img = gt_img / 255.
            sr_img = sr_img / 255.

            crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
            niqe = util.calculate_niqe(cropped_sr_img * 255)

            if total_nfe is not None:
                last_nfe = model.netG.module.conv_trunk.nfe - total_nfe
                total_nfe = model.netG.module.conv_trunk.nfe
            else:
                last_nfe = None

            if gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                if crop_border == 0:
                    cropped_sr_img_y = sr_img_y
                    cropped_gt_img_y = gt_img_y
                else:
                    cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                results.append((psnr, ssim, niqe, psnr_y, ssim_y))
                logger.info(
                    '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; NIQE: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}; NFE: {}'.
                        format(img_name, psnr, ssim, niqe, psnr_y, ssim_y, last_nfe))
            else:
                logger.info(
                    '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; NIQE: {:.6f}; NFE: {}'.format(img_name, psnr, ssim, niqe, last_nfe))
        else:
            logger.info(img_name)

    return results


def cascade_test_harness(opt):
    logger = setup_logging(opt)
    test_loaders = create_loaders(opt, logger)
    model = create_model(opt)
    all_results = []

    for test_loader in test_loaders:
        all_results.append(cascade_test_main(opt, logger, model, test_loader))

    logger.handlers.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YAML file.')
    raw_opt = option.load_yaml(parser.parse_args().opt)
    parsed_opt = option.parse_raw(raw_opt, is_train=False)
    parsed_opt = option.dict_to_nonedict(parsed_opt)

    cascade_test_harness(parsed_opt)
