import os.path as osp
from os import listdir
import logging
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from train import naming_convention


def setup_logging(opt):
    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    return logger


def create_loaders(opt, logger):
    # Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        print(phase)
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt, opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)
    return test_loaders


def log_metrics(logger, test_set_name, test_results):
    # Average PSNR/SSIM results
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    ave_niqe = sum(test_results['niqe']) / len(test_results['niqe'])
    logger.info(
        '----Average PSNR/SSIM/NIQE results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; NIQE: {:.6f}\n'.format(
            test_set_name, ave_psnr, ave_ssim, ave_niqe))
    if test_results['psnr_y'] and test_results['ssim_y']:
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        logger.info(
            '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
                format(ave_psnr_y, ave_ssim_y))


def test_main(opt, logger, model, test_loader, export_images=False):
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['niqe'] = []

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        model.feed_data(data, need_GT=need_GT)
        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test()
        visuals = model.get_current_visuals(need_GT=need_GT)
        sr_img = util.tensor2img(visuals['SR'])  # uint8

        # save images
        if export_images:
            suffix = opt['suffix']
            if suffix:
                save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = osp.join(dataset_dir, img_name + '.png')
            util.save_img(sr_img, save_img_path)

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
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['niqe'].append(niqe)

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
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                logger.info(
                    '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; NIQE: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                        format(img_name, psnr, ssim, niqe, psnr_y, ssim_y))
            else:
                logger.info(
                    '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; NIQE: {:.6f}'.format(img_name, psnr, ssim, niqe))
        else:
            logger.info(img_name)

        if need_GT:  # metrics
            log_metrics(logger, test_set_name, test_results)


def test_harness(opt, export_images=False):
    logger = setup_logging(opt)
    test_loaders = create_loaders(opt, logger)
    model = create_model(opt)

    for test_loader in test_loaders:
        test_main(opt, logger, model, test_loader, export_images=export_images)

    logger.handlers.clear()


def get_latest_numeric_model(directory):
    name_list = [model_name.split("_")[0] for model_name in listdir(directory)]
    return str(max([int(model_name) for model_name in name_list if model_name.isdigit()])) + "_G.pth"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YAML file.')
    parser.add_argument('-export_images', help='Whether to save output images', action='store_true')
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
    export_images = parser.parse_args().export_images

    for diff in diff_list:
        for time_dependent in time_dep_list:
            for adjoint in adjoint_list:
                dataset_name = raw_opt['network_G']['training_set']
                nb = raw_opt['network_G']['nb']
                raw_opt['network_G']['diff'] = diff
                raw_opt['network_G']['time_dependent'] = time_dependent
                raw_opt['network_G']['adjoint'] = adjoint
                raw_opt['name'] = original_name + \
                                  naming_convention(dataset_name, diff, time_dependent, adjoint, nb)
                # Check whether we're in manual mode (no parameter grid testing, no model explicitly passed)
                if max(len(diff_list), len(time_dep_list), len(adjoint_list)) > 1 or \
                        raw_opt['path']['pretrain_model_G'] is None:
                    directory = "../experiments/{}/models/".format(raw_opt['name'])
                    raw_opt['path']['pretrain_model_G'] = osp.join(directory, get_latest_numeric_model(directory))
                parsed_opt = option.dict_to_nonedict(option.parse_raw(raw_opt, is_train=False))

                test_harness(parsed_opt, export_images=export_images)
