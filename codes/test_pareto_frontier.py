import os
import argparse
import math
import json
from numpy.core.defchararray import isnumeric

import options.options as option
import utils.util as util
from test import setup_logging, create_loaders, create_model
from utils.progress_bar import progress_bar


def pareto_test_main(opt, logger, model, test_loader):
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))

    idx = 0
    avg_psnr = 0
    avg_niqe = 0

    for batch_num, val_data in enumerate(test_loader):
        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
        dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)
        img_dir = os.path.join(dataset_dir, img_name)
        util.mkdir(img_dir)

        model.feed_data(val_data)
        model.test()

        visuals = model.get_current_visuals()
        sr_img = util.tensor2img(visuals['SR'])  # uint8
        # ground truth image
        # gt_img = util.tensor2img(visuals['GT'])  # uint8

        # Save SR images for reference
        save_img_path = os.path.join(img_dir,
                                     '{:s}_{:d}.png'.format(img_name, opt['model_index']))
        util.save_img(sr_img, save_img_path)

        # calculate PSNR
        item_psnr = util.tensor_psnr(model.real_H, model.fake_H)
        if math.isfinite(item_psnr):
            avg_psnr += item_psnr
            idx += 1

        # calculate NIQE
        item_niqe = util.tensor_niqe(model.fake_H)
        # item_niqe = 0
        if math.isfinite(item_niqe):
            avg_niqe += item_niqe

        progress_bar(batch_num, len(test_loader), msg=None)

    avg_psnr = avg_psnr / idx
    avg_niqe = avg_niqe / idx
    logger.info("epoch {} PSNR {} NIQE {}".format(opt['model_index'], avg_psnr, avg_niqe))

    return avg_psnr, avg_niqe


def pareto_harness(opt):
    logger = setup_logging(opt)
    test_loaders = create_loaders(opt, logger)
    model = create_model(opt)
    all_results = []

    for test_loader in test_loaders:
        all_results.append(pareto_test_main(opt, logger, model, test_loader))

    return all_results


def get_pareto_epochs(validation_results):

    pareto_epochs = []
    indexed_results = [(epoch, psnr, niqe) for epoch, (psnr, niqe) in enumerate(validation_results)]
    indexed_results.sort(key=lambda x: x[1], reverse=True)

    current_niqe = math.inf
    for epoch, psnr, niqe in indexed_results:
        if niqe < current_niqe:
            pareto_epochs.append(epoch)
            current_niqe = niqe

    return sorted(pareto_epochs)


def determine_increment(folder):
    filenames = os.listdir(folder)
    indexes = [int(filename.split("_")[0]) for filename in filenames if isnumeric(filename.split("_")[0])]
    return min(indexes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YAML file.')
    raw_opt = option.load_yaml(parser.parse_args().opt)
    opt = option.parse_raw(raw_opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    experiment_folder = os.path.join(opt['path']['root'], 'experiments', opt['name'])
    validation_results = json.load(open(os.path.join(experiment_folder, 'validation_results.json')))
    pareto_epochs = get_pareto_epochs(validation_results)
    increment = determine_increment(os.path.join(experiment_folder, 'models'))
    pareto_results = []

    print(pareto_epochs)

    for pareto_epoch in pareto_epochs:
        print(pareto_epoch)
        print(increment)
        if increment > 0:
            index = (pareto_epoch + 1) * increment
        else:
            index = pareto_epoch
        print(index)
        opt['path']['pretrain_model_G'] = os.path.join(experiment_folder, "models",
                                                       "{}_G.pth".format(index))
        opt['model_index'] = pareto_epoch

        pareto_results.append(pareto_harness(opt))

    json.dump(pareto_results, os.path.join(experiment_folder, 'pareto_results.json'))
