from os import path as osp, listdir
import argparse
import math
import json
from numpy.core.defchararray import isnumeric

import options.options as option
import utils.util as util
from test import setup_logging, create_loaders, create_model
from utils.progress_bar import progress_bar


def pareto_test_main(opt, logger, model, test_loader, export_images=False):
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))

    idx = 0
    avg_psnr = 0
    avg_niqe = 0

    for batch_num, val_data in enumerate(test_loader):
        img_name = osp.splitext(osp.basename(val_data['LQ_path'][0]))[0]
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        model.feed_data(val_data)
        model.test()

        visuals = model.get_current_visuals()
        sr_img = util.tensor2img(visuals['SR'])  # uint8
        # ground truth image
        # gt_img = util.tensor2img(visuals['GT'])  # uint8

        # Save SR images for reference
        if export_images:
            suffix = opt['suffix']
            if suffix:
                save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = osp.join(dataset_dir, img_name + '.png')
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
    logger.info("PSNR {} NIQE {}".format(avg_psnr, avg_niqe))

    return avg_psnr, avg_niqe


def pareto_harness(opt, export_images=False):
    logger = setup_logging(opt)
    test_loaders = create_loaders(opt, logger)
    model = create_model(opt)
    all_results = []

    for test_loader in test_loaders:
        all_results.append(pareto_test_main(opt, logger, model, test_loader, export_images=export_images))

    logger.handlers.clear()

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
    filenames = listdir(folder)
    indexes = [int(filename.split("_")[0]) for filename in filenames if isnumeric(filename.split("_")[0])]
    return min(indexes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YAML file.')
    parser.add_argument('--export-images', help='Whether to save output images', action='store_true')
    parser.add_argument('--pareto', help='Only keep pareto frontier epochs', action='store_true')
    args = parser.parse_args()
    raw_opt = option.load_yaml(args.opt)
    opt = option.parse_raw(raw_opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    experiment_folder = osp.join(opt['path']['root'], 'experiments', opt['name'])
    pareto_results = {}
    export_images = args.export_images

    if args.pareto:
        validation_results = json.load(open(osp.join(experiment_folder, 'validation_results.json')))
        pareto_epochs = get_pareto_epochs(validation_results)
        # backwards compatibility with step-counted models, no change for current epoch-counted models, deprecate later
        increment = determine_increment(osp.join(experiment_folder, 'models'))
        print("Pareto epochs: {}".format(pareto_epochs))

    else:
        pareto_epochs = [model_name.split("_")[0]
                         for model_name in listdir(osp.join(experiment_folder, "models"))]
        print("Using all epochs: {}".format(pareto_epochs))
        increment = 0

    for pareto_epoch in pareto_epochs:

        # load current model
        if increment > 0:
            index = (pareto_epoch + 1) * increment
        else:
            index = pareto_epoch
        opt['path']['pretrain_model_G'] = osp.join(experiment_folder, "models",
                                                       "{}_G.pth".format(index))

        # setup different logging dir for every model
        opt['name'] = raw_opt['name'] + "_epoch_{}".format(pareto_epoch)
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

        # run things !
        pareto_results[pareto_epoch] = pareto_harness(opt, export_images=export_images)

    json.dump(pareto_results, open(osp.join(experiment_folder, 'pareto_results.json'), 'w'))
