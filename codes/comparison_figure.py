import json
import matplotlib.pyplot as plt
import argparse
import options.options as option
import os


def output_comparison_figure(diff_results, clas_results):
    diff_x, diff_y = zip(*diff_results)
    clas_x, clas_y = zip(*clas_results)
    fig = plt.figure()
    axes = plt.gca()
    axes.set_xlim([0,30])
    axes.set_ylim([0,15])
    ax1 = fig.add_subplot(111)
    ax1.scatter(diff_x, diff_y, s=10, c='r', marker="*", label='Differential network')
    ax1.scatter(clas_x, clas_y, s=10, c='b', marker="o", label='Classical network')
    plt.legend(loc='upper left')
    plt.xlabel("PSNR")
    plt.ylabel("NIQE")
    plt.savefig("compare_diff_classical.png")


if __name__ == "__main__":
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--diff_opt', type=str, required=True, help='Path to diff YAML file.')
    parser.add_argument('--clas_opt', type=str, required=True, help='Path to classical YAML file.')

    diff_opt = option.parse(parser.parse_args().diff_opt, is_train=True)
    diff_opt = option.dict_to_nonedict(diff_opt)
    clas_opt = option.parse(parser.parse_args().clas_opt, is_train=True)
    clas_opt = option.dict_to_nonedict(clas_opt)

    diff_results = json.load(open(os.path.join(diff_opt['path']['log'], 'validation_results.json')))
    clas_results = json.load(open(os.path.join(clas_opt['path']['log'], 'validation_results.json')))

    output_comparison_figure(diff_results, clas_results)
