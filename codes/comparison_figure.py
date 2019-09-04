import json
import matplotlib.pyplot as plt
import argparse
import options.options as option
import os.path as osp


def output_comparison_figure(diff_results, clas_results, baseline_scores, output_path="compare_diff_classical.png"):
    diff_x, diff_y = zip(*diff_results)
    clas_x, clas_y = zip(*clas_results)
    fig = plt.figure()
    axes = plt.gca()
    axes.set_xlim([10, 30])
    axes.set_ylim([3, 13])
    ax1 = fig.add_subplot(111)
    ax1.scatter(diff_x, diff_y, s=4, c='r', marker="*", label='Differential network')
    ax1.scatter(clas_x, clas_y, s=3, c='b', marker="o", label='Classical network')
    if baseline_scores is not None:
        base_x, base_y = baseline_scores
        ax1.scatter(base_x, base_y, s=5, c='g', marker="+", label='Baseline')
    plt.legend(loc='upper left')
    plt.xlabel("PSNR")
    plt.ylabel("NIQE")
    plt.savefig(output_path)


if __name__ == "__main__":
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--diff_opt', type=str, default=None, required=False, help='Path to diff YAML file.')
    parser.add_argument('--clas_opt', type=str, default=None, required=False, help='Path to classical YAML file.')
    parser.add_argument('--diff_name', type=str, default=None, required=False, help='Diff model name.')
    parser.add_argument('--clas_name', type=str, default=None, required=False, help='Clas model name.')

    args = parser.parse_args()
    diff_path, clas_path = None, None

    if args.diff_opt is not None:
        diff_opt = option.parse(args.diff_opt, is_train=True)
        diff_opt = option.dict_to_nonedict(diff_opt)
        diff_path = diff_opt['path']['log']
    if args.clas_opt is not None:
        clas_opt = option.parse(args.clas_opt, is_train=True)
        clas_opt = option.dict_to_nonedict(clas_opt)
        clas_path = clas_opt['path']['log']
    if args.diff_name is not None:
        diff_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, "experiments", args.diff_name))
    if args.clas_name is not None:
        clas_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, "experiments", args.clas_name))

    assert diff_path is not None and clas_path is not None

    diff_results = json.load(open(osp.join(diff_path, 'validation_results.json')))
    clas_results = json.load(open(osp.join(clas_path, 'validation_results.json')))

    baseline_scores = None
    if "BSD" in diff_path:
        baseline_scores = [24.1], [10.2]
    if "DIV" in diff_path:
        baseline_scores = [27.2], [10.7]

    output_comparison_figure(diff_results, clas_results, baseline_scores,
                             osp.join(diff_path, 'compare_diff_classical.png'))
