from Node_level_Models.configs.config import args_parser
from Node_level_Models.helpers.metrics_utils import log_test_results


import numpy as np
import wandb

from helperFunction import set_random_seed
args = args_parser()
project_name = [args.proj_name, args.proj_name+ "debug"]
proj_name = project_name[0]
import os


def main(args):
    model_name = args.model
    # 'data-{}_model-{}_IID-{}_num_workers-{}_num_mali-{}_epoch_backdoor-{}_frac_of_avg-{}_trigger_type-{}_trigger_position-{}_poisoning_intensity-{}'
    Alg_name = "Alg-" + args.agg_method
    file_name = Alg_name + 'D-{}_M-{}_IID-{}_NW-{}_NM-{}_EB-{}_TS-{}_TPye-{}_TPo-{}_PI-{}_OR-{}'.format(
        args.dataset,
        model_name,
        args.is_iid,
        args.num_workers,
        args.num_mali,
        args.epoch_backdoor,
        args.trigger_size,
        args.trigger_type,
        args.trigger_position,
        args.poisoning_intensity,
        args.overlapping_rate,
    )

    average_overall_performance_list, average_ASR_list, average_Flip_ASR_list, average_transfer_attack_success_rate_list = [], [], [], []
    a_list, r_list, v_list = [], [], []
    results_table = []
    metric_list = []
    if args.agg_method == "scaffold":
        from backdoor_node_clf_scaffold import main as backdoor_main
    else:
        from backdoor_node_clf import main as backdoor_main

    # 全局设置随机种子
    set_random_seed(args.seed)
    rs = np.random.RandomState(args.seed)
    for i in range(15):
        # 为每次实验设置随机种子
        set_random_seed(args.seed)

        # wandb init
        logger = wandb.init(
            # entity="hkust-gz",
            project=proj_name,
            group=file_name,
            name=f"round_{i}",
            config=args,
        )

        average_overall_performance, average_ASR, average_Flip_ASR, average_transfer_attack_success_rate, a, r, v = backdoor_main(
            args, logger)
        results_table.append(
            [average_overall_performance, average_ASR, average_Flip_ASR, average_transfer_attack_success_rate])
        a_list.append(a)
        r_list.append(r)
        v_list.append(v)
        logger.log({"average_overall_performance": average_overall_performance,
                    "average_ASR": average_ASR,
                    "average_Flip_ASR": average_Flip_ASR,
                    "average_transfer_attack_success_rate": average_transfer_attack_success_rate})

        average_overall_performance_list.append(average_overall_performance)
        average_ASR_list.append(average_ASR)
        average_Flip_ASR_list.append(average_Flip_ASR)
        average_transfer_attack_success_rate_list.append(average_transfer_attack_success_rate)
        # end the logger
        wandb.finish()

    # wandb table logger init
    columns = ["average_overall_performance", "average_ASR", "average_Flip_ASR", "average_transfer_attack_success_rate"]
    logger_table = wandb.Table(columns=columns, data=results_table)
    table_logger = wandb.init(
        # entity="hkust-gz",
        project=proj_name,
        group=file_name,
        name=f"exp_results",
        config=args,
    )
    table_logger.log({"results": logger_table})
    wandb.finish()

    mean_average_overall_performance = np.mean(np.array(average_overall_performance_list))
    mean_average_ASR = np.mean(np.array(average_ASR_list))
    mean_average_Flip_ASR = np.mean(np.array(average_Flip_ASR_list))
    mean_average_transfer_attack_success_rate = np.mean(np.array(average_transfer_attack_success_rate_list))

    std_average_overall_performance = np.std(np.array(average_overall_performance_list))
    std_average_ASR = np.std(np.array(average_ASR_list))
    std_average_Flip_ASR = np.std(np.array(average_Flip_ASR_list))
    std_average_transfer_attack_success_rate = np.std(np.array(average_transfer_attack_success_rate_list))

    header = ['dataset', 'model', "mean_average_overall_performance",
              "std_average_overall_performance", "mean_average_ASR", "std_average_ASR",
              "mean_average_Flip_ASR", "std_average_Flip_ASR",
              "mean_average_local_unchanged_acc", "std_average_transfer_attack_success_rate"]
    paths = "./checkpoints/Node/"

    metric_list.append(args.dataset)
    metric_list.append(model_name)
    metric_list.append(mean_average_overall_performance)
    metric_list.append(std_average_overall_performance)
    metric_list.append(mean_average_ASR)
    metric_list.append(std_average_ASR)
    metric_list.append(mean_average_Flip_ASR)
    metric_list.append(std_average_Flip_ASR)
    metric_list.append(mean_average_transfer_attack_success_rate)
    metric_list.append(std_average_transfer_attack_success_rate)

    paths = paths + "data-{}/".format(args.dataset) + "model-{}/".format(model_name) + file_name
    log_test_results(paths, header, file_name)
    log_test_results(paths, metric_list, file_name)

    # 写入txt文件
    if args.is_test == 0:
        result_dir = f"./results/{args.dataset}_{args.agg_method}_{args.num_mali}_results"
    else:
        result_dir = f"./results/test_{args.dataset}_{args.agg_method}_{args.num_mali}_results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_file = os.path.join(result_dir, f"{file_name}.txt")
    with open(result_file, 'w') as f:

        f.write(f"agg_method: {args.agg_method}\n")
        f.write(f"num_mali: {args.num_mali}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"inner_epochs: {args.inner_epochs}\n")
        f.write(f"device_id: {args.device_id}\n")
        f.write(f"inner_epochs: {args.inner_epochs}\n")
        f.write(f"ratio_training: {args.ratio_training}\n")
        f.write(f"ratio_val: {args.ratio_val}\n")
        f.write(f"ratio_testing: {args.ratio_testing}\n")

        for i in range(15):
            f.write(f"Experiment {i + 1}: a={a_list[i]}, r={r_list[i]}, v={v_list[i]}\n")

        # 记录v最高的5组并取其均值
        top_5_v = sorted(v_list, reverse=True)[:5]
        avg_top_5_v = np.mean(top_5_v)
        f.write(f"Top 5 v: {top_5_v}\n")
        f.write(f"Average of top 5 v: {avg_top_5_v}\n")

if __name__ == '__main__':
    args = args_parser()
    main(args)
