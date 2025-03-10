import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn

from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_cell_based_tiny_net, get_search_spaces
from nas_201_api import NASBench201API as API
from sparsemax import Sparsemax
sparsemax = Sparsemax(dim=1)

def _generate_z(network):
    # Generate a direction vector z (column vector) of length |alpha|.
    v = network.module.arch_parameters
    length = v.shape[0] * v.shape[1]  # To record all the lengths, spell them out as a vector.
    rand = torch.randn((length, 1))
    rand = rand / torch.norm(rand)
    z = rand.cuda()
    return z

def _prepare_disturbance(network, epoch, logger):  # Generate perturbed structural coefficients.
    model_new = deepcopy(network)
    z = _generate_z(network)  # Generate perturbation vectors.
    arch_param_vector = torch.cat([torch.flatten(v.data) for v in model_new.module.arch_parameters], dim=0)
    norm = torch.norm(arch_param_vector)
    mu = norm * 0.005
    logger.log("/epoch="+epoch+" norm={:.2f}".format(norm.item()))
    perturbation = torch.reshape(z, model_new.module.arch_parameters.shape) * mu
    model_new.module.arch_parameters.data = model_new.module.arch_parameters.data + perturbation.data
    return model_new, z, mu

def _backward_step_ours(
    network,
    network_perturb,
    criterion,
    arch_inputs,
    arch_targets,
    z,
    mu
):
    unrolled_model = deepcopy(network)
    unrolled_model.zero_grad()
    _, unrolled_logits = unrolled_model(arch_inputs)
    unrolled_loss = criterion(unrolled_logits, arch_targets)
    unrolled_loss.backward()

    implicit_grads = [v1.data - v2.data for v1, v2 in zip(network_perturb.module.get_weights(), network.module.get_weights())]
    implicit_grads = torch.cat([torch.flatten(v) for v in implicit_grads], dim=0).unsqueeze(dim=1) / mu
    
    grad_Lval_w = [v.grad.data for v in unrolled_model.module.get_weights()]
    grad_Lval_w = torch.cat([torch.flatten(v) for v in grad_Lval_w], dim=0).unsqueeze(dim=1)

    grad_Lval_a = unrolled_model.module.arch_parameters.grad
    grad_Lval_a = torch.cat([torch.flatten(v) for v in grad_Lval_a], dim=0).unsqueeze(dim=1)

    approx_grad = torch.mul((torch.mm(grad_Lval_a.t(), z) + torch.mm(implicit_grads.t(), grad_Lval_w)), z)
    approx_grad = torch.reshape(approx_grad, network.module.arch_parameters.shape)

    if network.module.arch_parameters.grad is None:
        network.module.arch_parameters.grad = deepcopy(approx_grad)
    else:
        network.module.arch_parameters.grad.data.copy_(approx_grad.data)

    return unrolled_loss.detach(), unrolled_logits.detach()


def _backward_step_ours_penalty(
        network,
        network_perturb,
        criterion,
        arch_inputs,
        arch_targets,
        z,
        mu
):
    unrolled_model = deepcopy(network)
    unrolled_model.zero_grad()
    _, unrolled_logits = unrolled_model(arch_inputs)
    unrolled_loss = criterion(unrolled_logits, arch_targets)
    unrolled_loss.backward()

    implicit_grads = [v1.data - v2.data for v1, v2 in
                      zip(network_perturb.module.get_weights(), network.module.get_weights())]
    implicit_grads = torch.cat([torch.flatten(v) for v in implicit_grads], dim=0).unsqueeze(dim=1) / mu

    grad_Lval_w = [v.grad.data for v in unrolled_model.module.get_weights()]
    grad_Lval_w = torch.cat([torch.flatten(v) for v in grad_Lval_w], dim=0).unsqueeze(dim=1)

    grad_Lval_a = unrolled_model.module.arch_parameters.grad
    grad_Lval_a = torch.cat([torch.flatten(v) for v in grad_Lval_a], dim=0).unsqueeze(dim=1)

    approx_grad = torch.mul((torch.mm(grad_Lval_a.t(), z) + torch.mm(implicit_grads.t(), grad_Lval_w)), z)
    approx_grad = torch.reshape(approx_grad, network.module.arch_parameters.shape)

    alpha = nn.functional.softmax(network.module.arch_parameters, dim=-1)
    gamma = 0.2
    grad_penalty = gamma * (2 * alpha - torch.ones_like(network.module.arch_parameters))
    grad_penalty = torch.reshape(grad_penalty, z.shape)
    penalty_grad_term = torch.mul(torch.mm(grad_penalty.t(), z), z)
    penalty_grad_term = torch.reshape(penalty_grad_term, approx_grad.shape)
    final_grad = approx_grad + penalty_grad_term

    if network.module.arch_parameters.grad is None:
        network.module.arch_parameters.grad = deepcopy(final_grad)
    else:
        network.module.arch_parameters.grad.data.copy_(final_grad.data)

    return unrolled_loss.detach(), unrolled_logits.detach()

def search_func_ours_F(
    xloader,
    network,
    criterion,
    scheduler,
    w_optimizer,
    a_optimizer,
    epoch_str,
    print_freq,
    logger,
):
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.train()
    end = time.time()
    rounds = 10

    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(
        xloader
    ):
        scheduler.update(None, 1.0 * step / len(xloader))
        base_targets = base_targets.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)

        if step % rounds == 0:
            # get a random minibatch from the search queue with replacement
            LR, WD, momentum = (
                w_optimizer.param_groups[0]["lr"],
                w_optimizer.param_groups[0]["weight_decay"],
                w_optimizer.param_groups[0]["momentum"],
            )

            network_perturb, z, mu = _prepare_disturbance(network, epoch_str, logger)

            optimizer_perturb = torch.optim.SGD(
                network_perturb.parameters(),
                LR,
                momentum=momentum,
                weight_decay=WD)

        w_optimizer.zero_grad()
        _, logits = network(base_inputs)
        base_loss = criterion(logits, base_targets)
        base_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        w_optimizer.step()

        optimizer_perturb.zero_grad()
        _, logits_perturb = network_perturb(base_inputs)
        perturb_loss = criterion(logits_perturb, base_targets)
        perturb_loss.backward()
        torch.nn.utils.clip_grad_norm_(network_perturb.parameters(), 5)
        optimizer_perturb.step()

        if (step + 1) % rounds == 0:
            a_optimizer.zero_grad()
            arch_loss, arch_logits = _backward_step_ours(network, network_perturb, criterion, arch_inputs, arch_targets, z, mu)
            a_optimizer.step()

            # Due to the sparsity of our iterations, the amount of times here will also be reduced, so if you have problems running it, consider just removing the following records!!!
            arch_prec1, arch_prec5 = obtain_accuracy(
                arch_logits.data, arch_targets.data, topk=(1, 5)
            )
            arch_losses.update(arch_loss.item(), arch_inputs.size(0))
            arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
            arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

        # record
        base_prec1, base_prec5 = obtain_accuracy(
            logits.data, base_targets.data, topk=(1, 5)
        )
        base_losses.update(base_loss.item(), base_inputs.size(0))
        base_top1.update(base_prec1.item(), base_inputs.size(0))
        base_top5.update(base_prec5.item(), base_inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
            Sstr = (
                "*SEARCH* "
                + time_string()
                + " [{:}][{:03d}/{:03d}]".format(epoch_str, step, len(xloader))
            )
            Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                batch_time=batch_time, data_time=data_time
            )
            Wstr = "Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                loss=base_losses, top1=base_top1, top5=base_top5
            )
            Astr = "Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                loss=arch_losses, top1=arch_top1, top5=arch_top5
            )
            logger.log(Sstr + " " + Tstr + " " + Wstr + " " + Astr)
    return base_losses.avg, base_top1.avg, base_top5.avg


def valid_func(xloader, network, criterion):
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.eval()
    end = time.time()
    with torch.no_grad():
        for step, (arch_inputs, arch_targets) in enumerate(xloader):
            arch_targets = arch_targets.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - end)
            # prediction
            _, logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)
            # record
            arch_prec1, arch_prec5 = obtain_accuracy(
                logits.data, arch_targets.data, topk=(1, 5)
            )
            arch_losses.update(arch_loss.item(), arch_inputs.size(0))
            arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
            arch_top5.update(arch_prec5.item(), arch_inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    train_data, valid_data, xshape, class_num = get_datasets(
        xargs.dataset, xargs.data_path, -1
    )
    config = load_config(
        xargs.config_path, {"class_num": class_num, "xshape": xshape}, logger
    )
    search_loader, _, valid_loader = get_nas_search_loaders(
        train_data,
        valid_data,
        xargs.dataset,
        "../../configs/nas-benchmark/",
        config.batch_size,
        xargs.workers,
    )
    logger.log(
        "||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(
            xargs.dataset, len(search_loader), len(valid_loader), config.batch_size
        )
    )
    logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

    search_space = get_search_spaces("cell", xargs.search_space_name)
    model_config = dict2config(
        {
            "name": "SPARSEZO",
            "C": xargs.channel,
            "N": xargs.num_cells,
            "max_nodes": xargs.max_nodes,
            "num_classes": class_num,
            "space": search_space,
            "affine": False,
            "track_running_stats": bool(xargs.track_running_stats),
        },
        None,
    )
    search_model = get_cell_based_tiny_net(model_config)
    logger.log("search-model :\n{:}".format(search_model))

    w_optimizer, w_scheduler, criterion = get_optim_scheduler(
        search_model.get_weights(), config
    )
    a_optimizer = torch.optim.Adam(
        search_model.get_alphas_list(),
        lr=xargs.arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=xargs.arch_weight_decay,
    )
    logger.log("w-optimizer : {:}".format(w_optimizer))
    logger.log("a-optimizer : {:}".format(a_optimizer))
    logger.log("w-scheduler : {:}".format(w_scheduler))
    logger.log("criterion   : {:}".format(criterion))
    flop, param = get_model_infos(search_model, xshape)

    logger.log("FLOP = {:.2f} M, Params = {:.2f} MB".format(flop, param))
    if xargs.arch_nas_dataset is None:
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log("{:} create API = {:} done".format(time_string(), api))

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )
    network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start".format(last_info)
        )
        last_info = torch.load(last_info)
        start_epoch = last_info["epoch"]
        checkpoint = torch.load(last_info["last_checkpoint"])
        genotypes = checkpoint["genotypes"]
        valid_accuracies = checkpoint["valid_accuracies"]
        search_model.load_state_dict(checkpoint["search_model"])
        w_scheduler.load_state_dict(checkpoint["w_scheduler"])
        w_optimizer.load_state_dict(checkpoint["w_optimizer"])
        a_optimizer.load_state_dict(checkpoint["a_optimizer"])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
                last_info, start_epoch
            )
        )
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, genotypes = (
            0,
            {"best": -1},
            {-1: search_model.genotype()},
        )

    # start training
    start_time, search_time, epoch_time, total_epoch = (
        time.time(),
        AverageMeter(),
        AverageMeter(),
        config.epochs + config.warmup,
    )
    for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.val * (total_epoch - epoch), True)
        )
        epoch_str = "{:03d}-{:03d}".format(epoch, total_epoch)
        min_LR = min(w_scheduler.get_lr())
        logger.log(
            "\n[Search the {:}-th epoch] {:}, LR={:}".format(
                epoch_str, need_time, min_LR
            )
        )

        search_w_loss, search_w_top1, search_w_top5 = search_func_ours_F(
            search_loader,
            network,
            criterion,
            w_scheduler,
            w_optimizer,
            a_optimizer,
            epoch_str,
            xargs.print_freq,
            logger,
        )
        search_time.update(time.time() - start_time)
        logger.log(
            "[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s".format(
                epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum
            )
        )
        valid_a_loss, valid_a_top1, valid_a_top5 = valid_func(
            valid_loader, network, criterion
        )
        logger.log(
            "[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
                epoch_str, valid_a_loss, valid_a_top1, valid_a_top5
            )
        )
        # check the best accuracy
        valid_accuracies[epoch] = valid_a_top1
        if valid_a_top1 > valid_accuracies["best"]:
            valid_accuracies["best"] = valid_a_top1
            genotypes["best"] = search_model.genotype()
            find_best = True
        else:
            find_best = False

        genotypes[epoch] = search_model.genotype()
        logger.log(
            "<<<--->>> The {:}-th epoch : {:}".format(epoch_str, genotypes[epoch])
        )
        # save checkpoint
        save_path = save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": deepcopy(xargs),
                "search_model": search_model.state_dict(),
                "w_optimizer": w_optimizer.state_dict(),
                "a_optimizer": a_optimizer.state_dict(),
                "w_scheduler": w_scheduler.state_dict(),
                "genotypes": genotypes,
                "valid_accuracies": valid_accuracies,
            },
            model_base_path,
            logger,
        )
        last_info = save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": deepcopy(args),
                "last_checkpoint": save_path,
            },
            logger.path("info"),
            logger,
        )
        if find_best:
            logger.log(
                "<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.".format(
                    epoch_str, valid_a_top1
                )
            )
            copy_checkpoint(model_base_path, model_best_path, logger)
        with torch.no_grad():
            logger.log(
                "arch-parameters :\n{:}".format(
                    sparsemax(search_model.arch_parameters).cpu()
                )
            )
        if api is not None:
            logger.log("{:}".format(api.query_by_arch(genotypes[epoch], "200")))
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log("\n" + "-" * 100)
    # check the performance from the architecture dataset
    logger.log(
        "ZO-DARTS : run {:} epochs, cost {:.1f} s, last-geno is {:}.".format(
            total_epoch, search_time.sum, genotypes[total_epoch - 1]
        )
    )
    if api is not None:
        logger.log("{:}".format(api.query_by_arch(genotypes[total_epoch - 1], "200")))
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ZO-DARTS")
    parser.add_argument("--data_path", type=str, default="../../nasbench201/dataset/cifar", help="The path to dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        default="cifar10",
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    # channels and number-of-cells
    parser.add_argument("--config_path", type=str, default="../../configs/nas-benchmark/algos/DARTS.config", help="The config path.")
    parser.add_argument("--search_space_name", type=str, default="nas-bench-201", help="The search space name.")
    parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
    parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
    parser.add_argument(
        "--num_cells", type=int, default=5, help="The number of cells in one stage."
    )
    parser.add_argument(
        "--track_running_stats",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether use track_running_stats or not in the BN layer.",
    )
    # architecture leraning rate
    parser.add_argument(
        "--arch_learning_rate",
        type=float,
        default=1.5e-2,
        help="learning rate for arch encoding",
    )
    parser.add_argument(
        "--arch_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for arch encoding",
    )
    # log
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--save_dir", type=str, default="OURS-V2+2", help="Folder to save checkpoints and log."
    )
    parser.add_argument(
        "--arch_nas_dataset",
        type=str,
        help="The path to load the architecture dataset (tiny-nas-benchmark).",
    )
    parser.add_argument("--print_freq", type=int, default=200, help="print frequency (default: 200)")
    parser.add_argument("--rand_seed", type=int, default=2, help="manual seed")
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    main(args)