#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Caffe2 to PyTorch checkpoint name converting utility."""

import re
import os
import torch
import pickle
import math
import copy
import logging

import numpy as np

from collections import OrderedDict

logger = logging.getLogger(__name__)


def get_name_convert_func():
    """
    Get the function to convert Caffe2 layer names to PyTorch layer names.
    Returns:
        (func): function to convert parameter name from Caffe2 format to PyTorch
        format.
    """
    pairs = [
        # ------------------------------------------------------------
        # 'nonlocal_conv3_1_theta_w' -> 's3.pathway0_nonlocal3.conv_g.weight'
        [
            r"^nonlocal_conv([0-9]+)_([0-9]+)_(.*)",
            r"s\1.pathway0_nonlocal\2_\3",
        ],
        # 'theta' -> 'conv_theta'
        [r"^(.*)_nonlocal([0-9]+)_(theta)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'g' -> 'conv_g'
        [r"^(.*)_nonlocal([0-9]+)_(g)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'phi' -> 'conv_phi'
        [r"^(.*)_nonlocal([0-9]+)_(phi)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'out' -> 'conv_out'
        [r"^(.*)_nonlocal([0-9]+)_(out)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'nonlocal_conv4_5_bn_s' -> 's4.pathway0_nonlocal3.bn.weight'
        [r"^(.*)_nonlocal([0-9]+)_(bn)_(.*)", r"\1_nonlocal\2.\3.\4"],
        # ------------------------------------------------------------
        # 't_pool1_subsample_bn' -> 's1_fuse.conv_f2s.bn.running_mean'
        [r"^t_pool1_subsample_bn_(.*)", r"s1_fuse.bn.\1"],
        # 't_pool1_subsample' -> 's1_fuse.conv_f2s'
        [r"^t_pool1_subsample_(.*)", r"s1_fuse.conv_f2s.\1"],
        # 't_res4_5_branch2c_bn_subsample_bn_rm' -> 's4_fuse.conv_f2s.bias'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_bn_(.*)",
            r"s\1_fuse.bn.\3",
        ],
        # 't_pool1_subsample' -> 's1_fuse.conv_f2s'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_(.*)",
            r"s\1_fuse.conv_f2s.\3",
        ],
        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b'
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway0_res\2.branch\3.\4_\5",
        ],
        # 'res_conv1_bn_' -> 's1.pathway0_stem.bn.'
        [r"^res_conv1_bn_(.*)", r"s1.pathway0_stem.bn.\1"],
        # 'conv1_xy_w_momentum' -> 's1.pathway0_stem.conv_xy.'
        [r"^conv1_xy(.*)", r"s1.pathway0_stem.conv_xy\1"],
        # 'conv1_w_momentum' -> 's1.pathway0_stem.conv.'
        [r"^conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
        # 'res4_0_branch1_w' -> 'S4.pathway0_res0.branch1.weight'
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway0_res\2.branch\3_\4",
        ],
        # 'res_conv1_' -> 's1.pathway0_stem.conv.'
        [r"^res_conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway1_res\2.branch\3.\4_\5",
        ],
        # 'res_conv1_bn_' -> 's1.pathway0_stem.bn.'
        [r"^t_res_conv1_bn_(.*)", r"s1.pathway1_stem.bn.\1"],
        # 'conv1_w_momentum' -> 's1.pathway0_stem.conv.'
        [r"^t_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
        # 'res4_0_branch1_w' -> 'S4.pathway0_res0.branch1.weight'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway1_res\2.branch\3_\4",
        ],
        # 'res_conv1_' -> 's1.pathway0_stem.conv.'
        [r"^t_res_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
        # ------------------------------------------------------------
        # pred_ -> head.projection.
        [r"pred_(.*)", r"head.projection.\1"],
        # '.b_bn_fc' -> '.se.fc'
        [r"(.*)b_bn_fc(.*)", r"\1se.fc\2"],
        # conv_5 -> head.conv_5.
        [r"conv_5(.*)", r"head.conv_5\1"],
        # conv_5 -> head.conv_5.
        [r"lin_5(.*)", r"head.lin_5\1"],
        # '.bn_b' -> '.weight'
        [r"(.*)bn.b\Z", r"\1bn.bias"],
        # '.bn_s' -> '.weight'
        [r"(.*)bn.s\Z", r"\1bn.weight"],
        # '_bn_rm' -> '.running_mean'
        [r"(.*)bn.rm\Z", r"\1bn.running_mean"],
        # '_bn_riv' -> '.running_var'
        [r"(.*)bn.riv\Z", r"\1bn.running_var"],
        # '_b' -> '.bias'
        [r"(.*)[\._]b\Z", r"\1.bias"],
        # '_w' -> '.weight'
        [r"(.*)[\._]w\Z", r"\1.weight"],
    ]

    def convert_caffe2_name_to_pytorch(caffe2_layer_name):
        """
        Convert the caffe2_layer_name to pytorch format by apply the list of
        regular expressions.
        Args:
            caffe2_layer_name (str): caffe2 layer name.
        Returns:
            (str): pytorch layer name.
        """
        for source, dest in pairs:
            caffe2_layer_name = re.sub(source, dest, caffe2_layer_name)
        return caffe2_layer_name

    return convert_caffe2_name_to_pytorch


def get_checkpoint_dir(path_to_job):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints")


def get_path_to_checkpoint(path_to_job, epoch, task=""):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    if task != "":
        name = "{}_checkpoint_epoch_{:05d}.pyth".format(task, epoch)
    else:
        name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job), name)


def is_checkpoint_epoch(cfg, cur_epoch, multigrid_schedule=None):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cfg (CfgNode): configs to save.
        cur_epoch (int): current number of epoch of the model.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True
    if multigrid_schedule is not None:
        prev_epoch = 0
        for s in multigrid_schedule:
            if cur_epoch < s[-1]:
                period = max((s[-1] - prev_epoch) // cfg.MULTIGRID.EVAL_FREQ + 1, 1)
                return (s[-1] - 1 - cur_epoch) % period == 0
            prev_epoch = s[-1]

    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0


def load_checkpoint(
        path_to_checkpoint,
        model,
        data_parallel=True,
        optimizer=None,
        scaler=None,
        inflation=False,
        convert_from_caffe2=False,
        epoch_reset=False,
        clear_name_pattern=(),
        image_init=False,
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        scaler (GradScaler): GradScaler to load the mixed precision scale.
        inflation (bool): if True, inflate the weights from the checkpoint.
        convert_from_caffe2 (bool): if True, load the model from caffe2 and
            convert it to pytorch.
        epoch_reset (bool): if True, reset #train iterations from the checkpoint.
        clear_name_pattern (string): if given, this (sub)string will be cleared
            from a layer name if it can be matched.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """

    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if data_parallel else model
    if convert_from_caffe2:
        with open(path_to_checkpoint, "rb") as f:
            caffe2_checkpoint = pickle.load(f, encoding="latin1")
        state_dict = OrderedDict()
        name_convert_func = get_name_convert_func()
        for key in caffe2_checkpoint["blobs"].keys():
            converted_key = name_convert_func(key)
            converted_key = c2_normal_to_sub_bn(converted_key, ms.state_dict())
            if converted_key in ms.state_dict():
                c2_blob_shape = caffe2_checkpoint["blobs"][key].shape
                model_blob_shape = ms.state_dict()[converted_key].shape

                # expand shape dims if they differ (eg for converting linear to conv params)
                if len(c2_blob_shape) < len(model_blob_shape):
                    c2_blob_shape += (1,) * (len(model_blob_shape) - len(c2_blob_shape))
                    caffe2_checkpoint["blobs"][key] = np.reshape(
                        caffe2_checkpoint["blobs"][key], c2_blob_shape
                    )
                # Load BN stats to Sub-BN.
                if (
                        len(model_blob_shape) == 1
                        and len(c2_blob_shape) == 1
                        and model_blob_shape[0] > c2_blob_shape[0]
                        and model_blob_shape[0] % c2_blob_shape[0] == 0
                ):
                    caffe2_checkpoint["blobs"][key] = np.concatenate(
                        [caffe2_checkpoint["blobs"][key]]
                        * (model_blob_shape[0] // c2_blob_shape[0])
                    )
                    c2_blob_shape = caffe2_checkpoint["blobs"][key].shape

                if c2_blob_shape == tuple(model_blob_shape):
                    state_dict[converted_key] = torch.tensor(
                        caffe2_checkpoint["blobs"][key]
                    ).clone()
                    logger.debug(
                        "{}: {} => {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
                else:
                    logger.debug(
                        "!! {}: {} does not match {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
            else:
                if not any(
                        prefix in key for prefix in ["momentum", "lr", "model_iter"]
                ):
                    logger.warning(
                        "!! {}: can not be converted, got {}".format(key, converted_key)
                    )
        diff = set(ms.state_dict()) - set(state_dict)
        diff = {d for d in diff if "num_batches_tracked" not in d}
        if len(diff) > 0:
            logger.warning("Not loaded {}".format(diff))
        ms.load_state_dict(state_dict, strict=False)
        epoch = -1
    else:
        # Load the checkpoint on CPU to avoid GPU mem spike.
        with open(path_to_checkpoint, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        model_state_dict_3d = (
            model.module.state_dict() if data_parallel else model.state_dict()
        )
        checkpoint["model_state"] = normal_to_sub_bn(
            checkpoint["model_state"], model_state_dict_3d
        )

        if clear_name_pattern:
            for item in clear_name_pattern:
                model_state_dict_new = OrderedDict()
                for k in checkpoint["model_state"]:
                    if item in k:
                        k_re = k.replace(
                            item, "", 1
                        )  # only repace first occurence of pattern
                        model_state_dict_new[k_re] = checkpoint["model_state"][k]
                        logger.debug("renaming: {} -> {}".format(k, k_re))
                    else:
                        model_state_dict_new[k] = checkpoint["model_state"][k]
                checkpoint["model_state"] = model_state_dict_new

            pre_train_dict = checkpoint["model_state"]
            model_dict = ms.state_dict()

            if image_init:
                if (
                        "pos_embed" in pre_train_dict.keys()
                        and "pos_embed_xy" in model_dict.keys()
                ):
                    logger.debug(
                        pre_train_dict["pos_embed"].shape,
                        model_dict["pos_embed_xy"].shape,
                        model_dict["pos_embed_class"].shape,
                    )
                    if (
                            pre_train_dict["pos_embed"].shape[1]
                            == model_dict["pos_embed_xy"].shape[1] + 1
                    ):
                        pre_train_dict["pos_embed_xy"] = pre_train_dict["pos_embed"][
                                                         :, 1:
                                                         ]
                        pre_train_dict["pos_embed_class"] = pre_train_dict["pos_embed"][
                                                            :, :1
                                                            ]

                if (
                        "patch_embed.proj.weight" in pre_train_dict.keys()
                        and "patch_embed.proj.weight" in model_dict.keys()
                ):
                    logger.debug(
                        pre_train_dict["patch_embed.proj.weight"].shape,
                        model_dict["patch_embed.proj.weight"].shape,
                    )
                    if (
                            len(pre_train_dict["patch_embed.proj.weight"].shape) == 4
                            and len(model_dict["patch_embed.proj.weight"].shape) == 5
                    ):  # img->video
                        t = model_dict["patch_embed.proj.weight"].shape[2]
                        pre_train_dict["patch_embed.proj.weight"] = pre_train_dict[
                                                                        "patch_embed.proj.weight"
                                                                    ][:, :, None, :, :].repeat(1, 1, t, 1, 1)
                        logger.debug(
                            f"inflate patch_embed.proj.weight to {pre_train_dict['patch_embed.proj.weight'].shape}"
                        )
                    elif (
                            len(pre_train_dict["patch_embed.proj.weight"].shape) == 5
                            and len(model_dict["patch_embed.proj.weight"].shape) == 4
                    ):  # video->img
                        orig_shape = pre_train_dict["patch_embed.proj.weight"].shape
                        # pre_train_dict["patch_embed.proj.weight"] = pre_train_dict["patch_embed.proj.weight"][:, :, orig_shape[2]//2, :, :] # take center
                        pre_train_dict["patch_embed.proj.weight"] = pre_train_dict[
                            "patch_embed.proj.weight"
                        ].sum(2)  # take avg
                        logger.debug(
                            f"deflate patch_embed.proj.weight from {orig_shape} to {pre_train_dict['patch_embed.proj.weight'].shape}"
                        )
                        if (
                                "pos_embed_spatial" in pre_train_dict.keys()
                                and "pos_embed" in model_dict.keys()
                        ):
                            pos_embds = pre_train_dict["pos_embed_spatial"]
                            if (
                                    "pos_embed_class" in pre_train_dict.keys()
                                    and pos_embds.shape != model_dict["pos_embed"].shape
                            ):
                                pos_embds = torch.cat(
                                    [
                                        pre_train_dict["pos_embed_class"],
                                        pos_embds,
                                    ],
                                    1,
                                )
                                pre_train_dict.pop("pos_embed_class")
                            if pos_embds.shape == model_dict["pos_embed"].shape:
                                pre_train_dict["pos_embed"] = pos_embds
                                pre_train_dict.pop("pos_embed_spatial")
                                logger.info(
                                    f"successful surgery of pos embed w/ shape {pos_embds.shape} "
                                )
                            else:
                                logger.warning(
                                    f"UNSUCCESSFUL surgery of pos embed w/ shape {pos_embds.shape} "
                                )

                qkv = [
                    "attn.pool_k.weight",
                    "attn.pool_q.weight",
                    "attn.pool_v.weight",
                ]
                for k in pre_train_dict.keys():
                    if (
                            any([x in k for x in qkv])
                            and pre_train_dict[k].shape != model_dict[k].shape
                    ):
                        # print(pre_train_dict[k].shape, model_dict[k].shape)
                        logger.debug(
                            f"inflate {k} from {pre_train_dict[k].shape} to {model_dict[k].shape}"
                        )
                        t = model_dict[k].shape[2]
                        pre_train_dict[k] = pre_train_dict[k].repeat(1, 1, t, 1, 1)

                for k in pre_train_dict.keys():
                    if (
                            "rel_pos" in k
                            and pre_train_dict[k].shape != model_dict[k].shape
                    ):
                        # print(pre_train_dict[k].shape, model_dict[k].shape)
                        logger.debug(
                            f"interpolating {k} from {pre_train_dict[k].shape} to {model_dict[k].shape}"
                        )
                        new_pos_embed = torch.nn.functional.interpolate(
                            pre_train_dict[k]
                            .reshape(1, pre_train_dict[k].shape[0], -1)
                            .permute(0, 2, 1),
                            size=model_dict[k].shape[0],
                            mode="linear",
                        )
                        new_pos_embed = (
                            new_pos_embed.reshape(-1, model_dict[k].shape[0])
                            .permute(1, 0)
                            .squeeze()
                        )
                        pre_train_dict[k] = new_pos_embed

            # Match pre-trained weights that have same shape as current model.
            pre_train_dict_match = {}
            not_used_layers = []
            for k, v in pre_train_dict.items():
                if k in model_dict:
                    if v.size() == model_dict[k].size():
                        pre_train_dict_match[k] = v
                    else:
                        if "attn.rel_pos" in k:
                            v_shape = v.shape
                            v = v.t().unsqueeze(0)
                            v = torch.nn.functional.interpolate(
                                v,
                                size=model_dict[k].size()[0],
                                mode="linear",
                            )
                            v = v[0].t()
                            pre_train_dict_match[k] = v
                            logger.debug(
                                "{} reshaped from {} to {}".format(k, v_shape, v.shape)
                            )
                        elif "pos_embed_temporal" in k:
                            v_shape = v.shape
                            v = torch.nn.functional.interpolate(
                                v.permute(0, 2, 1),
                                size=model_dict[k].shape[1],
                                mode="linear",
                            )
                            pre_train_dict_match[k] = v.permute(0, 2, 1)
                            logger.debug(
                                "{} reshaped from {} to {}".format(
                                    k, v_shape, pre_train_dict_match[k].shape
                                )
                            )
                        elif "pos_embed_spatial" in k:
                            v_shape = v.shape
                            pretrain_size = int(math.sqrt(v_shape[1]))
                            model_size = int(math.sqrt(model_dict[k].shape[1]))
                            assert pretrain_size * pretrain_size == v_shape[1]
                            assert model_size * model_size == model_dict[k].shape[1]
                            v = torch.nn.functional.interpolate(
                                v.reshape(1, pretrain_size, pretrain_size, -1).permute(
                                    0, 3, 1, 2
                                ),
                                size=(model_size, model_size),
                                mode="bicubic",
                            )
                            pre_train_dict_match[k] = v.reshape(
                                1, -1, model_size * model_size
                            ).permute(0, 2, 1)
                            logger.debug(
                                "{} reshaped from {} to {}".format(
                                    k, v_shape, pre_train_dict_match[k].shape
                                )
                            )
                        else:
                            not_used_layers.append(k)
                else:
                    not_used_layers.append(k)
            # Weights that do not have match from the pre-trained model.
            not_load_layers = [
                k for k in model_dict.keys() if k not in pre_train_dict_match.keys()
            ]
            # Log weights that are not loaded with the pre-trained weights.
            if not_load_layers:
                for k in not_load_layers:
                    logger.debug("Network weights {} not loaded.".format(k))
            if not_used_layers:
                for k in not_used_layers:
                    logger.debug("Network weights {} not used.".format(k))
            # Load pre-trained weights.
            missing_keys, unexpected_keys = ms.load_state_dict(
                pre_train_dict_match, strict=False
            )

            logger.warning("missing keys: {}".format(missing_keys))
            logger.warning("unexpected keys: {}".format(unexpected_keys))
            epoch = -1

            # Load the optimizer state (commonly not done when fine-tuning)
        if "epoch" in checkpoint.keys() and not epoch_reset:
            epoch = checkpoint["epoch"]
            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scaler:
                scaler.load_state_dict(checkpoint["scaler_state"])
        else:
            epoch = -1
    return epoch

def sub_to_normal_bn(sd):
    """
    Convert the Sub-BN paprameters to normal BN parameters in a state dict.
    There are two copies of BN layers in a Sub-BN implementation: `bn.bn` and
    `bn.split_bn`. `bn.split_bn` is used during training and
    "compute_precise_bn". Before saving or evaluation, its stats are copied to
    `bn.bn`. We rename `bn.bn` to `bn` and store it to be consistent with normal
    BN layers.
    Args:
        sd (OrderedDict): a dict of parameters whitch might contain Sub-BN
        parameters.
    Returns:
        new_sd (OrderedDict): a dict with Sub-BN parameters reshaped to
        normal parameters.
    """
    new_sd = copy.deepcopy(sd)
    modifications = [
        ("bn.bn.running_mean", "bn.running_mean"),
        ("bn.bn.running_var", "bn.running_var"),
        ("bn.split_bn.num_batches_tracked", "bn.num_batches_tracked"),
    ]
    to_remove = ["bn.bn.", ".split_bn."]
    for key in sd:
        for before, after in modifications:
            if key.endswith(before):
                new_key = key.split(before)[0] + after
                new_sd[new_key] = new_sd.pop(key)

        for rm in to_remove:
            if rm in key and key in new_sd:
                del new_sd[key]

    for key in new_sd:
        if key.endswith("bn.weight") or key.endswith("bn.bias"):
            if len(new_sd[key].size()) == 4:
                assert all(d == 1 for d in new_sd[key].size()[1:])
                new_sd[key] = new_sd[key][:, 0, 0, 0]

    return new_sd


def c2_normal_to_sub_bn(key, model_keys):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        key (OrderedDict): source dict of parameters.
        mdoel_key (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    if "bn.running_" in key:
        if key in model_keys:
            return key

        new_key = key.replace("bn.running_", "bn.split_bn.running_")
        if new_key in model_keys:
            return new_key
    else:
        return key


def normal_to_sub_bn(checkpoint_sd, model_sd):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        checkpoint_sd (OrderedDict): source dict of parameters.
        model_sd (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    for key in model_sd:
        if key not in checkpoint_sd:
            if "bn.split_bn." in key:
                load_key = key.replace("bn.split_bn.", "bn.")
                bn_key = key.replace("bn.split_bn.", "bn.bn.")
                checkpoint_sd[key] = checkpoint_sd.pop(load_key)
                checkpoint_sd[bn_key] = checkpoint_sd[key]

    for key in model_sd:
        if key in checkpoint_sd:
            model_blob_shape = model_sd[key].shape
            c2_blob_shape = checkpoint_sd[key].shape

            if (
                    len(model_blob_shape) == 1
                    and len(c2_blob_shape) == 1
                    and model_blob_shape[0] > c2_blob_shape[0]
                    and model_blob_shape[0] % c2_blob_shape[0] == 0
            ):
                before_shape = checkpoint_sd[key].shape
                checkpoint_sd[key] = torch.cat(
                    [checkpoint_sd[key]] * (model_blob_shape[0] // c2_blob_shape[0])
                )
                logger.debug(
                    "{} {} -> {}".format(key, before_shape, checkpoint_sd[key].shape)
                )
    return checkpoint_sd
