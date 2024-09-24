
import sys
import logging
import os.path as osp
from importlib import import_module
import numpy as np
import torch
from attrdict import AttrDict as adict

from .mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from .mobilenetv2 import mobilenetv2
from .losses import (AMSoftmaxLoss, AngleSimpleLinear, SoftTripleLinear,
                    SoftTripleLoss)

def load_checkpoint(checkpoint_path, net, map_location, optimizer=None, load_optimizer=False, strict=True):
    ''' load a checkpoint of the given model. If model is using for training with imagenet weights provided by
        this project, then delete some wights due to mismatching architectures'''
    print("\n==> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        unloaded = net.load_state_dict(checkpoint['state_dict'], strict=strict)
        missing_keys, unexpected_keys = (', '.join(i) for i in unloaded)
    else:
        unloaded = net.load_state_dict(checkpoint, strict=strict)
        missing_keys, unexpected_keys = (', '.join(i) for i in unloaded)
    if missing_keys or unexpected_keys:
        logging.warning(f'THE FOLLOWING KEYS HAVE NOT BEEN LOADED:\n\nmissing keys: {missing_keys}\
            \n\nunexpected keys: {unexpected_keys}\n')
        print('proceed traning ...')
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'epoch' in checkpoint:
        return checkpoint['epoch']

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))
    
def read_py_config(filename):
    filename = osp.abspath(osp.expanduser(filename))
    check_file_exist(filename)
    assert filename.endswith('.py')
    module_name = osp.basename(filename)[:-3]
    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')
    config_dir = osp.dirname(filename)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = adict({
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    })

    return cfg_dict


def build_model(config, device, strict=True, mode='train'):
    ''' build model and change layers depends on loss type'''
    parameters = dict(width_mult=config.model.width_mult,
                    prob_dropout=config.dropout.prob_dropout,
                    type_dropout=config.dropout.type,
                    mu=config.dropout.mu,
                    sigma=config.dropout.sigma,
                    embeding_dim=config.model.embeding_dim,
                    prob_dropout_linear = config.dropout.classifier,
                    theta=config.conv_cd.theta,
                    multi_heads = config.multi_task_learning)

    if config.model.model_type == 'Mobilenet2':
        model = mobilenetv2(**parameters)

        if config.model.pretrained and mode == "train":
            checkpoint_path = config.model.imagenet_weights
            load_checkpoint(checkpoint_path, model, strict=strict, map_location=device)
        elif mode == 'convert':
            model.forward = model.forward_to_onnx

        if (config.loss.loss_type == 'amsoftmax') and (config.loss.amsoftmax.margin_type != 'cross_entropy'):
            model.spoofer = AngleSimpleLinear(config.model.embeding_dim, 2)
        elif config.loss.loss_type == 'soft_triple':
            model.spoofer = SoftTripleLinear(config.model.embeding_dim, 2,
                                             num_proxies=config.loss.soft_triple.K)
    else:
        assert config.model.model_type == 'Mobilenet3'
        if config.model.model_size == 'large':
            model = mobilenetv3_large(**parameters)

            if config.model.pretrained and mode == "train":
                checkpoint_path = config.model.imagenet_weights
                load_checkpoint(checkpoint_path, model, strict=strict, map_location=device)
            elif mode == 'convert':
                model.forward = model.forward_to_onnx
        else:
            assert config.model.model_size == 'small'
            model = mobilenetv3_small(**parameters)

            if config.model.pretrained and mode == "train":
                checkpoint_path = config.model.imagenet_weights
                load_checkpoint(checkpoint_path, model, strict=strict, map_location=device)
            elif mode == 'convert':
                model.forward = model.forward_to_onnx

        if (config.loss.loss_type == 'amsoftmax') and (config.loss.amsoftmax.margin_type != 'cross_entropy'):
            model.scaling = config.loss.amsoftmax.s
            model.spoofer[3] = AngleSimpleLinear(config.model.embeding_dim, 2)
        elif config.loss.loss_type == 'soft_triple':
            model.scaling = config.loss.soft_triple.s
            model.spoofer[3] = SoftTripleLinear(config.model.embeding_dim, 2, num_proxies=config.loss.soft_triple.K)
    return model