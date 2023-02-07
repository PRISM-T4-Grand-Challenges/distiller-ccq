#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This is an example application for compressing image classification models.

The application borrows its main flow code from torchvision's ImageNet classification
training sample application (https://github.com/pytorch/examples/tree/master/imagenet).
We tried to keep it similar, in order to make it familiar and easy to understand.

Integrating compression is very simple: simply add invocations of the appropriate
compression_scheduler callbacks, for each stage in the training.  The training skeleton
looks like the pseudo code below.  The boiler-plate Pytorch classification training
is speckled with invocations of CompressionScheduler.

For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    save_checkpoint()
    compression_scheduler.on_epoch_end(epoch)

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)


This exmple application can be used with torchvision's ImageNet image classification
models, or with the provided sample models:

- ResNet for CIFAR: https://github.com/junyuseu/pytorch-cifar-models
- MobileNet for ImageNet: https://github.com/marvis/pytorch-mobilenet
"""

import math
import argparse
import time
import os
import sys
import random
import traceback
import logging
from collections import OrderedDict, defaultdict
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
from tabulate import tabulate
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
try:
    import distiller
except ImportError:
    sys.path.append(module_path)
    import distiller
import apputils
from distiller.data_loggers import *
import distiller.quantization as quantization
from models import ALL_MODEL_NAMES, create_model

import yaml
import yamlordereddictloader
import tabulate

# quantizer = 'pact_quantizer'
secondary = "./dump_space/tensor.yaml"

curr_bit_lib = {}
next_bit_lib = {}

res18_weights = {'conv1':9408,'layer1.0.conv1':36864,'layer1.0.conv2':36864,'layer1.1.conv1':36864,'layer1.1.conv2':36864,'layer2.0.conv1':73728,'layer2.0.conv2':147456,'layer2.0.downsample.0':8192, \
                'layer2.1.conv1':147456,'layer2.1.conv2':147456,'layer3.0.conv1':294912,'layer3.0.conv2':589824,'layer3.0.downsample.0':32768,'layer3.1.conv1':589824,'layer3.1.conv2':589824, \
                'layer4.0.conv1':1179648,'layer4.0.conv2':2359296,'layer4.0.downsample.0':131072,'layer4.1.conv1':2359296,'layer4.1.conv2':2359296,'fc':512000}
                
weight_list = np.asarray(list(res18_weights.values()))

bitmap = [8, 6, 4, 3, 2]

def dump_yaml(OrDict,name,mode):
    yaml.dump(
            OrDict,
            open(name, mode),
            Dumper=yamlordereddictloader.Dumper,
            default_flow_style=False)

def get_layers(sched_dict,quantizer):
    temp_hold = list(sched_dict['quantizers'][quantizer]['bits_overrides'].keys())
    layers = []
    for item in temp_hold:
        if 'relu' not in item:
            layers.append(item)
    return layers
    
def get_pact_headers(model):
    headers = []
    for param_tensor in model.state_dict():
        if 'clip_val' in param_tensor:
            headers.append(param_tensor)
    return headers
           

def write_layer_name(data,sched_dict):
    row = []
    with open("./dump_space/layer_to_quantize.csv", "a") as csvfile:
        writer = csv.writer(csvfile)
        row.append(data)
        writer.writerow(row)

def update_pact_alpha(pact_dict,model,headers):
    for param_tensor in headers:
        t = []
        a = float(pact_dict[param_tensor])
        t.append(a)
        model.state_dict()[param_tensor].data.copy_(torch.Tensor(t).cuda())        

def log_pact(pact_dict):
    data = list(pact_dict.items())
    t = tabulate.tabulate(data, headers=["Layers", "PACT_alpha"], tablefmt="psql") 
    msglogger.info('\n')
    msglogger.info(t)

# Logger handle
msglogger = None

##kaap_Jhaap begin
import csv

val_loss_file = "./dump_space/val_loss.csv"
train_loss_file = "./dump_space/train_loss.csv"

val_loss_track = {}
train_loss_track = {}


def write_loss(file,dict):
    with open(file, "a") as csvfile:
        writer = csv.writer(csvfile)
        key_list = list(dict.keys())
        num = len(key_list);
        for i in range(num):
            row = []
            name = str(key_list[i])
            val = dict[key_list[i]]
            row.append(name)
            row.append(val)
            writer.writerow(row)            
##kaap_Jhaap end
    

def float_range(val_str):
    val = float(val_str)
    if val < 0 or val >= 1:
        raise argparse.ArgumentTypeError('Must be >= 0 and < 1 (received {0})'.format(val_str))
    return val


def get_valid_acc(key_val):

    pos1 = key_val.find(',')
    new_val = key_val[pos1+1:]    

    pos2 = new_val.find(',')
    new_val = new_val[:pos2]
    
    return float(new_val)
    
def get_weight_volume(layers, sched_dict, quantizer, fp=False):
    
    weight_volume = 0
    
    for item in layers:
        if fp:
            bits = 32
        else:
            bits = sched_dict['quantizers'][quantizer]['bits_overrides'][item]['wts']
        
        weight_volume += bits*res18_weights[item]
    
    weight_volume /= (1024*1024*8)
    
    return weight_volume

parser = argparse.ArgumentParser(description='Distiller image classification model compression')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=ALL_MODEL_NAMES,
                    help='model architecture: ' +
                    ' | '.join(ALL_MODEL_NAMES) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--qsteps', default=None, type=int,
                    help='number of total quantization steps')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--act-stats', dest='activation_stats', choices=["train", "valid", "test"], default=None,
                    help='collect activation statistics (WARNING: this slows down training)')
parser.add_argument('--masks-sparsity', dest='masks_sparsity', action='store_true', default=False,
                    help='print masks sparsity table at end of each epoch')
parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                    help='log the paramter tensors histograms to file (WARNING: this can use significant disk space)')
SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params', 'onnx']
parser.add_argument('--summary', type=str, choices=SUMMARY_CHOICES,
                    help='print a summary of the model, and exit - options: ' +
                    ' | '.join(SUMMARY_CHOICES))
parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                    help='configuration file for pruning the model (default is to use hard-coded schedule)')
parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
                    help='test the sensitivity of layers to pruning')
parser.add_argument('--sense-range', dest='sensitivity_range', type=float, nargs=3, default=[0.0, 0.95, 0.05],
                    help='an optional paramaeter for sensitivity testing providing the range of sparsities to test.\n'
                    'This is equaivalent to creating sensitivities = np.arange(start, stop, step)')
parser.add_argument('--extras', default=None, type=str,
                    help='file with extra configuration information')
parser.add_argument('--deterministic', '--det', action='store_true',
                    help='Ensure deterministic execution for re-producible results.')
parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                    help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
parser.add_argument('--validation-size', '--vs', type=float_range, default=0.1,
                    help='Portion of training dataset to set aside for validation')
parser.add_argument('--adc', dest='ADC', action='store_true', help='temp HACK')
parser.add_argument('--adc-params', dest='ADC_params', default=None, help='temp HACK')
parser.add_argument('--confusion', dest='display_confusion', default=False, action='store_true',
                    help='Display the confusion matrix')
parser.add_argument('--earlyexit_lossweights', type=float, nargs='*', dest='earlyexit_lossweights', default=None,
                    help='List of loss weights for early exits (e.g. --lossweights 0.1 0.3)')
parser.add_argument('--earlyexit_thresholds', type=float, nargs='*', dest='earlyexit_thresholds', default=None,
                    help='List of EarlyExit thresholds (e.g. --earlyexit 1.2 0.9)')
parser.add_argument('--num-best-scores', dest='num_best_scores', default=1, type=int,
                    help='number of best scores to track and report (default: 1)')
parser.add_argument('--load-serialized', dest='load_serialized', action='store_true', default=False,
                    help='Load a model without DataParallel wrapping it')

quant_group = parser.add_argument_group('Arguments controlling quantization at evaluation time'
                                        '("post-training quantization)')
quant_group.add_argument('--quantize-eval', '--qe', action='store_true',
                         help='Apply linear-symmetric quantization to model before evaluation. Applicable only if'
                              '--evaluate is also set')
quant_group.add_argument('--qe-bits-acts', '--qeba', type=int, default=8, metavar='NUM_BITS',
                         help='Number of bits for quantization of activations')
quant_group.add_argument('--qe-bits-wts', '--qebw', type=int, default=8, metavar='NUM_BITS',
                         help='Number of bits for quantization of weights')
quant_group.add_argument('--qe-bits-accum', type=int, default=32, metavar='NUM_BITS',
                         help='Number of bits for quantization of the accumulator')
quant_group.add_argument('--qe-clip-acts', '--qeca', action='store_true',
                         help='Enable clipping of activations using max-abs-value averaging over batch')
quant_group.add_argument('--qe-no-clip-layers', '--qencl', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                         help='List of fully-qualified layer names for which not to clip activations. Applicable'
                              'only if --qe-clip-acts is also set')

distiller.knowledge_distillation.add_distillation_args(parser, ALL_MODEL_NAMES, True)

QE_dict = {}

def print_model(model):
    print("Model's state_dict:")
    #l = model.state_dict().keys()
    #print(l)
    for param_tensor in model.state_dict():
        if 'classifier.1' in param_tensor:
            print(param_tensor, "\t", model.state_dict()[param_tensor])

            
def print_qe(model):
    global QE_dict
    for param_tensor in model.state_dict():
        if 'float_weight' in param_tensor:
            fp = param_tensor
            quant = fp.replace('float_weight','weight')
            # add a if to compare two sizes
            n = float(torch.numel(model.state_dict()[fp]))
            quantization_error = torch.sum(torch.abs(model.state_dict()[fp]-model.state_dict()[quant]))/n
            QE_dict[quant]= quantization_error
    data = [(k,v) for k,v in QE_dict.items()]
    print (tabulate(data, headers=["Layers", "Q_Error"], tablefmt="psql", floatfmt=".7f"))
    
            
def check_pytorch_version():
    if torch.__version__ < '0.4.0':
        print("\nNOTICE:")
        print("The Distiller \'master\' branch now requires at least PyTorch version 0.4.0 due to "
              "PyTorch API changes which are not backward-compatible.\n"
              "Please install PyTorch 0.4.0 or its derivative.\n"
              "If you are using a virtual environment, do not forget to update it:\n"
              "  1. Deactivate the old environment\n"
              "  2. Install the new environment\n"
              "  3. Activate the new environment")
        exit(1)


def create_activation_stats_collectors(model, collection_phase):
    """Create objects that collect activation statistics.

    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels

    Args:
        model - the model on which we want to collect statistics
        phase - the statistics collection phase which is either "train" (for training),
                or "valid" (for validation)

    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    class missingdict(dict):
        """This is a little trick to prevent KeyError"""
        def __missing__(self, key):
            return None  # note, does *not* set self[key] - we don't want defaultdict's behavior

    distiller.utils.assign_layer_fq_names(model)

    activations_collectors = {"train": missingdict(), "valid": missingdict(), "test": missingdict()}
    if collection_phase is None:
        return activations_collectors
    collectors = missingdict({
        "sparsity":      SummaryActivationStatsCollector(model, "sparsity",
                                                         lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])})
    activations_collectors[collection_phase] = collectors
    return activations_collectors


def save_collectors_data(collectors, directory):
    """Utility function that saves all activation statistics to Excel workbooks
    """
    for name, collector in collectors.items():
        workbook = os.path.join(directory, name)
        msglogger.info("Generating {}".format(workbook))
        collector.to_xlsx(workbook)


import code, traceback, signal

def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)

def listen():
    signal.signal(signal.SIGUSR1, debug)  # Register handler        
        
def main():
    # listen()
    global msglogger
    global val_loss_track
    global train_loss_track
    global val_loss_file
    global train_loss_file
    global secondary
    global res18_weights
    global weight_list
    global bitmap
    check_pytorch_version()
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(sys.argv, gitroot=module_path)
    msglogger.debug("Distiller: %s", distiller.__version__)

    start_epoch = 0
    best_epochs = [distiller.MutableNamedTuple({'epoch': 0, 'top1': 0, 'sparsity': 0})
                   for i in range(args.num_best_scores)]

    if args.deterministic:
        # Experiment reproducibility is sometimes important.  Pete Warden expounded about this
        # in his blog: https://petewarden.com/2018/03/19/the-machine-learning-reproducibility-crisis/
        # In Pytorch, support for deterministic execution is still a bit clunky.
        if args.workers > 1:
            msglogger.error('ERROR: Setting --deterministic requires setting --workers/-j to 0 or 1')
            exit(1)
        # Use a well-known seed, for repeatability of experiments
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        cudnn.deterministic = True
    else:
        # This issue: https://github.com/pytorch/pytorch/issues/3659
        # Implies that cudnn.benchmark should respect cudnn.deterministic, but empirically we see that
        # results are not re-produced when benchmark is set. So enabling only if deterministic mode disabled.
        cudnn.benchmark = True

    if args.gpus is not None:
        try:
            args.gpus = [int(s) for s in args.gpus.split(',')]
        except ValueError:
            msglogger.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
            exit(1)
        available_gpus = torch.cuda.device_count()
        for dev_id in args.gpus:
            if dev_id >= available_gpus:
                msglogger.error('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                .format(dev_id, available_gpus))
                exit(1)
        # Set default device in case the first one on the list != 0
        torch.cuda.set_device(args.gpus[0])

    if 'cifar100' in args.arch:
        args.dataset = 'cifar100'
    elif 'cifar' in args.arch:
        args.dataset = 'cifar10'
    else:
        args.dataset = 'imagenet'
    
    if args.dataset == 'cifar100':
        args.num_classes = 100 
    elif args.dataset == 'cifar10':
        args.num_classes = 10
    else:
        args.num_classes = 1000
    ##edited by ffk##
        
    if args.earlyexit_thresholds:
        args.num_exits = len(args.earlyexit_thresholds) + 1
        args.loss_exits = [0] * args.num_exits
        args.losses_exits = []
        args.exiterrors = []

    # Create the model
    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)

    # capture thresholds for early-exit training
    if args.earlyexit_thresholds:
        msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)

    ##eikhane args resume
    # We can optionally resume from a checkpoint
    if args.resume:
        model, compression_scheduler, start_epoch = apputils.load_checkpoint(
            model, chkpt_file=args.resume)
    
    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    msglogger.info('Optimizer Type: %s', type(optimizer))
    msglogger.info('Optimizer Args: %s', optimizer.defaults)
    
    if args.ADC:
        return automated_deep_compression(model, criterion, pylogger, args)

    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        return summarize_model(model, args.dataset, which_summary=args.summary)

    # Load the datasets: the dataset to load is inferred from the model name passed
    # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
    # substring "_cifar", then cifar10 is used.
    if('imagenet' in args.dataset):
        train_loader, val_loader, test_loader, bandit_loader, _ = apputils.load_data(
            args.dataset, os.path.expanduser(args.data), args.batch_size,
            args.workers, args.validation_size, args.deterministic)
        msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d\n\tbandit=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler), len(bandit_loader.sampler))
    else:
        train_loader, val_loader, test_loader, _ = apputils.load_data(
            args.dataset, os.path.expanduser(args.data), args.batch_size,
            args.workers, args.validation_size, args.deterministic)
        msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    activations_collectors = create_activation_stats_collectors(model, collection_phase=args.activation_stats)

    if args.sensitivity is not None:
        sensitivities = np.arange(args.sensitivity_range[0], args.sensitivity_range[1], args.sensitivity_range[2])
        return sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)

    if args.evaluate:
        return evaluate_model(model, criterion, test_loader, pylogger, activations_collectors, args)
    
    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress, compression_scheduler)
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        model.cuda()
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)
    
    sched_dict =yaml.load(open(args.compress), Loader=yamlordereddictloader.Loader)
    kahini_quantizer = list(sched_dict['quantizers'].keys())
    quantizer = kahini_quantizer[0]
    layers = get_layers(sched_dict,quantizer)
    
    if quantizer == 'pact_quantizer':
        headers = get_pact_headers(model)
        pact_dict={}
        try:
            with open('./dump_space/pact_track.csv','r') as csvfile:
                reader = csv.DictReader(csvfile)
                for line in reader:
                    pass
                pact_dict = line
                
                update_pact_alpha(pact_dict,model,headers)
                
                msglogger.info("Loading Clip Parameter Value... ... ... ...")
                log_pact(pact_dict)
                # print(model)
                
        except:
            with open('./dump_space/pact_track.csv','w') as csvfile:
                for param_tensor in headers:
                    pact_dict[param_tensor] = float(model.state_dict()[param_tensor].data)
                writer = csv.DictWriter(csvfile,fieldnames = headers)
                writer.writeheader()
                writer.writerow(pact_dict)
                msglogger.info("Writing Clip Parameter Value... ... ... ...")
                log_pact(pact_dict)

    ## Calc Weight Volume:
    quant_weight_volume = get_weight_volume(layers, sched_dict, quantizer, fp=False)
    fp_weight_volume = get_weight_volume(layers, sched_dict, quantizer, fp=True)

    msglogger.info("Total quantized model size in MB {}".format(quant_weight_volume))
    msglogger.info("The relative compression ratio is {}".format(fp_weight_volume/quant_weight_volume))
    
    quant_weight_volume *= 1000
    quant_weight_volume = round(quant_weight_volume)
    quant_weight_volume /= 1000
    
    args.kd_policy = None
    if args.kd_teacher:
        teacher = create_model(args.kd_pretrained, args.dataset, args.kd_teacher, device_ids=args.gpus)
        if args.kd_resume:
            teacher, _, _ = apputils.load_checkpoint(teacher, chkpt_file=args.kd_resume)
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
        args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                         frequency=1)

        msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
        msglogger.info('\tTeacher Model: %s', args.kd_teacher)
        msglogger.info('\tTemperature: %s', args.kd_temp)
        msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                       ' | '.join(['{:.2f}'.format(val) for val in dlw]))
        msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)
	
    # vloss = float('inf') # fix something absurdly large not possible

    end_epoch = start_epoch + args.epochs 
    
    alpha_dict={}
    try:
        with open('./dump_space/alpha_track.csv','r') as csvfile:
            reader = csv.DictReader(csvfile)
            for line in reader:
                pass
            alpha_dict = line

    except:
        with open('./dump_space/alpha_track.csv','w') as csvfile:
            writer = csv.DictWriter(csvfile,fieldnames = layers)
            for layer in layers:
                alpha_dict[layer] = 1
            writer.writeheader()
            writer.writerow(alpha_dict)
    
    # for epoch in range(start_epoch, end_epoch):
    epoch = start_epoch
    while(1):
        # This is the main training loop.
        msglogger.info('\n')
        
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch) #1st intro to vloss
                      
        # Train for one epoch
        with collectors_context(activations_collectors["train"]) as collectors:
            train(train_loader, model, criterion, optimizer, epoch, start_epoch, compression_scheduler,
                  loggers=[tflogger, pylogger], args=args)                     
            distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])
            #print_qe(model)
            # distiller.log_quantization_error(model,loggers=[pylogger])
            distiller.log_activation_statsitics(epoch, "train", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            if args.masks_sparsity:
                msglogger.info(distiller.masks_sparsity_tbl_summary(model, compression_scheduler))
        
        # evaluate on validation set
        with collectors_context(activations_collectors["valid"]) as collectors:
            top1, top5, vloss = validate(val_loader, model, criterion, [pylogger], args, epoch)

            key = 'Epoch_' + str(epoch)
            val_loss_track[key]=str(vloss)+','+str(top1)+','+str(quant_weight_volume)+' mb'
            ##kaap_jhap end
            distiller.log_activation_statsitics(epoch, "valid", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir)

        stats = ('Peformance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5)]))
        distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1, log_freq=1,
                                        loggers=[tflogger])

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        # Update the list of top scores achieved so far, and save the checkpoint
        is_best = top1 > best_epochs[-1].top1
        if top1 > best_epochs[0].top1:
            best_epochs[0].epoch = epoch
            best_epochs[0].top1 = top1
            # Keep best_epochs sorted such that best_epochs[0] is the lowest top1 in the best_epochs list
            best_epochs = sorted(best_epochs, key=lambda score: score.top1)
        for score in reversed(best_epochs):
            if score.top1 > 0:
                msglogger.info('==> Best Top1: %.3f on Epoch: %d', score.top1, score.epoch)
        
        # tweak for ephsilon
        end_epoch = epoch+2
        
        key_val = val_loss_track[key]
        curr_acc = get_valid_acc(key_val)
        msglogger.info("Current Validation Accuracy: " + str(curr_acc))
        
        exit_flag = False
        
        if (epoch-start_epoch) > 2:
        
            if(curr_acc>0.99*top1):
                msglogger.info("Goal obtained!!")
                end_epoch = end_epoch-1
                exit_flag = True
            else:
                msglogger.info("Better Luck Next Time...!")

        apputils.save_checkpoint(epoch, end_epoch, args.arch, model, optimizer, compression_scheduler,
                                 best_epochs[-1].top1, is_best, args.name, msglogger.logdir)
        
        epoch = epoch+1

        if(exit_flag):
            break
        
        #print_model(model)
    write_loss(val_loss_file,val_loss_track)
    write_loss(train_loss_file,train_loss_track)
    # Finally run results on the test set
    #test(test_loader, model, criterion, [pylogger], activations_collectors, args=args)
    
    if quantizer == 'pact_quantizer':
        ## Update the pact param values
        for param_tensor in headers:
            pact_dict[param_tensor] = float(model.state_dict()[param_tensor])
        
        msglogger.info("Saving PACT param_value.......")
        # logging in the csv file
        with open('./dump_space/pact_track.csv','a') as csvfile:
            writer = csv.DictWriter(csvfile,fieldnames = headers)
            writer.writerow(pact_dict)
            
            msglogger.info("The PACT Clip Parameter Value............")
            log_pact(pact_dict)
    
    current_qstep = int(args.name.strip('resq'))
    if (current_qstep == args.qsteps):
        msglogger.info('The final alpha dict is:')
        data = list(alpha_dict.items())
        t = tabulate.tabulate(data, headers=["Layers", "Alpha"], tablefmt="psql") 
        msglogger.info('\n')
        msglogger.info(t)
        quit()
    
    ##CCQ
    ################################################################################################
    #To get the next level quantization error
    ################################################################################################
    
    filename = 'checkpoint.pth.tar' if args.name is None else args.name + '_checkpoint.pth.tar'
    fullpath = os.path.join(msglogger.logdir, filename)  
    
    msglogger.info('Doing the online step for all the layers')
    
    not_sleeping = []
    for item in layers:
        model_test = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
        model_test, megh_roddur, start_epoch = apputils.load_checkpoint(
            model_test, chkpt_file=fullpath, print_line=False)
        
        litmas = sched_dict['quantizers'][quantizer]['bits_overrides'][item]['wts']
        msglogger.info("layer: {} current bit-width: {}".format(item, litmas)) 
        
        if(litmas<=3):
            not_sleeping.append(0)
        else:
            not_sleeping.append(1)
            bp = sched_dict['quantizers'][quantizer]['bits_overrides'][item]['wts']
            current_config = bitmap.index(bp)
            sched_dict['quantizers'][quantizer]['bits_overrides'][item]['wts'] = int(bitmap[current_config+1])
            sched_dict['quantizers'][quantizer]['bits_overrides'][item]['acts'] = int(bitmap[current_config+1])
            flag = False
            if item == 'conv1':
                acts = 'relu'
                sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['wts'] = int(bitmap[current_config+1])
                sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['acts'] = int(bitmap[current_config+1])
                flag = True
            elif 'conv' in item:
                acts = item.replace('conv','relu')
                sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['wts'] = int(bitmap[current_config+1])
                sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['acts'] = int(bitmap[current_config+1])
                flag = True
            dump_yaml(sched_dict,secondary,'w')
            megh_roddur = distiller.file_config(model_test, optimizer, secondary, megh_roddur, print_line=False)
                        
            if quantizer == 'pact_quantizer':
                update_pact_alpha(pact_dict,model_test,headers)
                model_test.cuda()
                # print(model_test)
                
            top1, top5, vloss = _validate(bandit_loader, model_test, criterion,[pylogger], args, epoch)
            sched_dict['quantizers'][quantizer]['bits_overrides'][item]['wts'] = bp
            sched_dict['quantizers'][quantizer]['bits_overrides'][item]['acts'] = bp
            if(flag):
                sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['wts'] = bp
                sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['acts'] = bp

            alpha_dict[item] = float(alpha_dict[item])*math.exp(top1/100.0)
            # msglogger.info('Gradual change in alpha values')
            # data = list(alpha_dict.items())
            # t = tabulate.tabulate(data, headers=["Layers", "Alpha"], tablefmt="psql") 
            # msglogger.info(t)  
            msglogger.info("Done updating alpha for layer {}".format(item))  
        
    # logging in the csv file
    with open('./dump_space/alpha_track.csv','a') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames = layers)
        writer.writerow(alpha_dict)
    
    # Convert the layers to probability
    new_alpha_vals = np.asarray(list(alpha_dict.values()))
    alpha_vals = new_alpha_vals.astype(float)
   
    msglogger.info('Not Sleeping:')
    msglogger.info(not_sleeping)
    
    inter_probability = alpha_vals/sum(alpha_vals)
    msglogger.info('Inter Probability:')
    msglogger.info(list(inter_probability))
    
    not_sleeping = np.asarray(not_sleeping)
    
    # Taking bit*param_size 
    bit_config = []
    for item in layers:
        bits = sched_dict['quantizers'][quantizer]['bits_overrides'][item]['wts']
        if(bits<=3):
            diff = 0
            bit_config.append(diff)
        else:
            curr_id = bitmap.index(bits)
            next_bits = bitmap[curr_id+1]
            diff = bits-next_bits
            bit_config.append(diff)
            
    bit_config = np.array(bit_config)
    weight_list = weight_list*bit_config
    
    temp = weight_list*not_sleeping
    ri = temp/sum(temp)
    msglogger.info('Weight Effect:')
    msglogger.info(list(ri))
    
    ## Lamda schedule
    
    if current_qstep%2: # This will work if current_qstep is odd
        lamda = 0.0
    else:
        lamda = 0.6*(args.qsteps-current_qstep)/args.qsteps + 0.4 #This is hard coded for refar65
    
    if(lamda<0):
        lamda=0

    msglogger.info('The lambda value is: ' + str(lamda))
    
    probability = (1-lamda)*inter_probability + lamda*ri
    probability = list(probability)
    msglogger.info('Final Probability:')
    msglogger.info(probability)
    
    selection = np.random.choice(layers,replace=False,p=probability)
    msglogger.info(selection+', bp ='+ str(sched_dict['quantizers'][quantizer]['bits_overrides'][selection]['wts']))
    while(sched_dict['quantizers'][quantizer]['bits_overrides'][selection]['wts']>=2):
        if(sched_dict['quantizers'][quantizer]['bits_overrides'][selection]['wts']==4):
            alpha_dict[selection] = 0.0
            write_layer_name(selection,sched_dict)
            break
        elif(sched_dict['quantizers'][quantizer]['bits_overrides'][selection]['wts']<=3):
            selection = np.random.choice(layers,replace=False,p=probability)
            msglogger.info(selection+', bp ='+ str(sched_dict['quantizers'][quantizer]['bits_overrides'][selection]['wts']))
        else:
            write_layer_name(selection,sched_dict)
            break
        
    msglogger.info('New alpha values')
    data = list(alpha_dict.items())
    t = tabulate.tabulate(data, headers=["Layers", "Alpha"], tablefmt="psql") 
    msglogger.info('\n')
    msglogger.info(t)    
    
    msglogger.info('Probability')
    msglogger.info('\n')
    msglogger.info(probability)
    
    msglogger.info('Restating alpha...')
    
    for key in alpha_dict.keys():
        if(float(alpha_dict[key])!=0.0):
            alpha_dict[key] = 1.0
    
    msglogger.info('After Restatements')
    data = list(alpha_dict.items())
    t = tabulate.tabulate(data, headers=["Layers", "Alpha"], tablefmt="psql") 
    msglogger.info('\n')
    msglogger.info(t)
            
    # logging in the csv file
    with open('./dump_space/alpha_track.csv','a') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames = layers)
        writer.writerow(alpha_dict)
        
OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


def train(train_loader, model, criterion, optimizer, epoch, start_epoch,
          compression_scheduler, loggers, args):
    global train_loss_track
    """Training loop for one epoch."""
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    # For Early Exit, we define statistics for each exit
    # So exiterrors is analogous to classerr for the non-Early Exit case
    if args.earlyexit_lossweights:
        args.exiterrors = []
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to train mode
    model.train()
    end = time.time()
    
    # for pname, param in model.named_parameters():
        # s = pname + ',' + str(param.requires_grad)
        # print(s)
    
    total_iter = 10*steps_per_epoch
    # print(total_iter)
    lr_max = 0.001
    lr_min = args.lr
    # print(initial_lr)
    
    for train_step, (inputs, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to('cuda'), target.to('cuda')

        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)
        
        if((epoch-start_epoch)<5):
            curr_lr = lr_min
        elif((epoch-start_epoch)>14):
            curr_lr = lr_min
        else:
            ## add warm up schedule
            curr_step = ((epoch-start_epoch-5)*steps_per_epoch)+train_step
            ## print(curr_step)
            curr_lr = lr_min + 0.5*(lr_max-lr_min)*(1+math.cos(curr_step*math.pi/total_iter))
        
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
        
        if args.kd_policy is None:
            output = model(inputs)
        else:
            output = args.kd_policy.forward(inputs)

        if not args.earlyexit_lossweights:
            loss = criterion(output, target)
            
            #### loss tracking -> my addition
            if(train_step%50==0):
                key = 'Epoch_'+str(epoch)+'_Step_'+str(train_step)
                train_loss_track[key]=loss.cpu().item()
            
            # Measure accuracy and record loss
            classerr.add(output.data, target)
        else:
            # Measure accuracy and record loss
            loss = earlyexit_loss(output, target, criterion, args)
            ### this is the one

        losses[OBJECTIVE_LOSS_KEY].add(loss.item())

        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())
            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        ##insert gradient clipping --> FFK
        # if(epoch>220):
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        
        optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % args.print_freq == 0:
            # Log some statistics
            errs = OrderedDict()
            if not args.earlyexit_lossweights:
                errs['Top1'] = classerr.value(1)
                errs['Top5'] = classerr.value(5)
            else:
                # for Early Exit case, the Top1 and Top5 stats are computed for each exit.
                for exitnum in range(args.num_exits):
                    errs['Top1_exit' + str(exitnum)] = args.exiterrors[exitnum].value(1)
                    errs['Top5_exit' + str(exitnum)] = args.exiterrors[exitnum].value(5)

            stats_dict = OrderedDict()
            for loss_name, meter in losses.items():
                stats_dict[loss_name] = meter.mean
            stats_dict.update(errs)
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Peformance/Training/', stats_dict)

            params = model.named_parameters() if args.log_params_histograms else None
            distiller.log_training_progress(stats,
                                            params,
                                            epoch, steps_completed,
                                            steps_per_epoch, args.print_freq,
                                            loggers)
        end = time.time()


def validate(val_loader, model, criterion, loggers, args, epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch)


def test(test_loader, model, criterion, loggers, activations_collectors, args):
    """Model Test"""
    msglogger.info('--- test ---------------------')

    with collectors_context(activations_collectors["test"]) as collectors:
        top1, top5, lossses = _validate(test_loader, model, criterion, loggers, args)
        distiller.log_activation_statsitics(-1, "test", loggers, collector=collectors['sparsity'])
        save_collectors_data(collectors, msglogger.logdir)
    return top1, top5, lossses


def _validate(data_loader, model, criterion, loggers, args, epoch=-1):
    """Execute the validation/test loop."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

    if args.earlyexit_thresholds:
        # for Early Exit, we have a list of errors and losses for each of the exits.
        args.exiterrors = []
        args.losses_exits = []
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))
            args.losses_exits.append(tnt.AverageValueMeter())
        args.exit_taken = [0] * args.num_exits

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes)
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    for validation_step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():
            inputs, target = inputs.to('cuda'), target.to('cuda')
            # compute output from model
            output = model(inputs)

            if not args.earlyexit_thresholds:
                # compute loss
                loss = criterion(output, target)
                # measure accuracy and record loss
                losses['objective_loss'].add(loss.item())
                classerr.add(output.data, target)
                if args.display_confusion:
                    confusion.add(output.data, target)
            else:
                earlyexit_validate_loss(output, target, criterion, args)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step+1)
            if steps_completed % args.print_freq == 0:
                if not args.earlyexit_thresholds:
                    stats = ('',
                            OrderedDict([('Loss', losses['objective_loss'].mean),
                                         ('Top1', classerr.value(1)),
                                         ('Top5', classerr.value(5))]))
                else:
                    stats_dict = OrderedDict()
                    stats_dict['Test'] = validation_step
                    for exitnum in range(args.num_exits):
                        la_string = 'LossAvg' + str(exitnum)
                        stats_dict[la_string] = args.losses_exits[exitnum].mean
                        # Because of the nature of ClassErrorMeter, if an exit is never taken during the batch,
                        # then accessing the value(k) will cause a divide by zero. So we'll build the OrderedDict
                        # accordingly and we will not print for an exit error when that exit is never taken.
                        if args.exit_taken[exitnum]:
                            t1 = 'Top1_exit' + str(exitnum)
                            t5 = 'Top5_exit' + str(exitnum)
                            stats_dict[t1] = args.exiterrors[exitnum].value(1)
                            stats_dict[t5] = args.exiterrors[exitnum].value(5)
                    stats = ('Performance/Validation/', stats_dict)

                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)
    if not args.earlyexit_thresholds:
        msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                       classerr.value()[0], classerr.value()[1], losses['objective_loss'].mean)

        if args.display_confusion:
            msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
        return classerr.value(1), classerr.value(5), losses['objective_loss'].mean
    else:
        total_top1, total_top5, losses_exits_stats = earlyexit_validate_stats(args)
        return total_top1, total_top5, losses_exits_stats[args.num_exits-1]


def earlyexit_loss(output, target, criterion, args):
    loss = 0
    sum_lossweights = 0
    for exitnum in range(args.num_exits-1):
        loss += (args.earlyexit_lossweights[exitnum] * criterion(output[exitnum], target))
        sum_lossweights += args.earlyexit_lossweights[exitnum]
        args.exiterrors[exitnum].add(output[exitnum].data, target)
    # handle final exit
    loss += (1.0 - sum_lossweights) * criterion(output[args.num_exits-1], target)
    args.exiterrors[args.num_exits-1].add(output[args.num_exits-1].data, target)
    return loss


def earlyexit_validate_loss(output, target, criterion, args):
    # We need to go through each sample in the batch itself - in other words, we are
    # not doing batch processing for exit criteria - we do this as though it were batchsize of 1
    # but with a grouping of samples equal to the batch size.
    # Note that final group might not be a full batch - so determine actual size.
    this_batch_size = target.size()[0]
    earlyexit_validate_criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    for exitnum in range(args.num_exits):
        # calculate losses at each sample separately in the minibatch.
        args.loss_exits[exitnum] = earlyexit_validate_criterion(output[exitnum], target)
        # for batch_size > 1, we need to reduce this down to an average over the batch
        args.losses_exits[exitnum].add(torch.mean(args.loss_exits[exitnum]))

    for batch_index in range(this_batch_size):
        earlyexit_taken = False
        # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
        for exitnum in range(args.num_exits - 1):
            if args.loss_exits[exitnum][batch_index] < args.earlyexit_thresholds[exitnum]:
                # take the results from early exit since lower than threshold
                args.exiterrors[exitnum].add(torch.tensor(np.asarray(output[exitnum].data[batch_index], ndmin=2)),
                        torch.full([1], target[batch_index], dtype=torch.long))
                args.exit_taken[exitnum] += 1
                earlyexit_taken = True
                break                    # since exit was taken, do not affect the stats of subsequent exits
        # this sample does not exit early and therefore continues until final exit
        if not earlyexit_taken:
            exitnum = args.num_exits - 1
            args.exiterrors[exitnum].add(torch.tensor(np.asarray(output[exitnum].data[batch_index], ndmin=2)),
                    torch.full([1], target[batch_index], dtype=torch.long))
            args.exit_taken[exitnum] += 1

def earlyexit_validate_stats(args):
    # Print some interesting summary stats for number of data points that could exit early
    top1k_stats = [0] * args.num_exits
    top5k_stats = [0] * args.num_exits
    losses_exits_stats = [0] * args.num_exits
    sum_exit_stats = 0
    for exitnum in range(args.num_exits):
        if args.exit_taken[exitnum]:
            sum_exit_stats += args.exit_taken[exitnum]
            msglogger.info("Exit %d: %d", exitnum, args.exit_taken[exitnum])
            top1k_stats[exitnum] += args.exiterrors[exitnum].value(1)
            top5k_stats[exitnum] += args.exiterrors[exitnum].value(5)
            losses_exits_stats[exitnum] += args.losses_exits[exitnum].mean
    for exitnum in range(args.num_exits):
        if args.exit_taken[exitnum]:
            msglogger.info("Percent Early Exit %d: %.3f", exitnum,
                           (args.exit_taken[exitnum]*100.0) / sum_exit_stats)
    total_top1 = 0
    total_top5 = 0
    for exitnum in range(args.num_exits):
        total_top1 += (top1k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
        total_top5 += (top5k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
        msglogger.info("Accuracy Stats for exit %d: top1 = %.3f, top5 = %.3f", exitnum, top1k_stats[exitnum], top5k_stats[exitnum])
    msglogger.info("Totals for entire network with early exits: top1 = %.3f, top5 = %.3f", total_top1, total_top5)
    return(total_top1, total_top5, losses_exits_stats)

def evaluate_model(model, criterion, test_loader, loggers, activations_collectors, args):
    # This sample application can be invoked to evaluate the accuracy of your model on
    # the test dataset.
    # You can optionally quantize the model to 8-bit integer before evaluation.
    # For example:
    # python3 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --resume=checkpoint.pth.tar --evaluate

    if not isinstance(loggers, list):
        loggers = [loggers]

    if args.quantize_eval:
        model.cpu()
        quantizer = quantization.SymmetricLinearQuantizer(model, args.qe_bits_acts, args.qe_bits_wts,
                                                          args.qe_bits_accum, args.qe_clip_acts,
                                                          args.qe_no_clip_layers)
        quantizer.prepare_model()
        model.cuda()

    top1, _, _ = test(test_loader, model, criterion, loggers, activations_collectors, args=args)

    if args.quantize_eval:
        checkpoint_name = 'quantized'
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, best_top1=top1,
                                 name='_'.join([args.name, checkpoint_name]) if args.name else checkpoint_name,
                                 dir=msglogger.logdir)


def summarize_model(model, dataset, which_summary):
    if which_summary.startswith('png'):
        apputils.draw_img_classifier_to_file(model, 'model.png', dataset, which_summary == 'png_w_params')
    elif which_summary == 'onnx':
        apputils.export_img_classifier_to_onnx(model, 'model.onnx', dataset)
    else:
        distiller.model_summary(model, which_summary, dataset)


def sensitivity_analysis(model, criterion, data_loader, loggers, args, sparsities):
    # This sample application can be invoked to execute Sensitivity Analysis on your
    # model.  The ouptut is saved to CSV and PNG.
    msglogger.info("Running sensitivity tests")
    if not isinstance(loggers, list):
        loggers = [loggers]
    test_fnc = partial(test, test_loader=data_loader, criterion=criterion,
                       loggers=loggers, args=args,
                       activations_collectors=create_activation_stats_collectors(model, None))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=args.sensitivity)
    distiller.sensitivities_to_png(sensitivity, 'sensitivity.png')
    distiller.sensitivities_to_csv(sensitivity, 'sensitivity.csv')


def automated_deep_compression(model, criterion, loggers, args):
    import examples.automated_deep_compression.ADC as ADC
    HAVE_COACH_INSTALLED = True
    if not HAVE_COACH_INSTALLED:
        raise ValueError("ADC is currently experimental and uses non-public Coach features")

    if not isinstance(loggers, list):
        loggers = [loggers]

    train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_size, args.deterministic)

    args.display_confusion = True
    validate_fn = partial(validate, val_loader=test_loader, criterion=criterion,
                          loggers=loggers, args=args)

    if args.ADC_params is not None:
        ADC.summarize_experiment(args.ADC_params, args.dataset, args.arch, validate_fn)
        exit()

    save_checkpoint_fn = partial(apputils.save_checkpoint, arch=args.arch, dir=msglogger.logdir)
    ADC.do_adc(model, args.dataset, args.arch, val_loader, validate_fn, save_checkpoint_fn)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
