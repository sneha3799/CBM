"""
Train InceptionV3/BcosResNet18 Concept Bottleneck Models on the CUB-200-2011 dataset.

This script supports multiple architectures and training paradigms including:
- X → C       : Concept prediction
- C → Y       : Oracle classifier using ground-truth concepts
- X → Ĉ → Y   : Sequential (two-stage) model
- X → Y       : Standard CNN classifier
- X → (C, Y)  : Multitask model
- X → C → Y   : End-to-end Concept Bottleneck Model (joint training)

Includes training, validation, logging, model checkpointing, and optional
support for weighted losses, resampling, uncertain labels, and early stopping.
"""

import pdb
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np
from analysis import Logger, AverageMeter, accuracy, binary_accuracy

# from CUB import probe, tti, gen_cub_synthetic, hyperopt
import hyperopt
from dataset import load_data, find_class_imbalance
from config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY


def run_epoch_simple(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training):
    """
    Train or evaluate a simple attribute-to-class (A → Y) MLP model.

    This function handles cases where inputs are concept activations rather than images.
    It computes forward passes, classification loss, and top-1 accuracy, and optionally
    performs optimization during training.

    Args:
        model (nn.Module): Model to train or evaluate.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        loader (DataLoader): DataLoader providing attribute and label batches.
        loss_meter (AverageMeter): Tracks average loss across batches.
        acc_meter (AverageMeter): Tracks average accuracy across batches.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        args (argparse.Namespace): Experiment configuration.
        is_training (bool): Whether to train (True) or only evaluate (False).

    Returns:
        tuple: Updated (loss_meter, acc_meter).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            # inputs is list of attribute columns -> convert to NxD
            inputs = torch.stack(inputs).t().float()
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = inputs.to(device)
        labels_var = labels.to(device)

        outputs = model(inputs_var)
        # model may return e.g. list or single tensor; adapt:
        if isinstance(outputs, (tuple, list)) and not isinstance(outputs, torch.Tensor):
            # If a list and first element is main class logits, then take outputs[0]
            # else the model may return only a tensor-like container; try to use outputs[0]
            main_out = outputs[0] if len(outputs) > 0 else outputs
        else:
            main_out = outputs

        loss = criterion(main_out, labels_var)
        acc = accuracy(main_out.to('cpu'), labels.to('cpu'), topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc, inputs.size(0))

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss_meter, acc_meter

def run_epoch(model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, is_training):
    """
    Runs one full training or evaluation epoch for CNN-based architectures (e.g., InceptionV3, BcosResNet18).

    Supports various Concept Bottleneck configurations:
        - Standard classification (X → Y)
        - Bottleneck attribute prediction (X → C)
        - Joint/Multitask setups (X → (C, Y))
        - End-to-end CBM (X → C → Y)
    Compatible with auxiliary logits (Inception-style) and both binary and multiclass attributes.

    Args:
        model (nn.Module): Model to train or evaluate.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        loader (DataLoader): DataLoader providing inputs, labels, and attributes.
        loss_meter (AverageMeter): Tracks average loss.
        acc_meter (AverageMeter): Tracks average accuracy.
        criterion (nn.Module): Primary classification loss function.
        attr_criterion (list[nn.Module] or None): List of attribute-level loss functions.
        args (argparse.Namespace): Experiment configuration.
        is_training (bool): Whether to perform backpropagation and optimization.

    Returns:
        tuple: Updated (loss_meter, acc_meter).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            # build N x n_attributes tensor (or N x 1 for single attribute)
            if args.n_attributes > 1:
                # attr_labels is a list per attribute -> stack and transpose -> N x n_attributes
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()  # N x n_attributes
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels_var = attr_labels.float().to(device)

        inputs_var = inputs.to(device)
        labels_var = labels.to(device)

        # call model
        raw_outputs = model(inputs_var)

        # Support three common return shapes:
        # 1) (outputs, aux_outputs)   -> two-element tuple (old Inception training-time behavior)
        # 2) list/tuple of outputs    -> outputs[0] == class logits; outputs[1:] == attributes (or vice versa)
        # 3) single tensor            -> class logits only
        aux_outputs = None
        outputs = raw_outputs

        if isinstance(raw_outputs, tuple) and len(raw_outputs) == 2:
            outputs, aux_outputs = raw_outputs
        elif isinstance(raw_outputs, list) or isinstance(raw_outputs, tuple):
            # keep outputs as-is (list/tuple), aux_outputs remains None
            outputs = list(raw_outputs)
        else:
            outputs = raw_outputs  # single tensor

        # Convert any outputs that are tensors into list for uniform handling in later code:
        outputs_is_list = isinstance(outputs, (list, tuple))
        if not outputs_is_list:
            # single tensor - treat as [main_logits]
            outputs_list = [outputs]
        else:
            outputs_list = list(outputs)

        # If aux_outputs exists and is not a tensor/list of tensors, set None
        aux_list = None
        if aux_outputs is not None:
            aux_list = aux_outputs if isinstance(aux_outputs, (list, tuple)) else [aux_outputs]

        # Build losses list following previous convention:
        losses = []
        out_start = 0

        # If not bottleneck, the first element is main classification logits
        if not args.bottleneck:
            main_logits = outputs_list[0]
            # ensure device alignment
            if isinstance(main_logits, torch.Tensor):
                main_logits = main_logits.to(device)
            if is_training and aux_list is not None and args.use_aux:
                aux_main = aux_list[0]
                loss_main = 1.0 * criterion(main_logits, labels_var) + 0.4 * criterion(aux_main.to(device), labels_var)
            else:
                loss_main = criterion(main_logits, labels_var)
            losses.append(loss_main)
            out_start = 1

        # Attribute losses (if any)
        if attr_criterion is not None and args.attr_loss_weight > 0:
            # attr_criterion is a list of criterion functions, one per attribute
            for i in range(len(attr_criterion)):
                idx = i + out_start
                if idx < len(outputs_list):
                    out_i = outputs_list[idx]
                    # squeeze and cast as needed for BCEWithLogits (float) or CrossEntropy (long)
                    if isinstance(attr_criterion[i], torch.nn.BCEWithLogitsLoss):
                        logits_i = out_i.squeeze().to(device).float()
                        target_i = attr_labels_var[:, i].to(device)
                        if is_training and aux_list is not None and args.use_aux:
                            aux_logits_i = aux_list[idx].squeeze().to(device).float() if idx < len(aux_list) else None
                            if aux_logits_i is not None:
                                losses.append(args.attr_loss_weight * (1.0 * attr_criterion[i](logits_i, target_i) + 0.4 * attr_criterion[i](aux_logits_i, target_i)))
                            else:
                                losses.append(args.attr_loss_weight * attr_criterion[i](logits_i, target_i))
                        else:
                            losses.append(args.attr_loss_weight * attr_criterion[i](logits_i, target_i))
                    else:
                        # CrossEntropyLoss expects long targets
                        logits_i = out_i.squeeze().to(device)
                        target_i = attr_labels_var[:, i].long().to(device)
                        if is_training and aux_list is not None and args.use_aux:
                            aux_logits_i = aux_list[idx].squeeze().to(device) if idx < len(aux_list) else None
                            if aux_logits_i is not None:
                                losses.append(args.attr_loss_weight * (1.0 * attr_criterion[i](logits_i, target_i) + 0.4 * attr_criterion[i](aux_logits_i, target_i)))
                            else:
                                losses.append(args.attr_loss_weight * attr_criterion[i](logits_i, target_i))
                        else:
                            losses.append(args.attr_loss_weight * attr_criterion[i](logits_i, target_i))
                else:
                    # Missing outputs for this attribute - add zero loss to keep indexing consistent
                    losses.append(torch.tensor(0.0, device=device))

        # Accuracy computation
        if args.bottleneck:
            attr_logits_list = []
            for t in outputs_list:
                if isinstance(t, torch.Tensor):
                    # Ensure shape [batch, 1]
                    if t.dim() == 1:
                        t = t.unsqueeze(1)
                    elif t.dim() > 2:
                        t = t.view(t.size(0), -1)
                    attr_logits_list.append(t.to(device))
            if len(attr_logits_list) > 0:
                concat_logits = torch.cat(attr_logits_list, dim=1)
                sigmoid_outputs = torch.sigmoid(concat_logits)
                acc = binary_accuracy(sigmoid_outputs, attr_labels[:, :args.n_attributes])
                acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            # normal classification accuracy by comparing outputs_list[0] to labels
            main_logits = outputs_list[0]
            acc = accuracy(main_logits.to('cpu'), labels.to('cpu'), topk=(1,))
            acc_meter.update(acc, inputs.size(0))

        # Compose total_loss
        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = sum(losses) / max(1, args.n_attributes)
            else:
                # main loss + attribute losses
                total_loss = losses[0] + sum(losses[1:]) if len(losses) > 0 else sum(losses)
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
        else:
            total_loss = sum(losses) if len(losses) > 0 else torch.tensor(0.0, device=device)

        loss_meter.update(total_loss.item(), inputs.size(0))

        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return loss_meter, acc_meter

def train(model, args):
    """
    Core training loop handling initialization, loss setup, optimizer scheduling, 
    validation, checkpointing, and early stopping.

    Supports weighted and unweighted attribute losses, auxiliary logits,
    learning rate scheduling, and both image-based and concept-only models.

    Args:
        model (nn.Module): Model to train (e.g., ResNet18, BcosResNet18, MLP).
        args (argparse.Namespace): Experiment configuration including hyperparameters.

    Side Effects:
        - Saves best-performing model checkpoint to args.log_dir.
        - Logs metrics to log.txt.
        - Prints current learning rate and early stopping messages.
    """

    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        if args.weighted_loss == 'multiple':
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)

    if os.path.exists(args.log_dir): # job restarted by cluster
        for f in os.listdir(args.log_dir):
            os.remove(os.path.join(args.log_dir, f))
    else:
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()

    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_attr and not args.no_img:
        attr_criterion = [] #separate criterion (loss function) for each attribute
        if args.weighted_loss:
            assert(imbalance is not None)
            for ratio in imbalance:
                attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda()))
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.00001, min_lr=0.00001, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)

    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)

    if args.ckpt: #retraining
        train_loader = load_data([train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = None
    else:
        train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, args, is_training=True)
        else:
            train_loss_meter, train_acc_meter = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion, args, is_training=True)
 
        if not args.ckpt: # evaluate on val set
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
        
            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, args, is_training=False)
                else:
                    val_loss_meter, val_acc_meter = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, attr_criterion, args, is_training=False)

        else: #retraining
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(args.log_dir, 'best_model_%d.pth' % args.seed))
            #if best_val_acc >= 100: #in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
            #    break

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
        
        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                'Val loss: %.4f\tVal acc: %.4f\t'
                'Best val epoch: %d\n'
                % (epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg, best_val_epoch)) 
        logger.flush()
        
        if epoch <= stop_epoch:
            scheduler.step(epoch) #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            # get_last_lr is available in torch >=1.4; fallback to get_lr() if older
            lr_info = scheduler.get_last_lr() if hasattr(scheduler, "get_last_lr") else scheduler.get_lr()
            print('Current lr:', lr_info)

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break

def train_X_to_C(args):
    """
    Trains a model to predict visual concepts from images (X → C).
    Typically the first stage of a Concept Bottleneck Model.
    """

    model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                      n_attributes=args.n_attributes, expand_dim=args.expand_dim, three_class=args.three_class)
    train(model, args)

def train_oracle_C_to_y_and_test_on_Chat(args):
    """
    Trains an oracle model to map ground-truth concepts to class labels (C → Y).
    Tests on predicted concepts Ĉ during evaluation.
    """

    model = ModelOracleCtoY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                            num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_Chat_to_y_and_test_on_Chat(args):
    """
    Trains a sequential model that maps predicted concepts Ĉ to labels Y.
    """

    model = ModelXtoChat_ChatToY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                                 num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_X_to_C_to_y(args):
    """
    Trains a joint end-to-end Concept Bottleneck Model (X → C → Y).
    The model simultaneously learns to predict both concepts and labels.
    """

    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    train(model, args)

def train_X_to_y(args):
    """
    Trains a standard CNN classifier that predicts labels directly from images (X → Y).
    """

    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux)
    train(model, args)

def train_X_to_Cy(args):
    """
    Trains a multitask model that jointly predicts both concepts and class labels (X → (C, Y)).
    """

    model = ModelXtoCY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                       n_attributes=args.n_attributes, three_class=args.three_class, connect_CY=args.connect_CY)
    train(model, args)

def train_probe(args):
    """Runs probing experiments on pre-trained representations."""

    probe.run(args)

def test_time_intervention(args):
    """Performs Test-Time Intervention (TTI) experiments for CBM robustness."""

    tti.run(args)

def robustness(args):
    """Runs synthetic robustness experiments (e.g., CUB-S synthetic generation)."""

    gen_cub_synthetic.run(args)

def hyperparameter_optimization(args):
    """Performs hyperparameter tuning using HyperOpt."""

    hyperopt.run(args)


def parse_arguments(experiment):
    """
    Parses command-line arguments for different CBM experiment types.

    Depending on the experiment name, returns customized argument sets for:
        - Probe
        - Test-Time Intervention (TTI)
        - Robustness testing
        - Hyperparameter optimization
        - All major training setups (Standard, Joint, Multitask, etc.)

    Args:
        experiment (str): Type of experiment to configure.

    Returns:
        tuple: A single argparse.Namespace or related submodule-specific arguments.
    """
    
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')
    parser.add_argument('dataset', type=str, help='Name of the dataset.')
    parser.add_argument('exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                 'Standard', 'Multitask', 'Joint', 'Probe',
                                 'TTI', 'Robustness', 'HyperparameterSearch'],
                        help='Name of experiment to run.')
    parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')

    if experiment == 'Probe':
        return (probe.parse_arguments(parser),)

    elif experiment == 'TTI':
        return (tti.parse_arguments(parser),)

    elif experiment == 'Robustness':
        return (gen_cub_synthetic.parse_arguments(parser),)

    elif experiment == 'HyperparameterSearch':
        return (hyperopt.parse_arguments(parser),)

    else:
        parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
        parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
        parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
        parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
        parser.add_argument('-lr', type=float, help="learning rate")
        parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
        parser.add_argument('-pretrained', '-p', action='store_true',
                            help='whether to load pretrained model & just fine-tune')
        parser.add_argument('-freeze', action='store_true', help='whether to freeze the bottom part of inception network')
        parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
        parser.add_argument('-use_attr', action='store_true',
                            help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
        parser.add_argument('-attr_loss_weight', default=1.0, type=float, help='weight for loss by predicting attributes')
        parser.add_argument('-no_img', action='store_true',
                            help='if included, only use attributes (and not raw imgs) for class prediction')
        parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
        parser.add_argument('-weighted_loss', default='', # note: may need to reduce lr
                            help='Whether to use weighted loss for single attribute or multiple ones')
        parser.add_argument('-uncertain_labels', action='store_true',
                            help='whether to use (normalized) attribute certainties as labels')
        parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                            help='whether to apply bottlenecks to only a few attributes')
        parser.add_argument('-expand_dim', type=int, default=0,
                            help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
        parser.add_argument('-n_class_attr', type=int, default=2,
                            help='whether attr prediction is a binary or triary classification')
        parser.add_argument('-data_dir', default='official_datasets', help='directory to the training data')
        parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
        parser.add_argument('-resampling', help='Whether to use resampling', action='store_true')
        parser.add_argument('-end2end', action='store_true',
                            help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
        parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
        parser.add_argument('--dropout_p', type=float, default=0.5,
                    help='Dropout probability for regularization')
        parser.add_argument('-ckpt', default='', help='For retraining on both train + val set')
        parser.add_argument('-scheduler_step', type=int, default=1000,
                            help='Number of steps before decaying current learning rate by half')
        parser.add_argument('-normalize_loss', action='store_true',
                            help='Whether to normalize loss by taking attr_loss_weight into account')
        parser.add_argument('-use_relu', action='store_true',
                            help='Whether to include relu activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-use_sigmoid', action='store_true',
                            help='Whether to include sigmoid activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-connect_CY', action='store_true',
                            help='Whether to use concepts as auxiliary features (in multitasking) to predict Y')
        args = parser.parse_args()
        args.three_class = (args.n_class_attr == 3)
        return (args,)