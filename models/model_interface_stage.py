import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd
# from pytz import NonExistentTimeError
import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import pytorch_lightning as pl
from utils import cross_entropy_torch, save_img_from_tensor
from utils import get_local
import seaborn as sns
import matplotlib.pyplot as plt
get_local.activate()
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from utils import clopper_pearson, draw_roc_curve_ci, draw_confusion_matrix

class ModelInterfaceStage(pl.LightningModule):
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterfaceStage, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']
        self.result_path = kargs['result']
        self.stage = kargs['data'].stage_num
        self.img_size = kargs['data'].imgsize
        if self.stage == 2:
            self.imgseq_num = kargs['data'].imgseq_num
        else:
            self.imgseq_num = 0

        # accuracy
        self.train_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.val_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.test_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        # Metrics
        if self.n_classes > 2:
            self.AUROC = torchmetrics.AUROC(num_classes=self.n_classes, average='macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes=self.n_classes, average='micro'),
                                                     torchmetrics.CohenKappa(num_classes=self.n_classes),
                                                     torchmetrics.classification.f_beta.F1Score(num_classes=self.n_classes, average='macro'),
                                                     torchmetrics.Recall(num_classes=self.n_classes, average='macro'),
                                                     torchmetrics.Precision(num_classes=self.n_classes, average='macro'),
                                                     torchmetrics.Specificity(num_classes=self.n_classes, average='macro')])
        else:
            self.AUROC = torchmetrics.AUROC(num_classes=2, average='macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes=2, average='micro'),
                                                     torchmetrics.CohenKappa(num_classes=2),
                                                     torchmetrics.classification.f_beta.F1Score(num_classes=2, average='macro'),
                                                     torchmetrics.Recall(num_classes=2, average='macro'),
                                                     torchmetrics.Precision(num_classes=2, average='macro')])
        
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        # random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        results_dict = self.model(inputs)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        loss = self.loss(logits, targets)

        for i in range(self.n_classes):
            self.train_data[i]['count'] += (targets == i).sum().item()
            self.train_data[i]['correct'] += ((Y_hat == targets)*(targets == i)).sum().item()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def training_epoch_end(self, training_step_outputs):
        print('')
        for c in range(self.n_classes):
            count = self.train_data[c]['count']
            correct = self.train_data[c]['correct']
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print('train class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.train_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def validation_step(self, batch, batch_idx):
        if self.stage == 1:
            inputs, targets, filenames = batch
        else:
            raise NotImplementedError('Stage %d' % self.stage)
        results_dict = self.model(inputs)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        for c in range(self.n_classes):
            self.val_data[c]['count'] += (targets == c).sum().item()
            self.val_data[c]['correct'] += ((Y_hat == targets)*(targets == c)).sum().item()
        
        return {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': targets}
    
    def validation_epoch_end(self, val_step_outputs):
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim=0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim=0)
        predicts = torch.cat([x['Y_hat'] for x in val_step_outputs], dim=0)
        targets = torch.cat([x['label'] for x in val_step_outputs], dim=0)

        self.log('val_loss', cross_entropy_torch(logits, targets), prog_bar=True, on_epoch=True, logger=True)
        self.log('val_auc', self.AUROC(probs, targets.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(predicts.squeeze(), targets.squeeze()), on_epoch=True, logger=True)

        print('\n')
        for c in range(self.n_classes):
            count = self.val_data[c]['count']
            correct = self.val_data[c]['correct']
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print('val class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.val_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        if self.shuffle:
            self.count = self.count + 1
            random.seed(self.count*50)
    
    def test_step(self, batch, batch_idx):
        if self.stage == 1:
            inputs, targets, filenames = batch
        else:
            raise NotImplementedError('Stage %d' % self.stage)
        if self.stage == 1:
            results_dict = self.model(inputs)
        
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        for c in range(self.n_classes):
            self.test_data[c]['count'] += (targets == c).sum().item()
            self.test_data[c]['correct'] += ((Y_hat == targets)*(targets == c)).sum().item()
        
        if self.stage == 1:
            for i in range(inputs.shape[0]):
                if Y_hat[i] != targets[i]:
                    with open(os.path.join(self.result_path, 'patient_imgs', 'err_msg.txt'), 'a') as f:
                        f.write('%s, %d, %.4f\n' % (filenames[i], targets[i].item(), Y_prob[i, 1].item()))
                    img_filename = 'gt%d_%s_pred%.4f.jpg' % (targets[i].item(), filenames[i], Y_prob[i, 1].item())
                    save_img_from_tensor(os.path.join(self.result_path, 'patient_imgs'), inputs[i], img_filename, img_size=self.img_size)
                # img_path = os.path.join(self.result_path, 'patient_imgs', '%s_gt%d_pred%.4f.jpg' % (filenames[i], targets[i].item(), Y_prob[i, 1].item()))
                # visual_grad_cam(inputs[i].unsqueeze(0), self.model.net, self.hparams.model.model_name, img_path)

        if self.stage <= 3:
            return {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': targets}
    
    def test_epoch_end(self, output_results):
        logits = torch.cat([x['logits'] for x in output_results], dim=0)
        probs = torch.cat([x['Y_prob'] for x in output_results], dim=0)
        predicts = torch.cat([x['Y_hat'] for x in output_results], dim=0)
        targets = torch.cat([x['label'] for x in output_results], dim=0)

        if self.stage <= 3:
            confusion_matrix_filepath = os.path.join(self.result_path, 'confusion_matrix.jpg')
            draw_confusion_matrix(confusion_matrix_filepath, targets.cpu().numpy(), predicts.cpu().numpy())
            roc_auc_curve_filepath = os.path.join(self.result_path, 'roc_curve.jpg')
            roc_auc, roc_auc_ci = draw_roc_curve_ci(roc_auc_curve_filepath, targets.cpu().numpy(), probs[:, 1].cpu().numpy())
            self.log('sklearn_roc_auc', roc_auc, on_epoch=True, logger=True)
            self.log('roc_auc_ci_low', roc_auc_ci[0], on_epoch=True, logger=True)
            self.log('roc_auc_ci_high', roc_auc_ci[1], on_epoch=True, logger=True)
            np.savez(os.path.join(self.result_path, 'prediction.npz'), targets=targets.cpu().numpy(), predicts=predicts.cpu().numpy(), probs=probs.cpu().numpy())


            auc = self.AUROC(probs, targets.squeeze())
            metrics = self.test_metrics(predicts.squeeze(), targets.squeeze())
            metrics['auc'] = auc
            result = pd.DataFrame([metrics])
            result.to_csv(self.result_path / 'result.csv')
            self.log_dict(self.test_metrics(predicts.squeeze(), targets.squeeze()), on_epoch=True, logger=True)
        
        total_correct, total_count = 0, 0
        for c in range(self.n_classes):
            count = self.test_data[c]['count']
            correct = self.test_data[c]['correct']
            if count == 0:
                acc = None
                continue
            else:
                acc = float(correct) / count
            total_correct += correct
            total_count += count
            ci_range = clopper_pearson(correct, count)
            print('test class {}: acc {}({} - {}), correct {}/{}'.format(c, acc, ci_range[0], ci_range[1], correct, count))
            self.log('test_class%d_acc'%c, acc, on_epoch=True, logger=True)
            self.log('test_class%d_acc_ci_low'%c, ci_range[0], on_epoch=True, logger=True)
            self.log('test_class%d_acc_ci_high'%c, ci_range[1], on_epoch=True, logger=True)
        total_ci_range = clopper_pearson(total_correct, total_count)
        self.log('test_acc', total_correct/total_count, on_epoch=True, logger=True)
        self.log('test_acc_ci_low', total_ci_range[0], on_epoch=True, logger=True)
        self.log('test_acc_ci_high', total_ci_range[1], on_epoch=True, logger=True)
        self.test_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        

        self.log('test_loss', cross_entropy_torch(logits, targets), on_epoch=True, logger=True)
        if self.stage <= 3:
            self.log('test_auc', self.AUROC(probs, targets.squeeze()), on_epoch=True, logger=True)
        

    def configure_optimizers(self):
        if hasattr(self.optimizer, 'weight_decay'):
            weight_decay = self.optimizer.weight_decay
        else:
            weight_decay = 0
        
        if self.optimizer.opt == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.optimizer.lr, momentum=self.optimizer.momentum, weight_decay=weight_decay)
        elif self.optimizer.opt == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer.lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid Optimizer Type!')
        
        if self.optimizer.lr_scheduler is None:
            return [optimizer]
        else:
            if self.optimizer.lr_scheduler == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.optimizer.lr_decay_every, gamma=self.optimizer.lr_decay_by)
            elif self.optimizer.lr_scheduler == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.optimizer.lr_decay_every, eta_min=self.optimizer.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]
    
    def configure_loss(self, loss):
        base_loss = loss.base_loss
        if base_loss == 'CrossEntropyLoss':
            self.loss = nn.CrossEntropyLoss()
            print('Use CrossEntropyLoss')
        elif base_loss == 'L1Loss':
            self.loss = nn.L1Loss()
        else:
            raise ValueError('Invalid Loss Type!')
    
    def load_model(self):
        name = self.hparams.model.net_name
        if '_' in name:
            class_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            class_name = name
        try:
            Model = getattr(importlib.import_module(f'models.{name}'), class_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
    
    def instancialize(self, Model, **other_args):
        """Instancialize a model using the corresponding parameters
           from self.hparams dictionary. You can also input any args
           to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)

