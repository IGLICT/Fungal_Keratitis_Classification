from pathlib import Path
import os
import shutil
import sys
import time
from PIL import Image
import numpy as np
import torch
from bytecode import Bytecode, Instr
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import math
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

#---->read yaml
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

#---->load Loggers
from pytorch_lightning import loggers as pl_loggers
def load_loggers(cfg):

    log_path = cfg.General.log_path
    result_path = cfg.General.result_path
    Path(log_path).mkdir(exist_ok=True, parents=True)
    Path(result_path).mkdir(exist_ok=True, parents=True)
    log_name = Path(cfg.config).parent 
    # version_name = Path(cfg.config).name[:-5]
    version_name = cfg.General.exp_name
    cfg.log_path = Path(log_path) / log_name / version_name / f'fold{cfg.Data.fold}'
    if os.path.exists(cfg.log_path) and cfg.General.stage == 'train':
        response = input('A training run named "%s" already exists, overwrite? (y/n)' % (version_name))
        if response != "y":
            sys.exit()
        shutil.rmtree(cfg.log_path)
    if cfg.General.stage == 'test':
        cfg.result_path = Path(result_path) / log_name / version_name / f'fold{cfg.Data.fold}'
        if os.path.exists(cfg.result_path):
            response = input('The result run named "%s" already exists, overwrite? (y/n)' % (version_name))
            if response != "y":
                sys.exit()
            shutil.rmtree(cfg.result_path)
        Path(cfg.result_path).mkdir(exist_ok=True, parents=True)
        os.mkdir(os.path.join(cfg.result_path, 'patient_imgs'))
        os.mkdir(os.path.join(cfg.result_path, 'attention_map'))
    Path(cfg.log_path).mkdir(exist_ok=True, parents=True)
    print(f'---->Log dir: {cfg.log_path}')
    shutil.copyfile(cfg.config, os.path.join(cfg.log_path, Path(cfg.config).name[:-5]+'_'+cfg.General.stage+'_'+str(time.time())+'.yaml'))    
    
    #---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(log_path+'/'+str(log_name),
                                             name = version_name, version = f'fold{cfg.Data.fold}',
                                             log_graph = False, default_hp_metric = False)
    #---->CSV
    csv_logger = pl_loggers.CSVLogger(log_path+'/'+str(log_name),
                                      name = version_name, version = f'fold{cfg.Data.fold}', )
    
    return [tb_logger, csv_logger]


#---->load Callback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ProgressBarBase

class MyProgressBar(ProgressBarBase):
    def __init__(self):
        super().__init__()
        self.enable = True
    
    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

def load_callbacks(cfg):
    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    # early_stop_callback = EarlyStopping(
    #     monitor='val_auc',
    #     min_delta=1e-12,
    #     patience=cfg.General.patience,
    #     verbose=True,
    #     mode='max'
    # )
    # Mycallbacks.append(early_stop_callback)

    if cfg.General.stage == 'train' :
        Mycallbacks.append(ModelCheckpoint(monitor = 'val_auc',
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{val_auc:.4f}',
                                         verbose = True,
                                         save_last = False,
                                         save_top_k = 1,
                                         mode = 'max',
                                         save_weights_only = True))
    
    if cfg.Optimizer.lr_scheduler is not None:
        Mycallbacks.append(LearningRateMonitor(logging_interval='step'))
    # Mycallbacks.append(MyProgressBar())
    return Mycallbacks

#---->val loss
import torch
import torch.nn.functional as F
def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i], dim=-1) for i in range(len(x))]
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(len(y))])
    loss = - torch.sum(x_log) / len(y)
    return loss

def save_img_from_tensor(out_dir, img_tensor, filename, img_size=224):
    norm_mean = 0.34224; norm_std = 0.14083894
    output_filepath = os.path.join(out_dir, filename)
    img = (img_tensor.cpu().numpy()*norm_std+norm_mean).reshape(img_size, img_size)*255
    saveimg = Image.fromarray(img.astype(np.uint8))
    saveimg.save(output_filepath)

#----->Attention Visualization
# sorce code: https://github.com/luo3300612/Visualizer/blob/main/visualizer/visualizer.py
class get_local(object):
    cache = {}
    is_activate = False

    def __init__(self, varname):
        self.varname = varname
    
    def __call__(self, func):
        if not type(self).is_activate:
            return func
        
        type(self).cache[func.__qualname__] = []
        c = Bytecode.from_code(func.__code__)
        extra_code = [
                        Instr('STORE_FAST', '_res'),
                        Instr('LOAD_FAST', self.varname),
                        Instr('STORE_FAST', '_value'),
                        Instr('LOAD_FAST', '_res'),
                        Instr('LOAD_FAST', '_value'),
                        Instr('BUILD_TUPLE', 2),
                        Instr('STORE_FAST', '_result_tuple'),
                        Instr('LOAD_FAST', '_result_tuple'),
                     ]
        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        def wrapper(*args, **kwargs):
            res, values = func(*args, **kwargs)
            type(self).cache[func.__qualname__].append(values.detach().cpu().numpy())
            return res
        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []
    
    @classmethod
    def activate(cls):
        cls.is_activate = True

# 95% CI
# accuracy, sensitivity, specificity
def clopper_pearson(x, n, alpha=0.05):
    """Estimate the confidence interval for a sampled Bernoulli random
    variable.
    'x' is the number of successes and 'n' is the number trials (x <= n).
    'alpha' is the confidence level (i.e., the true probability is 
    inside the confidence interval with probability 1-alpha). The 
    function returns a '(low, high)' pair of numbers indicating the
    interval on the probability.
    """
    b = stats.beta.ppf
    lo = b(alpha / 2, x, n-x+1)
    hi = b(1 - alpha / 2, x+1, n-x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi

def bootstrap_auc(all_labels, all_values, n_bootstraps=1000):
    rng_seed = 1 # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(all_values), len(all_values))
        if len(np.unique(all_labels[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(all_labels[indices], all_values[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap # {} ROC area: {:0.3f}".format(i+1, score))
    return bootstrapped_scores

# ROC AUC score
def draw_roc_curve_ci(filepath, y_true, y_prob, alpha=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    all_auc = bootstrap_auc(y_true, y_prob)
    roc_auc_ci = stats.norm.interval(1-alpha, np.mean(all_auc), np.std(all_auc))
    
    fig = plt.figure(0)
    plt.plot(fpr, tpr, '#9400D3', label=u'AUC = %0.6f(%0.6f, %0.6f)'%(roc_auc, roc_auc_ci[0], roc_auc_ci[1]))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('Sensibility')
    plt.xlabel('1 - Specificity')
    plt.grid(linestyle='-.')
    plt.grid(True)
    plt.savefig(filepath)
    plt.close(fig)
    return roc_auc, roc_auc_ci

def roc_auc_ci(y_true, y_prob, alpha=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    all_auc = bootstrap_auc(y_true, y_prob)
    roc_auc_ci = stats.norm.interval(1-alpha, np.mean(all_auc), np.std(all_auc))
    return roc_auc, roc_auc_ci

def draw_confusion_matrix(filepath, y_true, y_pred):
    sns.set()
    f, ax = plt.subplots()

    C2 = confusion_matrix(y_true, y_pred, labels=[i for i in range(2)])
    sns.heatmap(C2, annot=True, fmt='.20g', ax=ax, cmap='Reds')
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('ground truth')
    plt.savefig(filepath)
    plt.close()

# visualize by grad-cam
import cv2
import timm
import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)

def reshape_transform(tensor, height=28, width=28):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def visual_grad_cam(input_tensor, model, model_name, image_path, method='gradcam', use_cuda=True):
    methods  = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
    
    if method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    
    with TemporaryGrad():
        model.eval()
        if model_name.startswith('swin'):
            # target_layers = [model.layers[-1].blocks[-1].norm2]
            target_layers = [model.layers[-3].blocks[-1].norm2]
            if method == 'ablationcam':
                cam = methods[method](model=model,
                                    target_layers=target_layers,
                                    use_cuda=use_cuda,
                                    reshape_transform=reshape_transform,
                                    ablation_layer=AblationLayerVit())
            else:
                cam = methods[method](model=model,
                                    target_layers=target_layers,
                                    use_cuda=use_cuda,
                                    reshape_transform=reshape_transform)
            # rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
            # rgb_img = cv2.resize(rgb_img, (224, 224))
            # rgb_img = np.float32(rgb_img) / 255
            # input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            
            cam.batch_size = 32
            targets = None # ClassifierOutputTarget(1)
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                eigen_smooth=True,
                                aug_smooth=True)
            
            grayscale_cam = grayscale_cam[0, :]
            
            img_size = input_tensor.shape[-1]
            norm_mean = 0.34224; norm_std = 0.14083894
            img = np.uint8((input_tensor.cpu().numpy()*norm_std+norm_mean).reshape(img_size, img_size)*255)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            rgb_img = np.float32(rgb_img) / 255
            ori_image = np.uint8(255*rgb_img)
            cam_image = show_cam_on_image(rgb_img, grayscale_cam)
            output_image = np.concatenate([ori_image, cam_image], axis=1)
            cv2.imwrite(image_path, output_image)
        else:
            raise NotImplementedError('method')

        




