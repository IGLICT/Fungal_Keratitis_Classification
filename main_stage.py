import argparse
import numpy as np
import glob

from datasets import DataInterfaceStage as DataInterface
from models import ModelInterfaceStage as ModelInterface
from utils import *

import pytorch_lightning as pl
from pytorch_lightning import Trainer

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='FungalKeratitis/SwinTransformer.yaml', type=str)
    # parser.add_argument('--gpus', default=[0, 1])
    args = parser.parse_args()
    return args

# main function
def main(cfg):
    # Initialize seed
    pl.seed_everything(cfg.General.seed)
    # load loggers
    cfg.load_loggers = load_loggers(cfg)
    # load callbacks
    cfg.callbacks = load_callbacks(cfg)

    # Define Data
    cfg.Data.exp_name = cfg.General.exp_name
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                          'train_num_workers': cfg.Data.train_dataloader.num_workers,
                          'test_batch_size': cfg.Data.test_dataloader.batch_size,
                          'test_num_workers': cfg.Data.test_dataloader.num_workers,
                          'dataset_name': cfg.Data.dataset_name,
                          'dataset_cfg': cfg.Data}
    dm = DataInterface(**DataInterface_dict)

    # Define Model
    ModelInterface_dict = {'model': cfg.Model,
                           'loss': cfg.Loss,
                           'optimizer': cfg.Optimizer,
                           'data': cfg.Data,
                           'log': cfg.log_path, 
                           'result': cfg.result_path}
    model = ModelInterface(**ModelInterface_dict)

    # Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0,
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs=cfg.General.epochs,
        gpus=cfg.General.gpus,
        accumulate_grad_batches=cfg.General.grad_acc,
        check_val_every_n_epoch=1,
    )

    # train or test
    if cfg.General.stage == 'train':
        trainer.fit(model=model, datamodule=dm)
    else:
        from utils import get_local
        get_local.activate()
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path)
            new_model.result_path=cfg.result_path
            new_model.stage = cfg.Data.stage_num
            new_model.imgseq_num = cfg.Data.imgseq_num
            trainer.test(model=new_model, datamodule=dm)
    

if __name__ == '__main__':
    args = make_parse()
    cfg = read_yaml(args.config)
    # torch.hub.set_dir('../ophthalmology/torch_hub/')
    # update
    cfg.config = args.config
    cfg.General.stage = args.stage

    main(cfg)