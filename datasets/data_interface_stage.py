import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class DataInterfaceStage(pl.LightningDataModule):
    def __init__(self, train_batch_size=64, train_num_workers=8, test_batch_size=64, test_num_workers=8, dataset_name=None, **kwargs):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        self.train_stage = self.kwargs['dataset_cfg']['stage_num']
        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = self.instancialize(state='train')
            self.val_dataset = self.instancialize(state='val')
        
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = self.instancialize(state='test')
    
    def train_dataloader(self):
        if self.kwargs['dataset_cfg']['oversample']:
            train_in_idx = self.train_dataset.get_oversampled_data()
            return DataLoader(self.train_dataset, batch_size=self.train_batch_size, sampler=WeightedRandomSampler(train_in_idx, len(train_in_idx)), num_workers=self.train_num_workers)
        else:
            return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.test_num_workers, shuffle=False)
    
    def load_data_module(self):
        class_name = ''.join([i.capitalize() for i in (self.dataset_name).split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                f'datasets.{self.dataset_name}'), class_name)
        except:
            raise ValueError('Invalid Dataset File Name or Invalid Class Name!')
    
    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)