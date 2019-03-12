import os
from collections import defaultdict

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm
from typing import Type

from dataset.neural_dataset import TrainDataset, ValDataset
from .loss import dice_round, dice, BCEDiceLoss, BCEDiceLossWeighted
from .callbacks import ModelSaver, TensorBoard, CheckpointSaver, Callbacks, StepLR
from dataset.reading_image_provider import MixedReadingImageProvider

from pytorch_zoo import unet
import numpy as np

import pdb
torch.backends.cudnn.benchmark = True

models = {
    'resnet34': unet.Resnet34_upsample
}

losses = {
    'bce_dice_loss': BCEDiceLoss,
    'bce_dice_loss_w': BCEDiceLossWeighted
}

optimizers = {
    'adam': optim.Adam
}

class Estimator:
    """
    class takes care about model, optimizer, loss and making step in parameter space
    """
    def __init__(self, model: torch.nn.Module, optimizer: Type[optim.Optimizer], loss: Type[torch.nn.Module], save_path,
                 iter_size=1, lr=1e-4, num_channels_changed=False):
        self.model = nn.DataParallel(model).cuda()
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.criterion = loss().cuda()
        self.iter_size = iter_size
        self.start_epoch = 0
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.num_channels_changed = num_channels_changed

        self.lr_scheduler = None
        self.lr = lr
        self.optimizer_type = optimizer

    def resume(self, checkpoint_name):
        """
        Loads model from checkpoint_name
        """
        try:
            checkpoint = torch.load(os.path.join(self.save_path, checkpoint_name))
        except FileNotFoundError:
            print("resume failed, file not found")
            return False

        self.start_epoch = checkpoint['epoch']

        model_dict = self.model.module.state_dict()
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if self.num_channels_changed:
            skip_layers = self.model.module.first_layer_params_names
            print('skipping: ', [k for k in pretrained_dict.keys() if any(s in k for s in skip_layers)])
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not any(s in k for s in skip_layers)}
            model_dict.update(pretrained_dict)
            self.model.module.load_state_dict(model_dict)
        else:
            model_dict.update(pretrained_dict)
            try:
                self.model.module.load_state_dict(model_dict)
            except:
                print('load state dict failed')
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

        print("resumed from checkpoint {} on epoch: {}".format(os.path.join(self.save_path, checkpoint_name), self.start_epoch))
        return True

    def make_step_itersize(self, images, ytrues, training, metrics):
        """
        Trains the image & returns the metrics as well as the actual outputs
        Returns the output of the model on the images, and the loss associated with the corresponding images
        (well, mean loss)
        """
        iter_size = self.iter_size
        if training:
            self.optimizer.zero_grad()

        inputs = images.chunk(iter_size)
        targets = ytrues.chunk(iter_size)
        outputs = []

        meter = {k: 0 for k,v in metrics}
        meter['loss'] = 0
        for input, target in zip(inputs, targets):
            input = torch.autograd.Variable(input.cuda(async=True)) 
            target = torch.autograd.Variable(target.cuda(async=True))
            print("input shape is:", input.shape, "target shape is:", target.shape)
            if not training:
                with torch.no_grad():
                    output = self.model(input)
                    loss = self.criterion(output, target) / iter_size
            else:
                output = self.model(input)
                loss = self.criterion(output, target) / iter_size
            # loss /= batch_size

            if training:
                loss.backward()

            meter['loss'] += loss.data.cpu().numpy() + 0.0
            output = torch.sigmoid(output)
            for name, func in metrics:
                acc = func(output.contiguous(), target.contiguous())
                meter[name] += acc.data.cpu().numpy() / iter_size

            outputs.append(output.data)


        if training:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.)
            self.optimizer.step()
        return meter, torch.cat(outputs, dim=0)

class MetricsCollection:
    def __init__(self):
        self.stop_training = False
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}


class PytorchTrain:
    """
    class for training process and callbacks in right places
    Loads in a distinct model for each fold
    """
    def __init__(self, estimator: Estimator, fold, metrics, callbacks=None, hard_negative_miner=None):
        self.fold = fold
        self.estimator = estimator
        self.metrics = metrics

        self.devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        if os.name == 'nt':
            self.devices = ','.join(str(d + 5) for d in map(int, self.devices.split(',')))

        self.hard_negative_miner = hard_negative_miner
        self.metrics_collection = MetricsCollection()

        self.estimator.resume("fold" + str(fold) + "_checkpoint.pth")

        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(self)

    def _run_one_epoch(self, epoch, loader, training=True):
        avg_meter = defaultdict(float)

        pbar = tqdm(enumerate(loader), total=len(loader), desc="Epoch {}{}".format(epoch, ' eval' if not training else ""), ncols=0)
        for i, data in pbar:
            self.callbacks.on_batch_begin(i)

            meter, ypreds = self._make_step(data, training)
            for k, val in meter.items():
                avg_meter[k] += val

            if training:
                if self.hard_negative_miner is not None:
                    self.hard_negative_miner.update_cache(meter, data)
                    if self.hard_negative_miner.need_iter():
                        self._make_step(self.hard_negative_miner.cache, training)
                        self.hard_negative_miner.invalidate_cache()

            pbar.set_postfix(**{k: "{:.5f}".format(v / (i + 1)) for k, v in avg_meter.items()})

            self.callbacks.on_batch_end(i)
            # SUPER IMPORTANT for MixedReadingImageProviders
            # This allows the mixed providers to switch the dataset it's providing from
            # loader.dataset.end_batch()
        return {k: v / len(loader) for k, v in avg_meter.items()}

    def _run_one_epoch_mixed(self, epoch, loaders, training=True):
        """Switch between loaders for training, lord help me for validation...."""
        pbars = []
        counts = []
        for idx, load in enumerate(loaders):
            pbars.append(tqdm(enumerate(loader), total=len(loader), desc="Epoch {}{}-{}".format(epoch, ' eval' if not training else "", idx), ncols=0))
            counts.append(0)

        to_choose = [pbars[i] if counts[i] < len(pbars[i]) for i in range(len(pbars))]
        actual_idx = 0
        while len(to_choose) > 0:
            curr_pbar = np.random.choice(to_choose)
            data = next(curr_pbar)

            self.callbacks.on_batch_begin(actual_idx)

            meter, ypreds = self._make_step(data, training)
            for k, val in meter.items():
                avg_meter[k] += val

            if training:
                if self.hard_negative_miner is not None:
                    self.hard_negative_miner.update_cache(meter, data)
                    if self.hard_negative_miner.need_iter():
                        self._make_step(self.hard_negative_miner.cache, training)
                        self.hard_negative_miner.invalidate_cache()

            pbar.set_postfix(**{k: "{:.5f}".format(v / (actual_idx + 1)) for k, v in avg_meter.items()})

            self.callbacks.on_batch_end(actual_idx)
            # SUPER IMPORTANT for MixedReadingImageProviders
            # This allows the mixed providers to switch the dataset it's providing from
            # loader.dataset.end_batch()

    def _make_step(self, data, training):
        images = data['image']
        ytrues = data['mask']

        meter, ypreds = self.estimator.make_step_itersize(images, ytrues, training, self.metrics)

        return meter, ypreds

    def switch_ds():
        if self.mixed_ds():
            self.ds_idx = np.random.choice(self.)
    def fit(self, train_loader, val_loader, nb_epoch):
        self.callbacks.on_train_begin()

        for epoch in range(self.estimator.start_epoch, nb_epoch):
            self.callbacks.on_epoch_begin(epoch)
            if self.estimator.lr_scheduler is not None:
                self.estimator.lr_scheduler.step(epoch)

            # Flip between saving the weight coefficients when training & when evaluating
            self.estimator.model.train()
            self.metrics_collection.train_metrics = self._run_one_epoch(epoch, train_loader, training=True)
            self.estimator.model.eval()
            self.metrics_collection.val_metrics = self._run_one_epoch(epoch, val_loader, training=False)

            self.callbacks.on_epoch_end(epoch)

            if self.metrics_collection.stop_training:
                break

        self.callbacks.on_train_end()


def train(ds, folds, config, num_workers=0, transforms=None, skip_folds=None, num_channels_changed=False):
    """
    here we construct all needed structures and specify parameters
    """
    os.makedirs(os.path.join(config.results_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, 'logs'), exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(folds):
        # train_idx = [train_idx[0]]
        if skip_folds and fold in skip_folds:
            continue

        save_path = os.path.join(config.results_dir, 'weights', config.folder)
        model = models[config.network](num_classes=1, num_channels=config.num_channels)
        estimator = Estimator(model, optimizers[config.optimizer], losses[config.loss], save_path,
                              iter_size=config.iter_size, lr=config.lr, num_channels_changed=num_channels_changed)

        callbacks = [
            StepLR(config.lr, num_epochs_per_decay=30, lr_decay_factor=0.1),
            ModelSaver(1, ("fold"+str(fold)+"_best.pth"), best_only=True),
            CheckpointSaver(1, ("fold"+str(fold)+"_checkpoint.pth")),
            # EarlyStopper(10),
            TensorBoard(os.path.join(config.results_dir, 'logs', config.folder, 'fold{}'.format(fold)))
        ]

        # hard_neg_miner = HardNegativeMiner(rate=10)
        metrics = [('dice', dice), ('bce', binary_cross_entropy), ('dice round', dice_round)]
        # metrics = []
        trainer = PytorchTrain(estimator,
                               fold=fold,
                               metrics=metrics,
                               callbacks=callbacks,
                               hard_negative_miner=None)

        if type(ds) == MixedReadingImageProvider:
            ds.set_batch_size(config.batch_size)
            shuffle = False
        train_loader = PytorchDataLoader(TrainDataset(ds, train_idx, config, transforms=transforms),
                                         batch_size=config.batch_size,
                                         shuffle=shuffle,
                                         drop_last=True,
                                         num_workers=num_workers,
                                         pin_memory=True)
        val_loader = PytorchDataLoader(ValDataset(ds, val_idx, config, transforms=None),
                                       batch_size=config.batch_size,
                                       shuffle=shuffle,
                                       drop_last=False,
                                       num_workers=num_workers,
                                       pin_memory=True)

        trainer.fit(train_loader, val_loader, config.nb_epoch)
