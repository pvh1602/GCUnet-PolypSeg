import numpy as np
import torch
import torch.nn as nn 
import tqdm

class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, device, data_loader, args=None,
                metric=None, lr_scheduler=None, warmup_schduler=None, scaler=None, n_epochs=100):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric = metric
        self.args = args
        self.device = device
        self.data_loader = data_loader
        self.n_epochs = n_epochs
        self.lr_scheduler = lr_scheduler
        self.warmup_schduler = warmup_schduler
        self.scaler = scaler
        # self.data, self.target = next(iter(self.data_loader))


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        self.model.train()
        if self.metric is not None:
            self.metric.reset()

        loop = tqdm.tqdm(self.data_loader, desc=f'Epoch {epoch}/{self.n_epochs}')

        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(self.device), target.float().to(self.device)

            
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.lr_scheduler is not None:
                # print("lr_scheduler")
                self.lr_scheduler.step()
            if self.warmup_schduler is not None:
                # print("warmup_schuler")
                self.warmup_schduler.dampen()

            self.metric.update(loss.item())
            loop.set_postfix(loss=loss.item())
            
        avg_loss = self.metric.show()
        self.args.logger.info(f'Epoch {epoch} \t Loss {avg_loss}')

        # exit()


    def train(self):
        loaded_epoch = 0
        if self.args.loaded_epoch is not None:
            loaded_epoch = self.args.loaded_epoch
        for epoch in range(loaded_epoch, self.n_epochs):
            self._train_epoch(epoch)
            self._save_checkpoint(epoch)



    def _save_checkpoint(self, epoch):
        print(f"Saving checkpoint at epoch {epoch} ...")
        if self.lr_scheduler is not None:
            state = {
                'epoch': epoch,
                'optimizer': self.optimizer.state_dict(),
                'model': self.model.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()
            }
        else:
            state = {
                'epoch': epoch,
                'optimizer': self.optimizer.state_dict(),
                'model': self.model.state_dict(),
            }
        torch.save(state, self.args.checkpoint_name)
        print(f'=> Saved checkpoint at {self.args.checkpoint_name}')



    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
