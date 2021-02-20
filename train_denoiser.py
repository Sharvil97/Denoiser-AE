import torch
from argparse import Namespace
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from data import ImgAugTransform, RandomCrop, ToTensor, Rescale, DirtyDocumentsDataset
from loss import loss_func
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl 
from model import AEDenoiser
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

seed_everything(5)

class Denoiser(pl.LightningModule):

    def __init__(self, hparams):
        super(Denoiser, self).__init__()
        self.hparams = hparams
        self.model = AEDenoiser()

        self.train_image = None
        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        train_image, clean_image = batch['train_image'], batch['clean_image']
        self.train_image = train_image
        output = self.forward(train_image)
        loss = loss_func(output, clean_image)
        run.log('loss', np.float(loss))
        tqdm_dict = {'d_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr, eps=1e-08)
    
    def prepare_data(self):
        # transforms for images
        composed = transforms.Compose([ImgAugTransform(), RandomCrop((256,256),0.25),Rescale((256,540)),ToTensor()])
        self.train_dataset = DirtyDocumentsDataset(dirty_dir=self.hparams.root_dir_noisy, clean_dir=self.hparams.root_dir_clean, transform=composed)
        return self.train_dataset

    
    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size= self.hparams.batch_size)
    

    def on_epoch_end(self):
        z = self.train_image
        # match gpu device (or keep as cpu)
        if self.on_gpu:
            z = z.cuda(self.last_imgs.device.index)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'Denoised images', grid, self.current_epoch)


if __name__ == "__main__":


    args = {
        'batch_size': 32,
        'lr': 0.0005,
        'root_dir_noisy':,
        'root_dir_clean':,
    }

    hparams = Namespace(**args)

    checkpoint_callback = ModelCheckpoint(
    filepath=r"./outputs/models",
    save_top_k=True,
    verbose=True,
    mode='min',
    prefix='')

    logger = TensorBoardLogger("./outputs/tb_log", name="aeDenoiser")

    ae = Denoiser(hparams)
    trainer = Trainer(gpus=2, distributed_backend='dp', logger=logger,\
         auto_lr_find=True, log_gpu_memory='all', default_root_path=r"./outputs", max_epochs=1000)
    trainer.fit(ae)

# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/