import os
import torch
import torch.utils
import torch.utils.data
from torchvision.transforms import ToTensor
import lightning as L

from config import Config
from model import ProSRModel
from dataset_greedy_strategy import GreedyPSORDataset

from utils.meter import Meter

class Lightning(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.model = ProSRModel(config)
        
        self.loss_meter = Meter()

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = out["loss"]

        self.loss_meter.update(loss)
        self.log_dict(self.loss_meter.avg(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        total_loss = sum(loss.values())
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch)
        prediction = out["prediction"]
        # print(prediction)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__=="__main__":
    import os
    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    config = Config()
    model = Lightning(config)

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=config.check_val_every_n_epoch
    )
    trainer.fit(
        model=model, 
        train_dataloaders=torch.utils.data.DataLoader(
            dataset=ProSORDataset(config, split="train"),
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=8,
            drop_last=True
        ),
        val_dataloaders=torch.utils.data.DataLoader(
            dataset=ProSORDataset(config, split="val"),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=4
        )
    )
    # torch.utils.data.DataLoader()