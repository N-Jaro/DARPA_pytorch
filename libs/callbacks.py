from pytorch_lightning.callbacks import Callback
import mlflow

class ValidPatchRateCallback(Callback):
    def __init__(self, initial_patch_rate=1.0, rate_decay=0.005, dynamic=False):
        super().__init__()
        self.initial_patch_rate = initial_patch_rate
        self.rate_decay = rate_decay
        self.dynamic = dynamic

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if self.dynamic or epoch == 0:
            new_patch_rate = max(0, self.initial_patch_rate - epoch * self.rate_decay)
            
            # Update datasets in the data module with the new valid_patch_rate
            trainer.datamodule.update_datasets(valid_patch_rate=new_patch_rate)

            # Reinitialize the dataloaders
            trainer.reset_train_dataloader()
            trainer.reset_val_dataloader()

            # Log the new valid_patch_rate
            pl_module.log('valid_patch_rate', new_patch_rate)
            mlflow.log_metric('valid_patch_rate', new_patch_rate)
