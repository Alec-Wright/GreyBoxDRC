import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.io.wavfile import write
import argparse
import pytorch_lightning as pl
import re
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from Model import model_configs
from Model import CompModel
from DataFuncs import ConditionedDataLoader

prsr = argparse.ArgumentParser(
    description='''This script trains a hybrid neural network and DSP model to emulate an analog compressor''')

# arguments for the training/test data locations and file names and config loading
prsr.add_argument('--model_config', '-mc',
                  help="File path, to a JSON config file, arguments listed in the config file will replace the defaults"
                  , default=1)
prsr.add_argument('--res_dir', '-rd',
                  help="name of the results directory", default='Results')

args = prsr.parse_args()

if __name__ == "__main__":
    # Create name for save directory
    save_dir = 'Model_' + str(args.model_config)

    # If a load_config argument was provided, construct the file path to the config file
    args = model_configs.main(int(args.model_config))

    gpus = 1 if torch.cuda.is_available() else 0

    # Create and set up dataloader
    data = ConditionedDataLoader.AudioDataModule(batch_size=30, batch_size_v=24,
                                                 device_name=args['data_set'], data_dir=args.pop('data_dir'))
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    test_dataloader = data.test_dataloader()

    # Create model
    model = CompModel.CompModel(**args, cond_name=val_dataloader.dataset.p_red_name)

    # Name log dir and create logger and early stopping callback
    logs_dir = 'logs_dev_' + args['data_set']
    early_stopping = EarlyStopping('val_loss_' + args['loss_func'], patience=5)
    checkpoint_callback = ModelCheckpoint(dirpath=logs_dir + '/' + save_dir, every_n_epochs=1, save_top_k=-1)
    logger = TensorBoardLogger(save_dir=logs_dir, version=1, name=save_dir)
    logger.sub_logger = SummaryWriter(logger.log_dir)

    # Create trainer object
    trainer = pl.Trainer(max_epochs=50, val_check_interval=0.25,
                         callbacks=[early_stopping, checkpoint_callback], logger=logger, gpus=gpus,
                         num_sanity_val_steps=-1)
    # Train model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_path = trainer.checkpoint_callback.best_model_path
    with open(logs_dir + '/' + save_dir + '/version_1/best_model_path.txt', 'w') as f:
        f.write(best_path)

    trainer.test(model=model, dataloaders=test_dataloader)

