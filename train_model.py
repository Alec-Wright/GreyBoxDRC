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
    # If a load_config argument was provided, construct the file path to the config file
    args = model_configs.main(int(args.model_config))
