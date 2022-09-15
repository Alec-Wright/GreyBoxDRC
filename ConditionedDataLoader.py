import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import json
import torch
import os


# Dataset class for loading the LA-2A compressor dataset
class AudioDataset(Dataset):
    def __init__(self, data_dir, set_name, segment_length_seconds: float = 2.5,
                 file_names=('target_138_LA2A_3c__0__0.wav',  'target_140_LA2A_3c__0__10.wav',
                             'target_142_LA2A_3c__0__20.wav', 'target_144_LA2A_3c__0__30.wav',
                             'target_146_LA2A_3c__0__40.wav', 'target_148_LA2A_3c__0__50.wav',
                             'target_150_LA2A_3c__0__60.wav', 'target_152_LA2A_3c__0__70.wav',
                             'target_154_LA2A_3c__0__80.wav', 'target_156_LA2A_3c__0__90.wav',
                             'target_158_LA2A_3c__0__100.wav')):
        self.inputs = []
        self.targets = []
        self.p_red = []
        self.set = set_name

        for file in file_names:
            target_file = file
            input_file = 'input' + file[6:10] + '_.wav'
            inp_data, self.fs = torchaudio.load(os.path.join(data_dir, set_name, input_file), channels_first=False)
            tgt_data, self.fs_t = torchaudio.load(os.path.join(data_dir, set_name, target_file), channels_first=False)
            assert (self.fs == self.fs_t)

            self.inputs.append(inp_data)
            self.targets.append(tgt_data)
            # Get peak reduction setting from the file name for conditioning data
            self.p_red.append(torch.tensor(float(file.split('__')[-1].split('.')[0])))

        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)
        self.p_orig = torch.stack(self.p_red)

        # Normalise peak_reduction values to range -1 to 1
        self.p_max = torch.max(self.p_orig)
        self.p_min = torch.min(self.p_orig)
        self.p_red = ((self.p_orig - self.p_min) / (self.p_max - self.p_min)).unsqueeze(1)
        self.p_red = (self.p_red*2) - 1

        data_length = self.inputs.shape[1]

        self.segment_length_samples = int(segment_length_seconds*self.fs)

        self.num_segments = int(len(self.p_red)*data_length / self.segment_length_samples)
        self.num_conds = len(self.p_red)
        self.segs_per_cond = self.num_segments//self.num_conds

        self.p_red_name = [str(x.item()) for x in self.p_orig]

    def __getitem__(self, index):
        cond_val = index//self.segs_per_cond
        index = index % self.segs_per_cond

        start = index * self.segment_length_samples
        stop = (index + 1) * self.segment_length_samples
        return self.inputs[cond_val, start:stop, :], self.targets[cond_val, start:stop, :], self.p_red[cond_val]

    def __len__(self):
        return self.num_segments


class AudioDataModule(pl.LightningDataModule):
    def __init__(
            self,
            device_name: str = "LA-2A",
            data_dir: str = "SignalTrainData",
            segment_length_seconds: float = 2.5,
            segment_length_seconds_v: int = 5,
            batch_size: int = None,
            batch_size_v: int = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.segment_length_samples = segment_length_seconds
        self.batch_size = batch_size
        self.segment_length_samples_v = segment_length_seconds_v
        self.batch_size_v = batch_size_v
        self.device_name = device_name
        self.datasets = {}

    def setup(self):

        def make_dataset(data_dir, set_name,  segment_length):
            return AudioDataset(data_dir, set_name, segment_length)

        self.datasets["train"] = make_dataset(self.data_dir, "train", self.segment_length_samples)
        self.datasets["val"] = make_dataset(self.data_dir, "val",  self.segment_length_samples_v)
        self.datasets["test"] = make_dataset(self.data_dir, "test",  self.segment_length_samples_v)

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], batch_size=self.batch_size_v, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size_v, shuffle=False, num_workers=0)


def json_load(file_name, dir_name=''):
    dir_name = [dir_name] if ((type(dir_name) != list) and (dir_name)) else dir_name
    file_name = file_name + '.json' if not file_name.endswith('.json') else file_name
    full_path = os.path.join(*dir_name, file_name)
    with open(full_path) as fp:
        return json.load(fp)


def load_config(args):
    # Load the configs and write them onto the args dictionary, this will add new args and/or overwrite old ones
    configs = json_load(args.load_config, args.config_location)
    for parameters in configs:
        args.__setattr__(parameters, configs[parameters])
    return args


def tbptt_split_batch(batch, warmup, split_size):
    total_steps = batch[0].shape[1]
    splits = [[x[:, :warmup, :] for i, x in enumerate(batch[0:2])]]
    splits[0].append(batch[2])
    for t in range(warmup, total_steps, split_size):
        batch_split = [x[:, t: t + split_size, :] for i, x in enumerate(batch[0:2])]
        batch_split.append(batch[2])
        splits.append(batch_split)
    return splits

