import argparse
import os
import glob
import soundfile as sf
import numpy as np

prsr = argparse.ArgumentParser(
    description='''This script does preprocessing to assemble a sub-dataset from the SignalTrain dataset directory''')

# arguments for the training/test data locations and file names and config loading
prsr.add_argument('--load_dir', '-ld',
                  help="path to the 'SignalTrain' directory, which contains the complete SignalTrain dataset"
                  , default='../SignalTrain')
prsr.add_argument('--save_dir', '-sd',
                  help="path to a directory where the dataset to be used during training will be saved"
                  , default='../dataset/')

args = prsr.parse_args()

if __name__ == "__main__":

    train_dir = os.path.join(args.save_dir, 'train')
    val_dir = os.path.join(args.save_dir, 'val')
    test_dir = os.path.join(args.save_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_seglen = 80
    val_seglen = 10
    test_seglen = 10
    prs = list(range(0, 105, 10))

    for subdir in ['Train', 'Test', 'Val']:
        for filename in glob.glob(os.path.join(args.load_dir, subdir, '*.wav')):
            init_path = os.path.join(*filename.split('/')[0:-1])
            name = filename.split('/')[-1]
            if name.__contains__('3c__0') and name.startswith('target') and\
                    int(name.split('__')[-1].split('.')[0]) in prs:

                target_name = name
                input_name = 'input' + target_name[6:10] + '_.wav'

                tgt, fs_t = sf.read(filename)
                inp, fs_i = sf.read(os.path.join(init_path, input_name))

                assert fs_i == fs_t
                assert tgt.shape == inp.shape

                # Some examples are slightly shorter in SignalTrain, so this makes them up to 20 minutes each
                if not len(inp)/fs_i == 1200:
                    tgt = np.append(tgt, np.zeros(fs_t*1200 - len(tgt)))
                    inp = np.append(inp, np.zeros(fs_t*1200 - len(inp)))

                startind = 0
                train_indata = np.empty(0)
                val_indata = np.empty(0)
                test_indata = np.empty(0)
                train_tadata = np.empty(0)
                val_tadata = np.empty(0)
                test_tadata = np.empty(0)

                while startind < len(inp):
                    train_indata = np.append(train_indata, inp[startind:startind + train_seglen*fs_i])
                    train_tadata = np.append(train_tadata, tgt[startind:startind + train_seglen*fs_i])
                    startind = startind + train_seglen*fs_i

                    val_indata = np.append(val_indata, inp[startind:startind + val_seglen*fs_i])
                    val_tadata = np.append(val_tadata, tgt[startind:startind + val_seglen*fs_i])
                    startind = startind + val_seglen*fs_i

                    test_indata = np.append(test_indata, inp[startind:startind + test_seglen*fs_i])
                    test_tadata = np.append(test_tadata, tgt[startind:startind + test_seglen*fs_i])
                    startind = startind + test_seglen*fs_i

                assert(len(train_tadata) == len(train_indata))
                assert(len(val_tadata) == len(val_indata))
                assert(len(test_tadata) == len(test_indata))

                sf.write(os.path.join(train_dir, input_name), train_indata, fs_i)
                sf.write(os.path.join(train_dir, target_name), train_tadata, fs_i)

                sf.write(os.path.join(val_dir, input_name), val_indata, fs_i)
                sf.write(os.path.join(val_dir, target_name), val_tadata, fs_i)

                sf.write(os.path.join(test_dir, input_name), test_indata, fs_i)
                sf.write(os.path.join(test_dir, target_name), test_tadata, fs_i)

