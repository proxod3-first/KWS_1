# © 2020 JumpML
import os
import torchaudio
import torch
import glob
import matplotlib.pyplot as plt

def get_fileList(dataset_path, filename):
    '''
    Given a file (filename) with a column of filenames, return a list of those files
    dataset_path: directory
    filename: file with column of file names
    '''
    with open(os.path.join(dataset_path, filename)) as f:
        fileList = f.read().split("\n")

    fileList = [os.path.join(dataset_path, fname) for fname in fileList]
    return(fileList)

def generate_background_files(dataset_path, num_samples=16000):

    background_source_files = glob.glob(os.path.join(
        dataset_path, "_background_noise_", "*.wav"))

    targetDir = os.path.join(dataset_path, "background")
    # Generate Background Files
    print('Generate 1s background files:\n')
    os.makedirs(targetDir, exist_ok=True)
    for f in background_source_files:
        waveform, sr = torchaudio.load(f)
        split_waveforms = torch.split(waveform, num_samples, dim=1)
        for idx, split_waveform in enumerate(split_waveforms):
            torchaudio.save(os.path.join(
                targetDir, f'{hash(waveform)}_nohash_{idx}.wav'), split_waveform, sample_rate=sr)

    background_target_files = glob.glob(
        os.path.join(targetDir, "*.wav"))
    return(background_target_files)

def get_filenames(path, searchstr='*', extension='wav'):
    return glob.glob(os.path.join(path, f'{searchstr}.{extension}'))

def plot_waveform(filePath):
    [waveform, sr] = torchaudio.load(filePath)
    plt.plot(waveform.view(-1).numpy())
    plt.show()

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')