# © 2020 JumpML
import torch
import torchaudio
from collections import defaultdict
import os
import shutil
import glob
from kws.preprocessing_data.utils import get_fileList, generate_background_files
from sklearn.model_selection import train_test_split

KNOWN_COMMANDS = ["yes",
                  "no",
                  "up",
                  "down",
                  "left",
                  "right",
                  "on",
                  "off",
                  "stop",
                  "go",
                  "background"]

DEFAULT_TRANSFORM = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=320, hop_length=161, n_mels=64),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80))

class SpeechCommandsData:
    def __init__(self, path='.', train_bs=64, test_bs=256, val_bs=64, n_mels=64, hop_length=161):

        # Setup transforms (separate for train, test and val if necessary)
        self.transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=320, hop_length=hop_length, n_mels=n_mels),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80))

        self.train_transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=320, hop_length=hop_length, n_mels=n_mels),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=int(n_mels*0.2)),
            torchaudio.transforms.TimeMasking(time_mask_param=int(0.2 * 16000/160)),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80))



        # Cleanup background files if needed
        backgroundDir = os.path.join(
            path, 'SpeechCommands', 'speech_commands_v0.02', 'background')
        if (os.path.isdir(backgroundDir)):
            print(
                'Found existing background directory. Removing files in that directory.')
            shutil.rmtree(backgroundDir)

        # Create separate datasets (or filelist iterators) for train, test and val
        print('Initialize/download SpeechCommandsDataset....\n')
        self.train_dataset = SpeechCommandsDataset(
            path=path, transform=self.train_transform)
        self.val_dataset = SpeechCommandsDataset(
            path=path, transform=self.transform)
        self.test_dataset = SpeechCommandsDataset(
            path=path, transform=self.transform)

        self.dataset_len = len(self.train_dataset)
        print(f'SpeechCommands Dataset Size: {self.dataset_len}\n')

        # Generate background files: creates files with 1 sec duration
        self.background_fileList = generate_background_files(
            self.train_dataset.dataset._path)
        print(f'Background files generated: {len(self.background_fileList)}\n')

        # Get validation and test file list from file
        self.val_fileList = get_fileList(self.val_dataset.dataset._path, "validation_list.txt")
        # print(len(self.val_fileList))
        # print(self.val_fileList[:5])
        self.test_fileList = get_fileList(self.test_dataset.dataset._path, "testing_list.txt")
        # print(len(self.test_fileList))
        # print(self.test_fileList[:5])

        self.val_ratio = len(self.val_fileList) / self.dataset_len
        self.test_ratio = len(self.test_fileList) / self.dataset_len
        self.train_ratio = 1.0 - self.val_ratio - self.test_ratio

        # Filter out files: modify _walker
        print(f'Extracting training dataset files...')
        self.train_dataset.dataset._walker = list(
            filter(lambda x: x not in self.val_fileList
                   and x not in self.test_fileList,
                   self.train_dataset.dataset._walker)
        )
        print(f'Train dataset extracted: {len(self.train_dataset)} files \n')
        print(f'Extracting test and val dataset files...')
        self.val_dataset.dataset._walker = list(
            filter(lambda x: x in self.val_fileList,
                   self.val_dataset.dataset._walker))
        print(f'Validation dataset extracted: {len(self.val_dataset)} files')
        self.test_dataset.dataset._walker = list(
            filter(lambda x: x in self.test_fileList,
                   self.test_dataset.dataset._walker))
        print(f'Test dataset extracted: {len(self.test_dataset)} files')
        
        print("Не хотите по-хорошему... Попробуем по-плохому")
        if len(self.val_dataset) == 0:
            temp = self.train_dataset.dataset._walker
            print("Общая выборка: ", len(temp))
            # print("Самый тяжёлый процесс начался...")
            self.train_dataset.dataset._walker, temp_valtest = train_test_split(temp, test_size=0.2)
            # print("Ура, он закончился")
            del temp
            print("Тренировочные данные: ", len(self.train_dataset.dataset._walker))
            # print("Процесс чуть полегче начался...")
            self.val_dataset.dataset._walker, self.test_dataset.dataset._walker = train_test_split(temp_valtest, test_size=0.5)
            # print("Ура, и он закончился")
            del temp_valtest
            print("Валидационные данные: ", len(self.val_dataset.dataset._walker))
            print("Тестовые данные: ", len(self.test_dataset.dataset._walker))

        # Add background files to walker
        idx_train = int(self.train_ratio * len(self.background_fileList))
        self.train_dataset.dataset._walker += self.background_fileList[:idx_train]
        idx_val = idx_train + \
            int(self.val_ratio * len(self.background_fileList))
        self.val_dataset.dataset._walker += self.background_fileList[idx_train:idx_val]
        idx_test = idx_val + int(self.test_ratio *
                                 len(self.background_fileList))
        self.test_dataset.dataset._walker += self.background_fileList[idx_val:idx_test]

        # Create Dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=train_bs, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=test_bs, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=val_bs, shuffle=False)


def preprocess_waveform(waveform, length=16000, transform=DEFAULT_TRANSFORM, padLR=False):
    padding = int((length - waveform.shape[1]))
    if padLR:
        waveform = torch.roll(torch.nn.functional.pad(waveform, (0, padding)), padding // 2)
    else:
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    features = transform(waveform)
    return(features)

def convert_word_to_class(word):
    # probably a better way to do this instead of computing on every call
    word2num = defaultdict(lambda: len(KNOWN_COMMANDS)-1)
    for num, command in enumerate(KNOWN_COMMANDS):
        word2num[command] = num

    return(word2num[word])

def preprocess_file(filePath, duration=1, transform=DEFAULT_TRANSFORM, padLR=False):
    waveform, sr = torchaudio.load(filePath)
    if sr == 16000:
        return(preprocess_waveform(waveform,int(duration*sr),transform,padLR))
    else:
        print("Error: Sample Rate must be 16000")
        return(-1)

def get_file_features(filePath, padLR=True):
    label = convert_word_to_class(filePath.split('_')[4]) # gets the label in the filename
    X = preprocess_file(filePath,padLR=padLR)
    X = torch.unsqueeze(X,0)
    return (X, label)

class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, path='.', transform=None):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(path,
                                                          url='speech_commands_v0.02',
                                                          folder_in_archive='SpeechCommands',
                                                          download=True)
        self.transform = transform

        # unknown word results in a default value of len(KNOWN_COMMANDS)
        self.word2num = defaultdict(lambda: len(KNOWN_COMMANDS)-1)
        for num, command in enumerate(KNOWN_COMMANDS):
            self.word2num[command] = num

    def __getitem__(self, index):
        (waveform, sample_rate, label, _, _) = self.dataset[index]
        # pad every waveform to 1 sec in samples
        # padding = int((sample_rate - waveform.shape[1]))
        # waveform = torch.nn.functional.pad(waveform, (0, padding))
        # features = self.transform(waveform)
        features = preprocess_waveform(waveform, sample_rate, self.transform)
        label = self.word2num[label]

        return features, label

    def __len__(self):
        return len(self.dataset)
