from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch
from torch.utils.data import DataLoader, Dataset
# from nnAudio.features.mel import MelSpectrogram
import matplotlib.pyplot as plt
import einops
# from nnAudio.features.mel import MFCC
from typing import Literal
from torch.nn.utils.rnn import pad_sequence
import lightning as L
from kws.preprocessing_data.preproccesing import SpeechCommandsData

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, dest, subset: str = None):
        super().__init__(dest, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

class AudioDataset(Dataset):
    def __init__(
            self, 
            destination, 
            set_type:Literal["training", "validation", "testing"] | None, 
            audio_rate:int=16000,
            labels = None
        ) -> None:
        super().__init__()
        self.audio_set = SubsetSC(dest=destination, subset=set_type)
        self.audio_rate = audio_rate

        if labels is None:
            self.labels = sorted(list(set(datapoint[2] for datapoint in self.audio_set)))
        else:
            self.labels = labels
    
    def __len__(self):
        return len(self.audio_set)

    def ensure_lenght(self, wave):
        if wave.shape[1] != self.audio_rate:
            wave = torch.cat((wave, torch.zeros((1, self.audio_rate - wave.shape[1]))), dim=1)
        return wave
    
    def key_translate(self, label_name):
        return torch.tensor(self.labels.index(label_name))

    def __getitem__(self, index):
        waveform, sample_rate, label, speaker_id, utterance_number = self.audio_set[index]
        waveform = self.ensure_lenght(waveform)
        label = self.key_translate(label)
        return waveform, label

class Audio_DataModule(L.LightningDataModule):
    def __init__(self, data_destination, batch_size=32, audio_rate:int=16000, labels=None) -> None:
        super().__init__()

        self.destination = data_destination
        self.batch_size = batch_size
        self.audio_rate = audio_rate
        self.labels = labels
    
    def prepare_data(self) -> None:
        AudioDataset(
            destination=self.destination, 
            set_type=None, 
            audio_rate=self.audio_rate, 
            labels=self.labels
        )
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_set = AudioDataset(self.destination, "training", self.audio_rate, self.labels)
            self.val_set = AudioDataset(self.destination, "validation", self.audio_rate, self.labels)
        if stage == "test" or stage is None:
            self.test_set = AudioDataset(self.destination, "testing", self.audio_rate, self.labels)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_set,
            shuffle=True,
            batch_size=self.batch_size
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_set,
            shuffle=False,
            batch_size=self.batch_size
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_set,
            shuffle=False,
            batch_size=self.batch_size
        )

class NewEra_AudioDataModule(L.LightningDataModule):
    def __init__(self, data_destination, batch_size, n_mfcc, hop_length) -> None:
        super().__init__()

        self.data_destination = data_destination
        self.batch_size = batch_size
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
    
    def prepare_data(self) -> None:
        self.sc_data = SpeechCommandsData(
            path=self.data_destination, 
            train_bs=self.batch_size, 
            test_bs=self.batch_size, 
            val_bs=self.batch_size, 
            n_mels=self.n_mfcc,
            hop_length=self.hop_length
        )
        
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            pass
        if stage == "test" or stage is None:
            pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.sc_data.train_loader
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.sc_data.val_loader
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.sc_data.test_loader