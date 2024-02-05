from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from torch import nn
from torch.nn import functional as F
import einops
import lightning as L
from torchmetrics.classification import MulticlassAccuracy, ConfusionMatrix
# from nnAudio.features.mel import MFCC
import matplotlib.pyplot as plt
import itertools
import numpy as np

class PatchEmbedding_for_audio(nn.Module):
    def __init__(self, time_window:int, frequency:int, patch_size_t:int, patch_size_f:int, embed_dim:int) -> None:
        super().__init__()

        self.T = time_window
        self.F = frequency
        self.d = embed_dim

        tf_dim = (self.T // patch_size_t) * (self.F // patch_size_f)

        self.positional_embedding = nn.Parameter(torch.rand(1, tf_dim, self.d))
        self.class_tokens = nn.Parameter(torch.rand(1, 1, self.d))

        self.patch_embeddings = nn.Conv2d(
            in_channels=1,
            out_channels=self.d,
            # kernel_size=(patch_size_t, patch_size_f),
            # stride=(patch_size_t, patch_size_f)
            kernel_size=(patch_size_f, patch_size_t),
            stride=(patch_size_f, patch_size_t)
        )

    def forward(self, audio):
        # try:
        #     # audio = einops.rearrange(audio, "b t f -> b 1 t f", t = self.T, f = self.F)
        #     audio = einops.rearrange(audio, "b f t -> b 1 f t", t = self.T, f = self.F)
        # except Exception:
        #     print(f"Что-то там с вашими размерностями, надо {self.T}x{self.F}, а у вас - {audio.shape}")
        
        # Применяем линейный слой
        patches = self.patch_embeddings(audio)
        # patches = einops.rearrange(patches, "b d tp fp -> b (tp fp) d")
        patches = einops.rearrange(patches, "b d fp tp -> b (fp tp) d")

        # Прибавляем позицонное кодирование
        patches = patches + self.positional_embedding.data

        # Добавляем токен класса
        b, tf, d = patches.shape
        class_tokens = einops.repeat(self.class_tokens.data, "() tf d -> b tf d", b=b)
        patches = torch.cat((patches, class_tokens), dim=1)

        return patches

class MLP(nn.Module):
    def __init__(self, in_features:int, hidden_features=None, out_features=None, drop=0.0, act_layer = nn.GELU()):
        super().__init__()

        if out_features is None:
            out_features = in_features
        
        if hidden_features is None:
            hidden_features = in_features

        # Linear Layers
        self.lin1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.lin2 = nn.Linear(
            in_features=hidden_features,
            out_features=out_features
        )

        # Activation(s)
        self.act = act_layer
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x):

        x = self.act(self.dropout(self.lin1(x)))
        x = self.act(self.lin2(x))

        return x

class Attention(nn.Module):
    def __init__(self, dim:int, num_heads:int, qkv_bias=False, attn_drop=0.0, out_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.soft = nn.Softmax(dim=-1) # Softmax по строкам матрицы внимания
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x):

        # Attention
        qkv_after_linear = self.qkv(x)
        qkv_after_reshape = einops.rearrange(qkv_after_linear, "b c (v h w) -> v b h c w", v=3, h=self.num_heads)
        q = qkv_after_reshape[0]
        k = qkv_after_reshape[1]
        k = einops.rearrange(k, "b h c w -> b h w c") # Транспонирование
        v = qkv_after_reshape[2]

        atten = self.soft(torch.matmul(q, k) * self.scale)
        atten = self.attn_drop(atten)
        out = torch.matmul(atten, v)
        out = einops.rearrange(out, "b h c w -> b c (h w)", h=self.num_heads)

        # Out projection
        x = self.out(out)
        x = self.out_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim:int, norm_type:str, num_heads:int, mlp_dim:int, qkv_bias=False, drop_rate=0.0):
        super().__init__()

        self.norm_type = norm_type

        # Normalization
        self.norm1 = nn.LayerNorm(
            normalized_shape=dim
        )
        self.norm2 = nn.LayerNorm(
            normalized_shape=dim
        )

        # Attention
        self.attention = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop_rate,
            out_drop=drop_rate
        )
        
        # MLP
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_dim
        )


    def forward(self, x):
        if self.norm_type == "prenorm":
            x_inner = self.norm1(x)
            # Attetnion
            x_inner = self.attention(x_inner)
            x = x_inner + x

            x_inner = self.norm2(x)
            # MLP
            x_inner = self.mlp(x_inner)
            x = x_inner + x
        
        if self.norm_type == "postnorm":
            x_inner = self.attention(x)
            x = x_inner + x
            x = self.norm1(x)
            x_inner = self.mlp(x)
            x = x_inner + x
            x =self.norm2(x)

        return x

class Transformer(nn.Module):
    def __init__(self, depth, dim, norm_type, num_heads, mlp_dim, qkv_bias=False, drop_rate=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, norm_type, num_heads, mlp_dim, qkv_bias, drop_rate) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ViT_audio(nn.Module):
    """ Vision Transformer with support for audio
    """
    def __init__(
            self, 
            time_window, frequency, patch_size_t, patch_size_f, embed_dim,
            num_classes, depth, num_heads, mlp_dim,
            norm_type,
            qkv_bias=False, drop_rate=0.0
        ):
        super().__init__()
        # Присвоение переменных
        # Path Embeddings, CLS Token, Position Encoding
        self.patch_emb = PatchEmbedding_for_audio(
            time_window = time_window,
            frequency = frequency,
            patch_size_t = patch_size_t,
            patch_size_f = patch_size_f,
            embed_dim = embed_dim
        )
        # Transformer Encoder
        self.transformer = Transformer(
            depth=depth,
            dim=embed_dim,
            norm_type=norm_type,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate
        )
        # Classifier
        self.head = MLP(
            in_features=embed_dim,
            out_features=num_classes,
            drop=drop_rate
        )

    def forward(self, x):
        x = self.patch_emb(x)
        x = self.transformer(x)
        x = self.head(x)
        return x

class ViT_Lightning(L.LightningModule):
    def __init__(
            self,
            time_window:int, frequency:int, patch_size_t:int, patch_size_f:int, embed_dim:int,
            num_classes:int, depth:int, num_heads:int, mlp_dim:int,
            norm_type:str,
            lr:float,
            qkv_bias=False, drop_rate=0.0,
            type_of_scheduler:str = "ReduceOnPlateau", patience_reduce:int = 5, factor_reduce:float=0.1, lr_coef_cycle:int = 2, total_num_of_epochs:int = 20,
            sample_rate:int = 16000, n_mffc:int = 128, n_mels:int = 128, n_fft:int = 480, hop_length:int=160,
            previous_model = None, need_mfcc = False
        ) -> None:
        super().__init__()
        if previous_model is None:
            self.vit_model = ViT_audio(
                time_window=time_window, frequency=frequency, patch_size_t=patch_size_t, patch_size_f=patch_size_f, embed_dim=embed_dim,
                num_classes=num_classes, depth=depth, num_heads=num_heads, mlp_dim=mlp_dim,
                norm_type=norm_type, qkv_bias=qkv_bias, drop_rate=drop_rate
            )
        else:
            self.vit_model = previous_model
        
        self.metric = MulticlassAccuracy(num_classes=num_classes)
        self.matrix = ConfusionMatrix(task = "multiclass", num_classes = num_classes)
        self.flag_cm = True
        self.num_classes = num_classes

        self.lr = lr
        self.type_of_scheduler = type_of_scheduler
        self.patience_reduce = patience_reduce
        self.factor_reduce = factor_reduce
        self.lr_coef_cycle = lr_coef_cycle
        self.total_num_of_epochs = total_num_of_epochs
        
        self.need_mfcc = need_mfcc
        # if self.need_mfcc:
        #     self.mfcc_layer = MFCC(
        #         sr=sample_rate,
        #         n_mfcc=n_mffc,
        #         n_mels=n_mels,
        #         n_fft=n_fft,
        #         hop_length=hop_length,
        #         # trainable_mel=True,
        #         # trainable_STFT=True
        #     )

        self.save_hyperparameters()
    
    def labels_translate(self, y):
        y_new = torch.zeros_like(y)
        for i in range(len(y)):
            match y[i]:
                case 30:
                    y_new[i] = 0 # up
                case 34:
                    y_new[i] = 1 # zero
                case 3:
                    y_new[i] = 2 # cat
                case 4:
                    y_new[i] = 3 # dog
                case 22:
                    y_new[i] = 4 # right
                case 11:
                    y_new[i] = 5 # go
                case 33:
                    y_new[i] = 6 # yes
                case 18:
                    y_new[i] = 7 # no
                case 28:
                    y_new[i] = 8 # three
                case 9:
                    y_new[i] = 9 # forward
                case 16:
                    y_new[i] = 10 # marvin
                case _:
                    y_new[i] = 11 # others
        return y_new

    def forward(self, x):
        return self.vit_model(x)
    
    def loss(self, y, y_hat):
        return F.cross_entropy(y, y_hat)

    def lr_scheduler(self, optimizer):
        if self.type_of_scheduler == "ReduceOnPlateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience_reduce, factor=self.factor_reduce)
            scheduler_out = {"scheduler": sched, "monitor": "val_loss"}
        if self.type_of_scheduler == "OneCycleLR":
            sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr * self.lr_coef_cycle, total_steps=self.total_num_of_epochs)
            scheduler_out = {"scheduler": sched}
        
        return scheduler_out

    def training_step(self, batch) -> STEP_OUTPUT:
        x, y = batch

        if self.need_mfcc:
            x = self.mfcc_layer(x)
        # x = einops.rearrange(x, "b f t -> b t f")

        out = self(x)[:,-1,:]
        # y = self.labels_translate(y)
        pred_loss = self.loss(out, y)
        
        self.log("train_loss", pred_loss)
        self.log("train_acc", self.metric(out, y))
        
        return pred_loss
    
    def validation_step(self, batch) -> STEP_OUTPUT:
        x, y = batch

        if self.need_mfcc:
            x = self.mfcc_layer(x)
        # x = einops.rearrange(x, "b f t -> b t f")

        out = self(x)[:,-1,:]
        # y = self.labels_translate(y)
        pred_loss = self.loss(out, y)

        self.log("val_loss", pred_loss)
        self.log("val_acc", self.metric(out, y))
        if self.flag_cm:
            self.conf_matrix = self.matrix(torch.softmax(out, dim=-1), y)
            self.flag_cm = False
        else:
            self.conf_matrix += self.matrix(torch.softmax(out, dim=-1), y)
        # print(self.conf_matrix)
    
    def test_step(self, batch) -> STEP_OUTPUT:
        pass
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        scheduler_dict = self.lr_scheduler(optimizer)
        return (
            {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
        )

class AudioConv(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.model_cnv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 20)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 8, kernel_size=(4, 10)),
            nn.BatchNorm2d(8)
        )

        self.flat = nn.Flatten()

        self.lin = nn.Sequential(
            nn.Linear(in_features=17280, out_features=128),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=35)
        )

        # self.mfcc_layer = MFCC(
        #     sr=16000,
        #     n_mfcc=40,
        #     n_mels=80,
        #     n_fft=480,
        #     hop_length=161
        # )

        self.metric = MulticlassAccuracy(num_classes=35)

        self.save_hyperparameters()
    
    def forward(self, x):
        x = self.model_cnv(x)
        x = self.flat(x)
        x = self.lin(x)
        return x

    def loss(self, y, y_hat):
        return F.cross_entropy(y, y_hat)

    def lr_scheduler(self, optimizer):
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=6e-4, total_steps=15)
        scheduler_out = {"scheduler": sched}
        return scheduler_out

    def training_step(self, batch) -> STEP_OUTPUT:
        x, y = batch

        x = self.mfcc_layer(x)
        x = einops.rearrange(x, "b f t -> b 1 f t")

        out = self(x)
        pred_loss = self.loss(out, y)
        
        self.log("train_loss", pred_loss)
        self.log("train_acc", self.metric(out, y))
        
        return pred_loss
    
    def validation_step(self, batch) -> STEP_OUTPUT:
        x, y = batch

        x = self.mfcc_layer(x)
        x = einops.rearrange(x, "b f t -> b 1 f t")

        out = self(x)
        pred_loss = self.loss(out, y)
        
        self.log("val_loss", pred_loss)
        self.log("val_acc", self.metric(out, y))
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.1)
        scheduler_dict = self.lr_scheduler(optimizer)
        return (
            {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
        )

class ConfMatrixLogging(L.Callback):
    def __init__(self, cls) -> None:
        super().__init__()
        self.cls = cls
    
    def make_img_matrix(self, matr):
        matr = matr.cpu()
        fig=plt.figure(figsize=(16, 8), dpi=80)
        plt.imshow(matr,  interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()

        tick_marks = np.arange(len(self.cls))
        plt.xticks(tick_marks, self.cls, rotation=45)
        plt.yticks(tick_marks, self.cls)

        fmt = 'd'
        thresh = matr.max() / 2.
        for i, j in itertools.product(range(matr.shape[0]), range(matr.shape[1])):
            plt.text(j, i, format(matr[i, j], fmt), horizontalalignment="center", color="white" if matr[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.show()
        return [fig]

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        trainer.logger.log_image(key="Validation Confusion Matrix", images=self.make_img_matrix(pl_module.conf_matrix))
        plt.close()
        pl_module.flag_cm = True