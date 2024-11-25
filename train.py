import torch
import torchvision
import torchmetrics

from pathlib import Path

from data_setup import data_download, get_dataloaders
from vit import ViT
from engine import train
from utils import create_writer, save_model

# Hyperparameters
COLOR_CHANNELS = 3
HEIGHT_WIDTH = (224, 224)

BATCH_SIZE = 32

PATCH_SIZE = (16, 16)
NUM_PATCHES = int((HEIGHT_WIDTH[0] / PATCH_SIZE[0]) ** 2)

EMBED_DIMS = 768
NUM_ATTN_HEADS = 12
RATIO_HIDDEN_MLP = 4
NUM_ENC_BLOCKS = 12

NUM_EPOCHS = 10

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Donwload data if it hasn't been downloaded
data_path = Path('./data/')
train_path = data_path / 'desserts' / 'train'
test_path = data_path / 'desserts' / 'test'

if not data_path.is_dir():
    data_download()

# Get dataloaders
train_dataloader, test_dataloader, class_labels = get_dataloaders(train_path=train_path,
                                                                  test_path=test_path,
                                                                  batch_size=BATCH_SIZE)

# Experiments
model_names = ['vit_train_fs', 'vit_finetune']
lrs = [0.005, 0.001]

# Loss function, Accuracy
loss_function = torch.nn.CrossEntropyLoss()
accuracy_function = torchmetrics.Accuracy(task='multiclass', num_classes=len(class_labels))

# Experimentation Loop
for model_name in model_names:
    for lr in lrs:
        print(f"[INFO] Model: {model_name}")
        print(f"[INFO] Learning Rate: {lr}")
        
        # Model to use
        if model_name == 'vit_trian_fs':
            model = ViT(in_channels=COLOR_CHANNELS,
                        out_dims=len(class_labels),
                        patch_size=PATCH_SIZE,
                        num_patches=NUM_PATCHES,
                        embed_dims=EMBED_DIMS,
                        num_attn_heads=NUM_ATTN_HEADS,
                        ratio_hidden_mlp=RATIO_HIDDEN_MLP,
                        num_encoder_blocks=NUM_ENC_BLOCKS)
        elif model_name == 'vit_finetune':
            model = torchvision.models.vit_b_16(weights = torchvision.models.ViT_B_16_Weights.DEFAULT)
            
            # Changing the classifier head
            model.heads = torch.nn.Linear(in_features=EMBED_DIMS,
                                          out_features=len(class_labels))
            
            # Stop grad tracking for other layers
            for param in model.conv_proj.parameters():
                param.requires_grad = False
            for param in model.encoder.parameters():
                    param.requires_grad = False
        
        # Optimizer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        
        # Writer (tensorboard)
        writer = create_writer(model_name=model_name,
                               experiment_name=str(model_name) + "_" + str(lr))
        
        # Train
        results = train(num_epochs=NUM_EPOCHS,
                        model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_function=loss_function,
                        optimizer=optimizer,
                        accuracy_function=accuracy_function,
                        device=device,
                        writer=writer)
        
        # Save the model
        save_model(model=model,
                   model_name="model_" + str(model_name) + "_lr_" + str(lr),
                   target_dir='./models/')
        
        # Cuda memory management
        del model
        torch.cuda.empty_cache()
        
        print("-"*50 + '\n')