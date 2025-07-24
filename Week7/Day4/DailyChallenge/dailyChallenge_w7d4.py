import numpy as np
import pandas as pd
import random
import string

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from sklearn.metrics import roc_auc_score

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # device move
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # device move
else:
    device = torch.device("cpu")

TRAIN_PATH = "train_essays.csv"
TEST_PATH = "test_essays.csv"
PROMPT_PATH = "train_prompts.csv"

src_train = pd.read_csv(TRAIN_PATH)
src_prompt = pd.read_csv(PROMPT_PATH)
src_sub = pd.read_csv(TEST_PATH)

# Model preparation
tokenizer_save_path = "bert_tokenizer"
model_save_path = "bert_model"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
pretrained_model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)  # device move
embedding_model = pretrained_model.bert.to(device)  # device move

train_batch_size = 32
test_batch_size = 64
lr = 0.0002
beta1 = 0.5
nz = 100
num_epochs = 5
num_hidden_layers = 6
train_ratio = 0.8

class GANDAIGDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

all_num = len(src_train)
train_num = int(all_num * train_ratio)
test_num = all_num - train_num

train_set = src_train.iloc[:train_num]
test_set = pd.concat([src_train.iloc[train_num:]]).reset_index(drop=True)

train_dataset = GANDAIGDataset(train_set["text"].tolist(), train_set["generated"].tolist())
test_dataset = GANDAIGDataset(test_set["text"].tolist(), test_set["generated"].tolist())

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

config = BertConfig(num_hidden_layers=num_hidden_layers)

class Generator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 256 * 128)
        self.conv_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256 * 128, 768),
        )
        self.bert_encoder = BertEncoder(config)

    def forward(self, x):
        x = self.fc(x)
        x = self.conv_net(x)
        x = self.bert_encoder(x.unsqueeze(1)).last_hidden_state
        return x

class SumBertPooler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sum_hidden = hidden_states.sum(dim=1)
        sum_mask = sum_hidden.sum(1).unsqueeze(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_hidden / sum_mask
        return mean_embeddings

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_encoder = BertEncoder(config)
        self.bert_encoder.layer = nn.ModuleList([
            layer for layer in pretrained_model.bert.encoder.layer[:6]
        ])
        self.pooler = SumBertPooler()
        self.classifier = torch.nn.Sequential(
            nn.Linear(768, 1)
        )

    def forward(self, input):
        out = self.bert_encoder(input)
        out = self.pooler(out.last_hidden_state)
        out = self.classifier(out)
        return torch.sigmoid(out).view(-1)

def eval_auc(model):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in test_loader:
            encodings = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt").to(device)  # device move
            input_ids = encodings['input_ids']
            token_type_ids = encodings['token_type_ids']
            embeded = embedding_model(input_ids=input_ids, token_type_ids=token_type_ids)
            embeded = embeded.last_hidden_state
            label = batch[1].float().to(device)  # device move

            outputs = model(embeded)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(label.cpu().numpy())

    auc = roc_auc_score(actuals, predictions)
    print("AUC:", auc)
    return auc

def get_model_info_dict(model, epoch, auc_score):
    current_device = next(model.parameters()).device
    model.to('cpu')
    model_info = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'auc_score': auc_score,
    }
    model.to(current_device)
    return model_info

def preparation_embedding(texts):
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)  # device move
    input_ids = encodings['input_ids']
    token_type_ids = encodings['token_type_ids']
    embeded = embedding_model(input_ids=input_ids, token_type_ids=token_type_ids)
    return embeded.last_hidden_state

def GAN_step(optimizerG, optimizerD, netG, netD, real_data, label, epoch, i):
    netD.zero_grad()
    batch_size = real_data.size(0)

    # Discriminator on real data (label=1)
    output = netD(real_data)
    label_real = torch.ones(batch_size, device=real_data.device, dtype=torch.float)
    errD_real = criterion(output, label_real)
    errD_real.backward()
    D_x = output.mean().item()

    print(f"[DEBUG] D(real) output min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {D_x:.4f}")

    # Generate fake data
    noise = torch.randn(batch_size, nz, device=real_data.device)
    fake_data = netG(noise)

    # Discriminator on fake data (label=0)
    output = netD(fake_data.detach())
    label_fake = torch.zeros(batch_size, device=real_data.device, dtype=torch.float)
    errD_fake = criterion(output, label_fake)
    errD_fake.backward()
    D_G_z1 = output.mean().item()

    print(f"[DEBUG] D(fake) output min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {D_G_z1:.4f}")

    errD = errD_real + errD_fake
    optimizerD.step()

    # Generator tries to fool discriminator (label=1)
    netG.zero_grad()
    output = netD(fake_data)
    errG = criterion(output, label_real)  # label_real = 1
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()

    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
          % (epoch, num_epochs, i, len(train_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    return optimizerG, optimizerD, netG, netD


netG = Generator(nz).to(device)  # device move
netD = Discriminator().to(device)  # device move

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

model_infos = []
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        with torch.no_grad():
            embeded = preparation_embedding(data[0])

        optimizerG, optimizerD, netG, netD = GAN_step(
            optimizerG=optimizerG,
            optimizerD=optimizerD,
            netG=netG,
            netD=netD,
            real_data=embeded.to(device),  # device move redundant but safe
            label=data[1].float().to(device),  # device move
            epoch=epoch, i=i)

    auc_score = eval_auc(netD)
    model_infos.append(get_model_info_dict(netD, epoch, auc_score))

print('Train complete！')

# Inference

max_auc_model_info = max(model_infos, key=lambda x: x['auc_score'])
model = Discriminator()
model.load_state_dict(max_auc_model_info['model_state_dict'])
model.to(device)  # device move
model.eval()

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, idx):
        return self.texts[idx]

    def __len__(self):
        return len(self.texts)

sub_dataset = InferenceDataset(src_sub["text"].tolist())
inference_loader = DataLoader(sub_dataset, batch_size=test_batch_size, shuffle=False)

sub_predictions = []
with torch.no_grad():
    for batch in inference_loader:
        encodings = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)  # device move
        input_ids = encodings['input_ids']
        token_type_ids = encodings['token_type_ids']
        embeded = embedding_model(input_ids=input_ids, token_type_ids=token_type_ids)
        embeded = embeded.last_hidden_state.to(device)  # device move

        outputs = model(embeded)
        sub_predictions.extend(outputs.cpu().numpy())

sub_ans_df = pd.DataFrame({"id": src_sub["id"], "generated": sub_predictions})
print(sub_ans_df)
