import csv
import math
import multiprocessing
import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms

params = {
    'batch_size': 16,
    'weight_decay':0.0001,
    'modelname':'V_2D_nabla_params',
    'epochs':50,
    'save_every':10,
}

class Regression(nn.Module):
    def __init__(self, output_dim1, output_dim2, output_dim3):
        super().__init__()
        self.model_ft = models.resnet18(pretrained=False)
        num_ftrs = 1000
        self.out1 = nn.Linear(num_ftrs, output_dim1)
        nn.init.xavier_normal_(self.out1.weight)
        self.out2 = nn.Linear(num_ftrs, output_dim2)
        nn.init.xavier_normal_(self.out2.weight)
        self.out3 = nn.Linear(num_ftrs, output_dim3)
        nn.init.xavier_normal_(self.out3.weight)
    def forward(self, x):
        h = self.model_ft(x)
        y1 = self.out1(h)
        y2 = self.out2(h)
        y3 = self.out3(h)
        return y1, y2, y3

class ImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        ext = 'jpg'
        folder = '../data/shape'
        image_size = 256 if torch.cuda.is_available() else 64
        fnames = ['2D', 'V', 'nabla']
        self.paths = [p for fname in fnames for p in Path(f'{folder}').glob(f'{fname}/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        folder_label = Path('../data/coefficient')
        fnames_label = ['2D', 'V', 'nabla']
        cnames_label = ['label_Psi_a_coef', 'label_Ld_coef', 'label_Lq_coef']
        self.labels = []
        for cname in cnames_label:
            df = pd.DataFrame()
            for fname in fnames_label:
                df = pd.concat([df,
                                pd.read_csv(folder_label / fname / f'{cname}_{fname}_scaled.csv',
                                            index_col=0)])
            df.index = range(df.shape[0])
            self.labels.append(df)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        Psi_a = self.labels[0].iloc[index].values
        Ld = self.labels[1].iloc[index].values
        Lq = self.labels[2].iloc[index].values
        return self.transform(img), Psi_a, Ld, Lq

def set_data_src(NUM_CORES, batch_size, world_size, rank):
    dataset = ImageDataset()
    n_samples = len(dataset)
    train_size = int(n_samples*0.8)
    test_size = n_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_sampler = DistributedSampler(
        train_dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True
        ) if is_ddp else None
    train_dataloader = DataLoader(
        train_dataset,
        num_workers = math.ceil(NUM_CORES / world_size),
        batch_size = math.ceil(batch_size / world_size),
        sampler = train_sampler,
        shuffle = not is_ddp,
        drop_last = True,
        pin_memory = True
        )
    test_sampler = DistributedSampler(
        test_dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True
        ) if is_ddp else None
    test_dataloader = DataLoader(
        test_dataset,
        num_workers = math.ceil(NUM_CORES / world_size),
        batch_size = math.ceil(batch_size / world_size),
        sampler = test_sampler,
        shuffle = not is_ddp,
        drop_last = True,
        pin_memory = True
        )
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CORES = multiprocessing.cpu_count()
    world_size = torch.cuda.device_count()
    is_ddp = world_size > 1
    rank = 0
    batch_size = params['batch_size']
    base_dir = './'
    results_dir = 'results'
    name = params['modelname']
    dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M')[2:]
    name = dt_now + '_' + name
    base_dir = Path(base_dir)
    (base_dir / results_dir / name).mkdir(parents=True, exist_ok=True)

    def model_name(num):
        return str(base_dir / results_dir / name / f'model_{num}.pt')
    def save_model(model, num):
        torch.save(model, model_name(num))
    def save_result(result, num):
        with open(str(base_dir / results_dir / name / f'result_{num}.csv'), 'w', encoding='Shift_jis') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(result)

    train_loader, test_loader = set_data_src(NUM_CORES, batch_size, world_size, rank)
    model = Regression(4,10,15).to(device)

    def compute_loss(label, pred):
        return criterion(pred.float(), label.float())
    def train_step(x, t1, t2, t3):
        model.train()
        preds = model(x)
        loss1 = compute_loss(t1, preds[0])
        loss2 = compute_loss(t2, preds[1])
        loss3 = compute_loss(t3, preds[2])
        optimizer.zero_grad()
        loss = 3*loss1 + 2*loss2 + loss3
        loss.backward()
        optimizer.step()
        return (loss1, loss2, loss3), preds
    def test_step(x, t1, t2, t3):
        model.eval()
        preds = model(x)
        loss1 = compute_loss(t1, preds[0])
        loss2 = compute_loss(t2, preds[1])
        loss3 = compute_loss(t3, preds[2])
        return (loss1, loss2, loss3), preds
    criterion = nn.MSELoss()
    optimizer = optimizers.Adam(model.parameters(), weight_decay=params['weight_decay'])

    epochs = params['epochs']
    save_every = params['save_every']
    results = []
    time_start = time.time()
    for epoch in range(epochs):
        train_loss1 = 0.
        train_loss2 = 0.
        train_loss3 = 0.
        test_loss1 = 0.
        test_loss2 = 0.
        test_loss3 = 0.
        for (x, t1, t2, t3) in train_loader:
            x, t1, t2, t3 = x.to(device), t1.to(device), t2.to(device), t3.to(device)
            loss, _ = train_step(x, t1, t2, t3)
            train_loss1 += loss[0].item()
            train_loss2 += loss[1].item()
            train_loss3 += loss[2].item()
        train_loss1 /= len(train_loader)
        train_loss2 /= len(train_loader)
        train_loss3 /= len(train_loader)
        for (x, t1, t2, t3) in test_loader:
            x, t1, t2, t3 = x.to(device), t1.to(device), t2.to(device), t3.to(device)
            loss, _ = test_step(x, t1, t2, t3)
            test_loss1 += loss[0].item()
            test_loss2 += loss[1].item()
            test_loss3 += loss[2].item()
        test_loss1 /= len(test_loader)
        test_loss2 /= len(test_loader)
        test_loss3 /= len(test_loader)
        elapsed_time = time.time()-time_start
        print('Epoch: {}, Train rmse: {}, Test rmse: {}, Elapsed time: {:.1f}sec'.format(
            epoch+1,
            (train_loss1, train_loss2, train_loss3),
            (test_loss1, test_loss2, test_loss3),
            elapsed_time
        ))
        results.append([
            epoch+1,
            train_loss1,
            train_loss2,
            train_loss3,
            test_loss1,
            test_loss2,
            test_loss3,
            elapsed_time
        ])
        if (epoch+1) % save_every == 0: save_model(model.state_dict(), epoch+1)
        save_result(results, epoch+1)