import torch
import os
import argparse

from datasets import GraphDataset
from sklearn import metrics
from models import MLP, GatedGNN
from tqdm import tqdm
from torch.utils.data import DataLoader


def to_tensor_device(*args, device):
    output = []
    for arg in args:
        output.append(torch.tensor(arg).to(device).float())
    return output


def evaluate(features, support, mask, labels, placeholders):
    ...


def train(model, train_loader, val_loader, epochs, args):
    min_val_loss = float('inf')
    model = model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print('Starting training')
    for epoch in range(epochs):
        epoch_loss = 0
        total_samples = 0
        model.train()
        for feat, adj, mask, y in tqdm(train_loader):
            inputs = to_tensor_device(feat, mask, adj, device=args.device)
            output = model(*inputs)
            weights_loss = model.l2_loss()
            classification_loss = criterion(output, y.to(args.device))
            loss = weights_loss + classification_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * feat.shape[0]
            total_samples += feat.shape[0]

        epoch_loss /= total_samples
        print(f'Epoch: {epoch}, loss: {epoch_loss}')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieval model parameters')
    parser.add_argument('--dataset', default='mr', type=str)
    parser.add_argument('--learning_rate', default=0.005, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--input_dim', default=300, type=int)
    parser.add_argument('--hidden', default=96, type=int)
    parser.add_argument('--gnn_steps', default=2, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=0, type=int)
    parser.add_argument('--early_stopping', default=-1, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()

    print('Loading dataset')
    train_dataset = GraphDataset(dataset_str=args.dataset, part='train')
    val_dataset = GraphDataset(dataset_str=args.dataset, part='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    # test_dataset = GraphDataset(dataset_str=args.dataset, part='test')

    model = GatedGNN(args, output_dim=train_dataset.output_dim)

    train(model, train_loader, val_loader, args.epochs, args)
