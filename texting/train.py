import os
import argparse
import time

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

def save_model(model, epoch, train_loss, val_loss, 
                train_acc, val_acc, save_path):
    print(f'Saving model at epoch: {epoch} to {save_path}')
    torch.save({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'state_dict': model.state_dict()
    }, save_path)

def train_one_epoch(model, train_loader, args, criterion, optimizer):
    epoch_loss = 0
    total_samples = 0
    total_correct = 0
    model.train()
    for feat, adj, mask, y in tqdm(train_loader):
        inputs = to_tensor_device(feat, mask, adj, device=args.device)
        y = y.to(args.device)
        output = model(*inputs)
        weights_loss = model.l2_loss()
        classification_loss = criterion(output, y)
        loss = weights_loss + classification_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * feat.shape[0]
        total_samples += feat.shape[0]
        total_correct += (output.argmax(dim=1) == y).sum().item()

    epoch_loss /= total_samples
    accuracy = round(total_correct/total_samples, 4)
    print(f'Training:\nloss: {epoch_loss}, accuracy: {accuracy}')
    return epoch_loss, accuracy

def evaluate(model, val_loader, criterion, args):
    model.eval()
    val_loss = 0
    total_samples = 0
    total_correct = 0
    with torch.no_grad():
        for feat, adj, mask, y in tqdm(val_loader):
            inputs = to_tensor_device(feat, mask, adj, device=args.device)
            y = y.to(args.device)
            output = model(*inputs)
            weights_loss = model.l2_loss()
            classification_loss = criterion(output, y)
            loss = weights_loss + classification_loss
            val_loss += loss.item() * feat.shape[0]
            total_samples += feat.shape[0]
            total_correct += (output.argmax(dim=1) == y).sum().item()

    val_loss /= total_samples
    val_accuracy = round(total_correct/total_samples, 4)
    print(f'Validation:\nloss: {val_loss}, accuracy: {val_accuracy}')
    return val_loss, val_accuracy

def train(model, train_loader, val_loader, epochs, args):
    min_val_loss = float('inf')
    model = model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print('Starting training')
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        train_loss, train_acc = train_one_epoch(model, train_loader, args,
                                                criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model(model, epoch, train_loss, val_loss, 
                        train_acc, val_acc, os.path.join(args.save_dir, 'model.pt'))


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
    parser.add_argument('--save_dir', default='./saved_models/', type=str)
    parser.add_argument('--data_dir', default='/common/home/as3503/as3503/courses/cs533/TextING-PyTorch/texting/data', type=str)

    args = parser.parse_args()

    print('Loading dataset')
    train_dataset = GraphDataset(dataset_str=args.dataset, data_dir=args.data_dir, part='train')
    val_dataset = GraphDataset(dataset_str=args.dataset, data_dir=args.data_dir, part='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    # test_dataset = GraphDataset(dataset_str=args.dataset, part='test')

    os.makedirs(args.save_dir, exist_ok=True)

    model = GatedGNN(args, output_dim=train_dataset.output_dim)

    train(model, train_loader, val_loader, args.epochs, args)
