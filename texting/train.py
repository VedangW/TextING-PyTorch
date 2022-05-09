import os
import argparse
import time
import wandb
import torch


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
        # print(f'{output=}, {weights_loss=}, {classification_loss=}')
        # time.sleep(1)
        loss = weights_loss + classification_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * feat.shape[0]
        total_samples += feat.shape[0]
        total_correct += (output.argmax(dim=1) == y).sum().item()

    epoch_loss /= total_samples
    accuracy = round(total_correct/total_samples, 4)
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
    return val_loss, val_accuracy

def train(model, train_loader, val_loader, epochs, args):
    if args.wandb:
        wandb.init(project='TextING-cs533', config=args)
        wandb.config.update(args)
        os.makedirs(os.path.join(args.save_dir, wandb.run.id), exist_ok=True)
    max_val_acc = 0
    model = model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print('Starting training')
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, args,
                                                criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args)
        print(f'{epoch=}, {train_loss=}, {train_acc=}, {val_loss=}, {val_acc=}')
        if args.wandb:
            log_dict = dict(
                epoch=epoch, 
                train_loss=train_loss, 
                train_acc=train_acc, 
                val_loss=val_loss, 
                val_acc=val_acc
            )
            wandb.log(log_dict)

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            save_model(model, epoch, train_loss, val_loss, 
                        train_acc, val_acc, os.path.join(args.save_dir, wandb.run.id, f'model_{args.dataset}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TextING training options')
    parser.add_argument('--dataset', default='mr', type=str)
    parser.add_argument('--wandb', default=True, type=bool)
    parser.add_argument('--learning_rate', default=0.005, type=float)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--batch_size', default=4096, type=int)
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
