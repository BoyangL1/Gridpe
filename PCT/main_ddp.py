from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import ModelNet40
from model_pct_xyz import PctXYZ
from model_pct import Pct
from model_pct_grid import PctGrid
from model_pct_rope import PctRope
import numpy as np
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import time

def _init_(args):
    if dist.get_rank() != 0:
        return

    model_file_map = {
        'pct': 'model_pct.py',
        'pctxyz': 'model_pct_xyz.py',
        'pctgrid': 'model_pct_grid.py',
        'pctrope': 'model_pct_rope.py',
    }
    selected_model_file = model_file_map.get(args.model.lower(), 'unknown_model.py')

    os.makedirs(f'checkpoints/{args.exp_name}/models', exist_ok=True)
    os.system(f'cp main_ddp.py checkpoints/{args.exp_name}/main.py.backup')
    os.system(f'cp {selected_model_file} checkpoints/{args.exp_name}/model.py.backup')
    os.system(f'cp util.py checkpoints/{args.exp_name}/util.py.backup')
    os.system(f'cp data.py checkpoints/{args.exp_name}/data.py.backup')

def is_main_process():
    return dist.get_rank() == 0

def train(args, io):
    train_dataset = ModelNet40(partition='train', num_points=args.num_points)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)

    test_dataset = ModelNet40(partition='test', num_points=args.num_points)
    test_sampler = DistributedSampler(test_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=8, batch_size=args.test_batch_size, sampler=test_sampler, drop_last=False)

    device = torch.device("cuda", args.local_rank)

    # model choice
    model_dict = {
        'pct': Pct,
        'pctxyz': PctXYZ,
        'pctgrid': PctGrid,
        'pctrope': PctRope,
    }
    if args.model.lower() not in model_dict:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(model_dict.keys())}")
    model = model_dict[args.model.lower()](args).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.use_sgd:
        if is_main_process(): print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=5e-4)
    else:
        if is_main_process(): print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss
    best_test_acc = 0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        scheduler.step()

        model.train()
        train_loss = 0.0
        count = 0.0
        train_pred = []
        train_true = []
        total_time = 0.0

        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            opt.zero_grad()
            start_time = time.time()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        if is_main_process():
            print('train total time is', total_time)
            outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (
                epoch,
                train_loss * 1.0 / count,
                metrics.accuracy_score(train_true, train_pred),
                metrics.balanced_accuracy_score(train_true, train_pred))
            io.cprint(outstr)

        # ------------------ Test ------------------
        model.eval()
        test_loss = 0.0
        count = 0.0
        test_pred = []
        test_true = []
        total_time = 0.0

        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            with torch.no_grad():
                start_time = time.time()
                logits = model(data)
                end_time = time.time()
                total_time += (end_time - start_time)

                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        if is_main_process():
            print('test total time is', total_time)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (
                epoch, test_loss * 1.0 / count, test_acc, avg_per_class_acc)
            io.cprint(outstr)

            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), f'checkpoints/{args.exp_name}/models/model.t7')

def test(args, io):
    # output_dir = "./attn_output/"
    # if not dist.is_initialized() or dist.get_rank() == 0:
    #     os.makedirs(output_dir, exist_ok=True)

    #     model_name = args.model.lower()
    #     dist_path = os.path.join(output_dir, f"{model_name}_attn_distance.txt")
    #     ent_path  = os.path.join(output_dir, f"{model_name}_attn_entropy.txt")

    #     with open(dist_path, "a") as f_dist, open(ent_path, "a") as f_ent:
    #         f_dist.write(f"Input points: {args.num_points}\n")
    #         f_ent.write (f"Input points: {args.num_points}\n")
                    
    device = torch.device("cuda", args.local_rank)
    test_dataset = ModelNet40(partition='test', num_points=args.num_points)
    test_sampler = DistributedSampler(test_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, drop_last=False)

    # model choice
    model_dict = {
        'pct': Pct,
        'pctxyz': PctXYZ,
        'pctgrid': PctGrid,
        'pctrope': PctRope,
    }
    if args.model.lower() not in model_dict:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(model_dict.keys())}")
    model = model_dict[args.model.lower()](args).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        with torch.no_grad():
            logits = model(data)
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    if is_main_process():
        outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
        io.cprint(outstr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pct', choices=['pct', 'pctxyz', 'pctgrid', 'pctrope'], help='Model type choices')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--use_sgd', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')

    _init_(args)

    if is_main_process():
        io = IOStream(f'checkpoints/{args.exp_name}/run.log')
        io.cprint(str(args))
    else:
        io = None

    if not args.eval:
        train(args, io)
    else:
        test(args, io)