import torch
import torch.nn as nn
import torch.nn.functional as F
from collator import collator
import random
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from model import GT
from lr import PolynomialDecayLR
import argparse
import math
from tqdm import tqdm
from preprocess_data import node_sampling, process_data
from torch.nn.functional import normalize
import scipy.sparse as sp
from numpy.linalg import inv


def train(args, model, device, loader, optimizer, lr_scheduler):
    model.train()

    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        pred = model(batch)
        y_true = batch.y.view(-1)
        loss = F.nll_loss(pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_train(args, model, device, loader):
    y_true = []
    y_pred = []
    loss_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Iteration"):
            batch = batch.to(device)
            pred = model(batch)
            loss_list.append(F.nll_loss(pred, batch.y.view(-1)).item())
            y_true.append(batch.y)
            y_pred.append(pred.argmax(1))

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    correct = (y_pred == y_true).sum()
    acc = correct.item() / len(y_true)

    return acc, np.mean(loss_list)


def eval(args, model, device, loader):
    y_true = []
    y_pred = []
    loss_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Iteration"):
            batch = batch.to(device)
            pred = model(batch)
            loss_list.append(F.nll_loss(pred, batch.y.view(-1)).item())
            y_true.append(batch.y)
            y_pred.append(pred.argmax(1))

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    pred_list = []
    for i in torch.split(y_pred, args.num_data_augment, dim=0):
        pred_list.append(i.bincount().argmax().unsqueeze(0))
    y_pred = torch.cat(pred_list)
    y_true = y_true.view(-1, args.num_data_augment)[:, 0]
    correct = (y_pred == y_true).sum()
    acc = correct.item() / len(y_true)

    return acc, np.mean(loss_list)


def get_reward(args, model, device, loader, p):
    column_normalized_adj = sp.load_npz('./dataset/' + args.dataset_name + '/column_normalized_adj.npz')
    normalized_adj = sp.load_npz('./dataset/'+args.dataset_name+'/normalized_adj.npz')
    data_x = torch.load('./dataset/' + args.dataset_name + '/x.pt')
    normalized_adj1 = normalized_adj*normalized_adj
    eigen_adj = 0.15 * inv((sp.eye(normalized_adj.shape[0]) - (1 - 0.15) * column_normalized_adj).toarray())
    eigen_adj1 = normalized_adj.toarray()
    eigen_adj2 = normalized_adj1.toarray()
    x = normalize(data_x, dim=1)
    eigen_adj3 = np.array(torch.matmul(x, x.transpose(1, 0)))
    r = [[], [], [], []]
    reward = np.zeros(4)
    model.eval()
    n_node = 10
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            scores = model(batch, get_score=True)
        scores = scores[:, 1:n_node]
        for i, score in enumerate(torch.split(scores, args.num_data_augment, dim=0)):
            ids = batch.ids[i*args.num_data_augment:(i+1)*args.num_data_augment]
            id = ids[0, 0].cpu().item()
            ids = ids[:, 1:n_node]
            s = eigen_adj[id]
            s1 = eigen_adj1[id]
            s2 = eigen_adj2[id]
            s3 = eigen_adj3[id]
            s[id], s1[id], s2[id], s3[id] = 0, 0, 0, 0
            s = torch.tensor(np.maximum(s, 0)).to(device)
            s = s/(s.sum()+1e-5)
            s1 = torch.tensor(np.maximum(s1, 0)).to(device)
            s1 = s1/(s1.sum()+1e-5)
            s2 = torch.tensor(np.maximum(s2, 0)).to(device)
            s2 = s2/(s2.sum()+1e-5)
            s3 = torch.tensor(np.maximum(s3, 0)).to(device)
            s3 = s3 / (s3.sum() + 1e-5)
            phi = p[0]*s + p[1]*s1 + p[2]*s2 + p[3]*s3 + 1e-5
            r[0].append(torch.sum(score * s[ids] / phi[ids]) / args.num_data_augment)
            r[1].append(torch.sum(score * s1[ids] / phi[ids]) / args.num_data_augment)
            r[2].append(torch.sum(score * s2[ids] / phi[ids]) / args.num_data_augment)
            r[3].append(torch.sum(score * s3[ids] / phi[ids]) / args.num_data_augment)
    reward[0] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[0]])).cpu().numpy()
    reward[1] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[1]])).cpu().numpy()
    reward[2] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[2]])).cpu().numpy()
    reward[3] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[3]])).cpu().numpy()
    return reward


def random_split(data_list, frac_train, frac_valid, frac_test, seed):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    random.seed(seed)
    all_idx = np.arange(len(data_list))
    random.shuffle(all_idx)
    train_idx = all_idx[:int(frac_train * len(data_list))]
    val_idx = all_idx[int(frac_train * len(data_list)):int((frac_train+frac_valid) * len(data_list))]
    test_idx = all_idx[int((frac_train+frac_valid) * len(data_list)):]
    train_list = []
    test_list = []
    val_list = []
    for i in train_idx:
        train_list.append(data_list[i])
    for i in val_idx:
        val_list.append(data_list[i])
    for i in test_idx:
        test_list.append(data_list[i])
    return train_list, val_list, test_list


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of graph transformer')
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--attn_bias_dim', type=int, default=6)
    parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--peak_lr', type=float, default=2e-4)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--num_data_augment', type=int, default=8)
    parser.add_argument('--num_global_node', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--perturb_feature', type=bool, default=False)
    parser.add_argument('--weight_update_period', type=int, default=10000, help='epochs to update the sampling weight')
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    data_list = torch.load('./dataset/'+args.dataset_name+'/data.pt')
    feature = torch.load('./dataset/'+args.dataset_name+'/feature.pt')
    y = torch.load('./dataset/'+args.dataset_name+'/y.pt')
    train_dataset, test_dataset, valid_dataset = random_split(data_list, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)
    print('dataset load successfully')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn=partial(collator, feature=feature, shuffle=True, perturb=args.perturb_feature))
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=partial(collator, feature=feature, shuffle=False))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=partial(collator, feature=feature, shuffle=False))
    print(args)

    model = GT(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        input_dim=feature.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=y.max().item()+1,
        attn_bias_dim=args.attn_bias_dim,
        attention_dropout_rate=args.attention_dropout_rate,
        dropout_rate=args.dropout_rate,
        intput_dropout_rate=args.intput_dropout_rate,
        ffn_dim=args.ffn_dim,
        num_global_node=args.num_global_node
    )
    if not args.test and not args.validate:
        print(model)
    print('Total params:', sum(p.numel() for p in model.parameters()))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_epochs,
            tot=args.epochs,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)

    val_acc_list, test_acc_list = [], []
    sampling_weight = np.ones(4)
    weight_history = []
    p_min = 0.05
    p = (1 - 4 * p_min) * sampling_weight / sum(sampling_weight) + p_min

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train(args, model, device, train_loader, optimizer, lr_scheduler)
        lr_scheduler.step()

        print("====Evaluation")
        train_acc, train_loss = eval_train(args, model, device, train_loader)

        val_acc, val_loss = eval(args, model, device, val_loader)
        test_acc, test_loss = eval(args, model, device, test_loader)

        print("train_acc: %f val_acc: %f test_acc: %f" % (train_acc, val_acc, test_acc))
        print("train_loss: %f val_loss: %f test_loss: %f" % (train_loss, val_loss, test_loss))
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

        if epoch % args.weight_update_period == 0:
            r = get_reward(args, model, device, val_loader, p)
            print('reward:', r)
            sampling_weight = sampling_weight*np.exp(2.0*(r+0.01/p))
            p = (1 - 4 * p_min) * sampling_weight / sum(sampling_weight) + p_min
            print('p:', p)
            weight_history.append(p)
            data_list, feature = node_sampling(p)
            train_dataset, valid_dataset, test_dataset = random_split(data_list, frac_train=0.6, frac_valid=0.2,
                                                                      frac_test=0.2, seed=args.seed)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers,
                                      collate_fn=partial(collator, feature=feature, shuffle=True,
                                                         perturb=args.perturb_feature))
            val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers,
                                    collate_fn=partial(collator, feature=feature, shuffle=False))
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     collate_fn=partial(collator, feature=feature, shuffle=False))

    print('best validation acc: ', max(val_acc_list))
    print('best test acc: ', max(test_acc_list))
    print('best acc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])
    np.save('./exps/'+args.dataset_name+'/weight_history', np.array(weight_history))
    np.save('./exps/' + args.dataset_name + '/test_acc_list', np.array(test_acc_list))
    np.save('./exps/' + args.dataset_name + '/val_acc_list', np.array(val_acc_list))


if __name__ == "__main__":
    main()
