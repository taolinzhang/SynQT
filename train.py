import torch
from torch.optim import AdamW
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import *
from synqt import set_synqt
from IPython import embed
import wandb
import torch.nn as nn
from loguru import logger
import sys

def train(config, model, dl, opt, scheduler, epoch):
    model.train()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    for ep in range(epoch):
        pbar = tqdm(dynamic_ncols=True, leave=True, desc=False)
        model.train()
        model = model.cuda()
        total_loss = []
        total_size = len(dl)
        for i, batch in enumerate(dl):
            adjust_learning_rate(opt, i / len(dl) + ep, epoch, args)
            opt.zero_grad()
            pbar.set_description(
                f"Epoch {ep}/{epoch} | Training | Total Iter {total_size}")
            pbar.update(1)
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = criterion(out,y)
            total_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        logger.info(f"\nEpoch {ep} Loss: {sum(total_loss)/len(total_loss)}")
        log_dict = {"Train Loss": sum(total_loss)/len(total_loss)}
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(model, test_dl)
            torch.cuda.empty_cache()
            logger.info(f"\nEpoch {ep} Test Acc: {acc}")
            log_dict.update({"Test Acc": acc})
            if acc > config['best_acc']:
                config['best_acc'] = acc
    model = model.cpu()
    return model

@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    total_size = len(dl)
    pbar = tqdm(dynamic_ncols=True, leave=True, desc=False)
    model = model.cuda()
    for batch in dl:  # pbar:
        pbar.set_description(f"Testing | Total Iter {total_size}")
        pbar.update(1)
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y)

    return acc.result()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--model', type=str,
                        default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument("--method", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--feat_scale", type=float, default=1)
    parser.add_argument("--attn_scale", type=float, default=1)
    parser.add_argument("--ffn_scale", type=float, default=1)
    parser.add_argument("--token_nums", type=int, default=1)
    parser.add_argument("--search", type=int)
    args = parser.parse_args()
    fmt = "[{time: MM-DD hh:mm:ss}] {message}"
    logger_config = {
        "handlers": [
            {"sink": sys.stderr, "format": fmt},
        ],
    }
    logger.configure(**logger_config)
    config = get_config(args.ckpt, args.dataset)
    # load attn_scale and token_nums
    args.feat_scale = config["feat_scale"]
    args.attn_scale = config["attn_scale"]
    args.ffn_scale = config["ffn_scale"]
    args.token_nums = config["token_nums"]
    file_root = f"log/{args.dataset}/{args.exp_name}/feat_{args.feat_scale}_attn_{args.attn_scale}_ffn_{args.ffn_scale}_token_{args.token_nums}_lr_{args.lr}_wd_{args.wd}_{args.dataset}"
    file_path = f"{file_root}/{args.exp_name}.txt"
    # if os.path.exists(file_path):
    #     print("file already exists")
    #     exit(0)
    if not os.path.exists(file_root):
        os.makedirs(file_root)
        open(file_path,'w').close()
    logger.add(file_path,format=fmt)    
    logger.info(args)
    set_seed(args.seed)
    if args.ckpt == 'vit_1k':
        checkpoint_path = "ckpt/1k.npz"
    else:
        checkpoint_path = "ckpt/21k.npz"
    model = create_model(
        args.model, checkpoint_path=checkpoint_path, drop_path_rate=0.1)
    ref_model = create_model(
        args.model, checkpoint_path=checkpoint_path, drop_path_rate=0.1)
    train_dl, test_dl = get_data(args.dataset)

    if args.method == 'synqt':
        set_synqt(args, model)
    else:
        exit()

    trainable = []
    model.reset_classifier(config['class_num'])

    config['best_acc'] = 0
    config['method'] = args.method

    if args.method == "synqt":
        for n, p in model.named_parameters():
            if 'head' in n or 'synqt' in n:
                trainable.append(p)
            else:
                p.requires_grad = False
    else:
        exit()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable param: {n_parameters/1e6} M")
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=50,
                                  warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6)
    model = train(config, model, train_dl,
                  opt, scheduler, epoch=50)
    logger.info(f"Best Acc: {config['best_acc']}")