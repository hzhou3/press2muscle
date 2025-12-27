import os
import random
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import torch.multiprocessing as mp
import time
import datetime

from utils.parser_util import get_args
from utils.loss_util import * 

from dataloader.dataloader import get_dataloader, load_data, TrainDataset
from models.full_model import TransformerAutoEncoder

from full_test import test_process




def loss_function(args, y_pred, y_gt):
    loss_func = torch.nn.MSELoss(reduction='none')
    recons_y = (loss_func(y_gt, y_pred) ).mean()
    g_loss = smooth_loss(y_pred, y_gt) # this is smooth loss
    
    loss_w = args.LOSS
    loss_all = (
                loss_w.recons_y * recons_y + 
                loss_w.smooth_loss * g_loss 
                )
    loss = {
        "loss": loss_all,
        "recons_y": recons_y,
        'g_loss': g_loss,
    }
    return loss


def save_checkpoint(states, output_dir):
    if True:
        torch.save(
            states['state_dict'],
            os.path.join(output_dir, 'best.pth.tar')
        )


def do_train(args, model, train_dataloader, log_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    begin_epoch = 0
    output_dir = args.SAVE_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, 'checkpoint.pth.tar')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.LR, betas=(0.9, 0.99), weight_decay=args.WEIGHT_DECAY)
    if os.path.exists(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.MILESTONES, gamma=1 / 4, last_epoch=begin_epoch if begin_epoch else -1)

    model.train()

    for epoch in range(begin_epoch, args.EPOCH):
        tqdm.write(f"Starting epoch {epoch}")
        tqdm.write(f"current lr:{scheduler.get_last_lr()}")
        train_dataloader = tqdm(train_dataloader, dynamic_ncols=True)

        all_loss = 0

        for x, y, uid, act_gt in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            uid = uid.to(device)
            act_gt = act_gt.to(device)
            
            if args.MASK:
                B, T, C = x.shape  # Extract dimensions
                random_mask = torch.rand((B, T, C))  # Shape (B, T, C)
                mask = random_mask < args.MASK_RATIO  # Boolean mask, True where random_mask < mask_ratio
                x[mask] = 0.01

            y_pred, _ = model(x=x, uid=uid)
            loss = loss_function(args, y_pred, y)

            optimizer.zero_grad()
            loss["loss"].backward()
            all_loss += loss["loss"].item()
            optimizer.step()
            train_dataloader.set_description(f"e:{epoch},rc_y:{loss['recons_y']:.2e},g_loss:{loss['g_loss']:.2e}")

            # scheduler.step()

        
        with open(log_path, 'a') as f:
            f.write(f"\n==> epoch:{epoch}, loss:{all_loss/len(train_dataloader):.4f}\n")

        scheduler.step()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, output_dir)

        test_process(args, log_path, split='val', epoch=epoch)
        train_dataloader.close()


def main():
    args = get_args()
    torch.backends.cudnn.benchmark = False
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    if args.SAVE_DIR is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)

    output_dir = args.SAVE_DIR
    
    timestamp = time.time()
    dt = datetime.datetime.fromtimestamp(timestamp)
    formatted_dt = dt.strftime("%Y%m%d_%H%M")
    log_path = os.path.join(output_dir, f"{args.NAME}_{formatted_dt}.log")

    with open(log_path, 'w') as f:
        f.write(str(args) + '\n')
        print(f"Args saving to {log_path}")

    inputs, outputs, files = load_data(
        args.DATASET_PATH,
        "train",
        protocol=args.PROTOCOL,
        input_motion_length=args.INPUT_MOTION_LENGTH,
        input_dim=args.INPUT_DIM,
        output_dim=args.OUTPUT_DIM,
    )
    train_dataset = TrainDataset(
        inputs,
        outputs,
        files,
        args.INPUT_MOTION_LENGTH,
        args.TRAIN_DATASET_REPEAT_TIMES,
        with_context=True, # bio vector
    )
    train_dataloader = get_dataloader(
        train_dataset, "train", batch_size=args.BATCH_SIZE, num_workers=args.NUM_WORKERS
    )

    print("creating model...")
    print(f"{args.SAVE_DIR}")

    in_dim = args.INPUT_DIM
    out_dim = args.OUTPUT_DIM
    model_cfg = args.MODEL_CFG
    block_size = args.INPUT_MOTION_LENGTH + 1
    model = TransformerAutoEncoder(in_dim=in_dim, out_dim=out_dim, n_layers=model_cfg.n_layers, hid_dim=model_cfg.hid_dim, heads=model_cfg.heads,
                             dropout=model_cfg.dropout, embed_type=model_cfg.embed_type, block_size=block_size)

    print("Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training...")
    do_train(args, model, train_dataloader, log_path)
    print("Done.")


if __name__ == '__main__':
    mp.set_sharing_strategy('file_system')
    main()
