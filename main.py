import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from engine import train_one_epoch, evaluate
from dataset.image_dataset import get_dataset
from my_models.unet import UNet
from tools.utils import setup_seed, save_checkpoint
from tools.losses import get_loss
from tools.samplers import DistributedSampler

def parse_args():
    parser = argparse.ArgumentParser(description='U-Net Image Segmentation')
    parser.add_argument('--data_dir', help='dataset directory')
    parser.add_argument('--dataset', default='cityscapes', help='dataset name: cityscapes, voc')
    parser.add_argument('--model', default='unet', help='model name')
    parser.add_argument('--depth', type=int, default=5, help='U-Net depth')
    parser.add_argument('--num_channels', type=int, default=64, help='U-Net base channels')
    parser.add_argument('--img_size', type=int, default=256, help='input image size')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--opt', default='adamw', help='optimizer')
    parser.add_argument('--sched', default='cosine', help='scheduler')
    parser.add_argument('--output_dir', default='./outputs', help='output directory')
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('--initial_checkpoint', default='', help='path to checkpoint')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()

def main_worker(rank, world_size, args):
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    setup_seed(42 + rank)

    # Dataset
    train_dataset, val_dataset, num_classes = get_dataset(args.dataset, args.data_dir, args.img_size)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    # Model
    model = UNet(num_classes=num_classes, depth=args.depth, num_channels=args.num_channels).cuda(rank)
    model = DDP(model, device_ids=[rank])

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    criterion = get_loss('cross_entropy')

    # Load checkpoint if provided
    if args.initial_checkpoint:
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    # Training/Evaluation Loop
    if args.eval:
        evaluate(model, val_loader, criterion, rank)
    else:
        for epoch in range(args.epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, rank)
            evaluate(model, val_loader, criterion, rank)
            if rank == 0:
                save_checkpoint(model, optimizer, epoch, args.output_dir)

    dist.destroy_process_group()

def main():
    args = parse_args()
    mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
    
if __name__ == '__main__':
    main()