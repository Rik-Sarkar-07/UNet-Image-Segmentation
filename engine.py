import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, rank):
    model.train()
    for images, targets in tqdm(dataloader, desc=f"Epoch {epoch}", disable=rank != 0):
        images, targets = images.cuda(rank), targets.cuda(rank)
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()

def evaluate(model, dataloader, criterion, rank):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        imbar = tqdm(dataloader, desc="Evaluation", disable=rank != 0)
        for images, targets in imbar:
            images, targets = images.cuda(rank), targets.cuda(rank)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    if rank == 0:
        print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}")