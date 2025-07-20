import torch

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path, scheduler):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)  # 保存检查点
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, checkpoint_path, scheduler):
    checkpoint = torch.load(checkpoint_path)  # 加载检查点
    model.load_state_dict(checkpoint['model_state_dict'])  # 恢复模型的参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 恢复优化器的参数
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 恢复调度器状态
    epoch = checkpoint['epoch']  # 恢复 epoch
    loss = checkpoint['loss']  # 恢复最后的损失
    print(f"Checkpoint loaded from epoch {epoch}")
    return model, optimizer, epoch, loss, scheduler
