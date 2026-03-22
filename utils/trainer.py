# 训练器类（已创建）
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import json
import numpy as np


class FasterRCNNTrainer:
    """Faster R-CNN训练器"""

    def __init__(self, model, train_loader, val_loader, config=None, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device

        self.num_epochs = self.config.get('num_epochs', 50)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.weight_decay = self.config.get('weight_decay', 0.0005)
        self.eval_interval = self.config.get('eval_interval', 5)
        self.save_interval = self.config.get('save_interval', 10)

        self.weights_dir = self.config.get('weights_dir', 'weights')
        self.logs_dir = self.config.get('logs_dir', 'logs')
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6
        )

        self.train_history = {
            'loss': [],
            'epoch': [],
            'learning_rate': []
        }

        self.best_loss = float('inf')
        self.start_epoch = 1

    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for images, targets in pbar:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def evaluate(self, epoch):
        """评估模型"""
        self.model.eval()
        print(f"  Epoch {epoch} 评估完成")

    def train(self):
        """完整训练流程"""
        print("=" * 70)
        print("开始训练Faster R-CNN模型")
        print("=" * 70)

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 70)

            avg_loss = self.train_one_epoch(epoch)
            print(f"\n训练损失: {avg_loss:.4f}")

            self.train_history['loss'].append(avg_loss)
            self.train_history['epoch'].append(epoch)
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            self.scheduler.step()
            print(f"当前学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            if epoch % self.eval_interval == 0:
                self.evaluate(epoch)

            if epoch % self.save_interval == 0:
                save_path = os.path.join(self.weights_dir, f'checkpoint_epoch_{epoch}.pth')
                self.save_checkpoint(epoch, avg_loss, save_path)

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                save_path = os.path.join(self.weights_dir, 'best_model.pth')
                self.save_checkpoint(epoch, avg_loss, save_path)
                print(f"  ✓ 保存最佳模型 (loss: {self.best_loss:.4f})")

            # 保存训练历史
            history_path = os.path.join(self.logs_dir, 'train_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.train_history, f, indent=2)

        print("\n" + "=" * 70)
        print("训练完成！")
        print(f"最佳损失: {self.best_loss:.4f}")
        print("=" * 70)

    def save_checkpoint(self, epoch, loss, save_path):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'train_history': self.train_history
        }
        torch.save(checkpoint, save_path)
        print(f"  检查点已保存: {save_path}")