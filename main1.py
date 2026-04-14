import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
import os
from torchsummary import summary
from tqdm import tqdm 
import statistics
import glob
import joblib
from torch.utils.data import TensorDataset, DataLoader, Dataset
import warnings
import math

# 关闭警告
warnings.filterwarnings("ignore")

# 恢复原有的固定种子逻辑（更快，无随机耗时）
def set_seed(seed=45):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动调优，减少耗时
    os.environ['PYTHONHASHSEED'] = str(seed)

# 自定义数据集类 - 完全恢复原有逻辑
class TrajectoryDataset(Dataset):
    def __init__(self, data, seq_length=1100):
        self.data = data
        self.seq_length = seq_length
        self.num_trajectories = len(data) // seq_length
        self.valid_data = data[:self.num_trajectories * seq_length]
        
    def __len__(self):
        return self.num_trajectories
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        trajectory = self.valid_data[start_idx:end_idx]
        return torch.tensor(trajectory, dtype=torch.float32)

# 模型定义 - 完全不变
class PositionPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ExcavatorAccelerationPredictor:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        # set_seed(42)  # 恢复固定种子，减少随机耗时
        
        # 核心参数 - 完全恢复原有值
        self.trajectory_length = 1100
        self.window_size = 100
        self.seq_length = self.window_size  # 仅为对齐保存逻辑，不影响计算
        self.num_autoreg = 10
        self.bon_p = 0.5
        
        # 模型和归一化器
        self.model = None
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        # Scaler参数
        self.y_scaler_mean = None
        self.y_scaler_scale = None
        self.x_scaler_mean_pos_vel = None
        self.x_scaler_scale_pos_vel = None
        
        # 数据相关
        self.raw_data = None
        self.norm_data = None
        self.timestep = 0.01
        self.sampling_rate = 100
        
        # 数据加载器
        self.train_loader = None
        self.val_loader = None
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_weighted_loss': float('inf'),  # 新增：记录最佳加权损失
            # 新增MAE指标记录
            'val_pos_mae': [],
            'val_vel_mae': []
        }
        
        # 加权参数（可根据需求调整）
        self.train_loss_weight = 0.8  # 训练损失权重
        self.val_loss_weight = 0.2    # 验证损失权重
        
    def load_data(self, csv_file=None, data_dir=None, val_split=0.2, batch_size=8):
        """恢复原有数据加载逻辑，移除耗时的打乱/调小batch操作"""
        # 加载原始数据 - 逻辑不变
        if data_dir:
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            if not csv_files:
                raise ValueError(f"在目录 {data_dir} 中没有找到CSV文件")
            
            all_data = []
            sampling_rates = []
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    features = df[['boom_position','arm_position', 'bucket_position', 
                                  "boom_velocity", "arm_velocity",  "bucket_velocity", 
                                  'pwm_boom','pwm_arm', 'pwm_bucket']].values
                    all_data.append(features)
                    
                    if 'timestamp' in df.columns:
                        timestamps = pd.to_numeric(df['timestamp'])
                        time_diffs = np.diff(timestamps)
                        avg_interval = np.mean(time_diffs)
                        sr = int(np.ceil(1.0 / avg_interval))
                        sampling_rates.append(sr)
                    else:
                        sampling_rates.append(100)
                        
                except Exception as e:
                    print(f"加载文件 {csv_file} 时出错: {str(e)}")
            
            self.raw_data = np.vstack(all_data)
            self.sampling_rate = int(np.mean(sampling_rates))
            
        else:
            df = pd.read_csv(csv_file)
            self.raw_data = df[['boom_position','arm_position', 'bucket_position', 
                               "boom_velocity", "arm_velocity",  "bucket_velocity", 
                               'pwm_boom','pwm_arm', 'pwm_bucket']].values
            
            if 'timestamp' in df.columns:
                timestamps = pd.to_numeric(df['timestamp'])
                time_diffs = np.diff(timestamps)
                avg_interval = np.mean(time_diffs)
                self.sampling_rate = int(np.ceil(1.0 / avg_interval))
        
        self.timestep = 1.0 / self.sampling_rate
        print(f"原始数据形状: {self.raw_data.shape}")
        print(f"采样率: {self.sampling_rate}Hz, 时间步长: {self.timestep:.6f}s")
        
        # 归一化数据 - 逻辑不变
        self.norm_data = self.x_scaler.fit_transform(self.raw_data)
        
        # 恢复scaler Tensor参数 - 逻辑不变
        self.y_scaler_mean = torch.tensor(self.y_scaler.mean_ if hasattr(self.y_scaler, 'mean_') else [0,0,0], 
                                         dtype=torch.float32).to(self.device)
        self.y_scaler_scale = torch.tensor(self.y_scaler.scale_ if hasattr(self.y_scaler, 'scale_') else [1,1,1], 
                                          dtype=torch.float32).to(self.device)
        self.x_scaler_mean_pos_vel = torch.tensor(self.x_scaler.mean_[:6], dtype=torch.float32).to(self.device)
        self.x_scaler_scale_pos_vel = torch.tensor(self.x_scaler.scale_[:6], dtype=torch.float32).to(self.device)
        
        # ========== 恢复原有分割逻辑：移除全局打乱 ==========
        train_size = int(len(self.norm_data) * (1 - val_split))
        train_data = self.norm_data[:train_size]
        val_data = self.norm_data[train_size:]
        
        # 创建数据集 - 逻辑不变
        train_dataset = TrajectoryDataset(train_data, self.trajectory_length)
        val_dataset = TrajectoryDataset(val_data, self.trajectory_length)
        
        # ========== 恢复原有batch_size：不调小 ==========
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"训练轨迹数: {len(train_dataset)}, 验证轨迹数: {len(val_dataset)}")
        print(f"每条轨迹长度: {self.trajectory_length}步")
        
        # 初始化模型 - 新增input_dim但计算逻辑不变
        input_dim = self.window_size * 9  # 本质还是100*9，只是命名对齐
        output_dim = 3
        hidden_dim = 128
        self.model = PositionPredictor(input_dim, output_dim, hidden_dim).to(self.device)
        
        print("模型初始化完成:")
        summary(self.model, input_size=(input_dim,))
    
    # ========== 核心计算逻辑：完全未改 ==========
    def calculate_next_state(self, current_state_norm, accel_pred_norm):
        current_pos_vel_norm = current_state_norm[:, :6]
        current_pos_vel_denorm = current_pos_vel_norm * self.x_scaler_scale_pos_vel + self.x_scaler_mean_pos_vel
        
        accel_pred_denorm = accel_pred_norm * self.y_scaler_scale + self.y_scaler_mean
        
        current_pos_denorm = current_pos_vel_denorm[:, :3]
        current_vel_denorm = current_pos_vel_denorm[:, 3:]
        
        next_vel_denorm = current_vel_denorm + accel_pred_denorm * self.timestep
        next_pos_denorm = current_pos_denorm + (current_vel_denorm + next_vel_denorm) / 2 * self.timestep
        
        next_pos_vel_denorm = torch.cat([next_pos_denorm, next_vel_denorm], dim=1)
        next_pos_vel_norm = (next_pos_vel_denorm - self.x_scaler_mean_pos_vel) / self.x_scaler_scale_pos_vel
        
        pwm_norm = current_state_norm[:, 6:]
        next_state_norm = torch.cat([next_pos_vel_norm[:, :3], next_pos_vel_norm[:, 3:], pwm_norm], dim=1)
        
        return next_state_norm
    
    def bon_select(self, real_window, fake_window, p=0.5):
        batch_size = real_window.shape[0]
        mask = (torch.rand(batch_size, self.window_size, 1, device=self.device) < p).float()
        mixed_window = mask * fake_window + (1 - mask) * real_window
        return mixed_window
    
    def generate_fake_trajectories(self, real_trajectory):
        batch_size = real_trajectory.shape[0]
        fake_trajectories = torch.zeros(batch_size, self.num_autoreg, self.trajectory_length, 9, 
                                       device=self.device)
        
        fake_trajectories[:, :, :self.window_size, :] = real_trajectory[:, :self.window_size, :].unsqueeze(1)
        
        for n in range(self.num_autoreg):
            for i in range(self.window_size, self.trajectory_length):
                if n == 0:
                    window = real_trajectory[:, i-self.window_size:i, :]
                else:
                    window = torch.zeros(batch_size, self.window_size, 9, device=self.device)
                    for k in range(1, min(n+1, self.window_size, self.num_autoreg)):
                        if i - k >= 0:
                            window[:, -k, :] = fake_trajectories[:, n-k, i-k, :]
                    real_start = max(0, self.window_size - min(n, self.num_autoreg))
                    window[:, :real_start, :] = real_trajectory[:, i-self.window_size:i-self.window_size+real_start, :]
                
                window_flat = window.reshape(batch_size, -1)
                accel_pred = self.model(window_flat)
                
                current_state = window[:, -1, :]
                next_state = self.calculate_next_state(current_state, accel_pred)
                
                fake_trajectories[:, n, i, :] = next_state
        
        return fake_trajectories
    
    # ========== 新增：正确的验证损失计算函数 ==========
    def _validate_with_state_update(self, criterion):
        self.model.eval()
        total_loss = 0.0
        total_pos_mae = torch.zeros(3, device=self.device)
        total_vel_mae = torch.zeros(3, device=self.device)
        total_samples = 0
        
        with torch.no_grad():
            for real_trajectory in self.val_loader:
                real_trajectory = real_trajectory.to(self.device)
                batch_size = real_trajectory.shape[0]
                total_samples += batch_size
                
                # 生成fake trajectory（和训练逻辑一致）
                fake_trajectories = self.generate_fake_trajectories(real_trajectory)
                fake_last = fake_trajectories[:, -1, :, :]
                
                # 只计算window_size之后的预测损失（和训练一致）
                pred_pos_vel = fake_last[:, self.window_size:, :6]
                true_pos_vel = real_trajectory[:, self.window_size:, :6]
                
                # 反归一化到真实尺度（关键：验证阶段也要用真实尺度计算损失）
                pred_pos_vel_denorm = pred_pos_vel * self.x_scaler_scale_pos_vel + self.x_scaler_mean_pos_vel
                true_pos_vel_denorm = true_pos_vel * self.x_scaler_scale_pos_vel + self.x_scaler_mean_pos_vel
                
                # 分离位置和速度
                pred_pos = pred_pos_vel_denorm[..., :3]
                true_pos = true_pos_vel_denorm[..., :3]
                pred_vel = pred_pos_vel_denorm[..., 3:]
                true_vel = true_pos_vel_denorm[..., 3:]
                
                # 计算位置和速度损失（权重和训练逻辑对齐）
                pos_loss = criterion(pred_pos, true_pos)
                vel_loss = criterion(pred_vel, true_vel)
                loss = 0.7 * pos_loss + 0.3 * vel_loss
                
                # 按样本数累加损失（核心修正）
                total_loss += loss.item() * batch_size
                
                # 计算MAE指标（按维度）
                pos_abs_error = torch.abs(pred_pos - true_pos)
                vel_abs_error = torch.abs(pred_vel - true_vel)
                
                for i in range(3):
                    total_pos_mae[i] += torch.mean(pos_abs_error[..., i]) * batch_size
                    total_vel_mae[i] += torch.mean(vel_abs_error[..., i]) * batch_size
        
        # 按总样本数计算平均（核心修正）
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_pos_mae = total_pos_mae / total_samples if total_samples > 0 else torch.zeros_like(total_pos_mae)
        avg_vel_mae = total_vel_mae / total_samples if total_samples > 0 else torch.zeros_like(total_vel_mae)
        
        mae_metrics = {
            'boom_pos_mae': avg_pos_mae[0].item(),
            'arm_pos_mae': avg_pos_mae[1].item(),
            'bucket_pos_mae': avg_pos_mae[2].item(),
            'boom_vel_mae': avg_vel_mae[0].item(),
            'arm_vel_mae': avg_vel_mae[1].item(),
            'bucket_vel_mae': avg_vel_mae[2].item()
        }
        
        return avg_loss, mae_metrics
    
    def train(self, epochs=100, lr=5e-3):
        if self.model is None:
            raise ValueError("请先加载数据再训练模型")
        
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        print(f"开始训练，共{epochs}个epochs")
        print(f"轨迹长度: {self.trajectory_length}, 窗口大小: {self.window_size}, 自回归层数: {self.num_autoreg}")
        print(f"伯努利采样概率: {self.bon_p}")
        # 打印加权参数
        print(f"损失加权参数: 训练损失权重={self.train_loss_weight}, 验证损失权重={self.val_loss_weight}")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_samples = 0
            
            for real_trajectory in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                real_trajectory = real_trajectory.to(self.device)
                batch_size = real_trajectory.shape[0]
                train_samples += batch_size
                
                optimizer.zero_grad()
                
                fake_trajectories = self.generate_fake_trajectories(real_trajectory)
                fake_last = fake_trajectories[:, -1, :, :]
                faker = torch.zeros_like(real_trajectory)
                faker[:, :self.window_size, :] = real_trajectory[:, :self.window_size, :]
                
                for i in range(self.window_size, self.trajectory_length):
                    real_window = real_trajectory[:, i-self.window_size:i, :]
                    fake_window = fake_last[:, i-self.window_size:i, :]
                    mixed_window = self.bon_select(real_window, fake_window, self.bon_p)
                    
                    window_flat = mixed_window.reshape(batch_size, -1)
                    accel_pred = self.model(window_flat)
                    
                    current_state = mixed_window[:, -1, :]
                    next_state = self.calculate_next_state(current_state, accel_pred)
                    
                    faker[:, i, :] = next_state
                
                # 训练损失计算（反归一化到真实尺度）
                pred_pos_vel = faker[:, self.window_size:, :6]
                true_pos_vel = real_trajectory[:, self.window_size:, :6]
                
                pred_pos_vel_denorm = pred_pos_vel * self.x_scaler_scale_pos_vel + self.x_scaler_mean_pos_vel
                true_pos_vel_denorm = true_pos_vel * self.x_scaler_scale_pos_vel + self.x_scaler_mean_pos_vel
                
                pred_pos = pred_pos_vel_denorm[..., :3]
                true_pos = true_pos_vel_denorm[..., :3]
                pred_vel = pred_pos_vel_denorm[..., 3:]
                true_vel = true_pos_vel_denorm[..., 3:]
                
                pos_loss = criterion(pred_pos, true_pos)
                vel_loss = criterion(pred_vel, true_vel)
                loss = 0.7 * pos_loss + 0.3 * vel_loss
                
                loss.backward()
                optimizer.step()
                
                # 按样本数累加训练损失
                train_loss += loss.item() * batch_size
            
            # 计算平均训练损失
            avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0
            self.train_history['train_loss'].append(avg_train_loss)
            
            # ========== 使用新的验证函数计算验证损失 ==========
            avg_val_loss, mae_metrics = self._validate_with_state_update(criterion)
            self.train_history['val_loss'].append(avg_val_loss)
            
            # 记录MAE指标
            avg_pos_mae = (mae_metrics['boom_pos_mae'] + mae_metrics['arm_pos_mae'] + mae_metrics['bucket_pos_mae']) / 3
            avg_vel_mae = (mae_metrics['boom_vel_mae'] + mae_metrics['arm_vel_mae'] + mae_metrics['bucket_vel_mae']) / 3
            self.train_history['val_pos_mae'].append(avg_pos_mae)
            self.train_history['val_vel_mae'].append(avg_vel_mae)
            
            scheduler.step(avg_val_loss)
            
            # 计算加权损失（训练损失*训练权重 + 验证损失*验证权重）
            weighted_loss = (avg_train_loss * self.train_loss_weight) + (avg_val_loss * self.val_loss_weight)
            
            # 使用加权损失判定是否保存最佳模型
            if weighted_loss < self.train_history['best_weighted_loss']:
                self.train_history['best_weighted_loss'] = weighted_loss
                self.train_history['best_val_loss'] = avg_val_loss  # 保留原有字段兼容逻辑
                # 只保存和best_model_mae.pth一致的5个核心参数
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'input_dim': self.window_size * 9,    # 900
                    'output_dim': 3,                     # 输出维度
                    'hidden_dim': 128,                   # 隐藏层维度
                    'timestep': self.timestep            # 时间步
                }
                # 移除weights_only参数，兼容低版本PyTorch
                torch.save(checkpoint, 'best_model_autoreg_20.pth')
                print(f"✅ 保存最佳模型 (加权损失: {weighted_loss:.6f}, 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}) - 仅保留核心5个参数")
                self.save_scaler_separately()
            
            # 每轮打印loss - 增加加权损失和MAE展示
            print(f'Epoch [{epoch+1:3d}/{epochs}]: ')
            print(f'  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Weighted Loss: {weighted_loss:.6f}')
            print(f'  Val Pos MAE: {avg_pos_mae:.6f} | Val Vel MAE: {avg_vel_mae:.6f}')
            print(f'  - Boom Pos MAE: {mae_metrics["boom_pos_mae"]:.6f}, Arm Pos MAE: {mae_metrics["arm_pos_mae"]:.6f}, Bucket Pos MAE: {mae_metrics["bucket_pos_mae"]:.6f}')
            print(f'  - Boom Vel MAE: {mae_metrics["boom_vel_mae"]:.6f}, Arm Vel MAE: {mae_metrics["arm_vel_mae"]:.6f}, Bucket Vel MAE: {mae_metrics["bucket_vel_mae"]:.6f}')
        
        self.plot_training_history()
    
    def save_scaler_separately(self, scaler_path='scalers_mae.pkl'):
        """训练完成后手动调用一次，保存归一化器到独立文件，不混入模型参数"""
        scalers = {
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler,
            'timestep': self.timestep,
            'y_scaler_mean': self.y_scaler_mean.cpu().numpy() if self.y_scaler_mean is not None else np.array([0,0,0]),
            'y_scaler_scale': self.y_scaler_scale.cpu().numpy() if self.y_scaler_scale is not None else np.array([1,1,1]),
            'x_scaler_mean_pos_vel': self.x_scaler_mean_pos_vel.cpu().numpy(),
            'x_scaler_scale_pos_vel': self.x_scaler_scale_pos_vel.cpu().numpy(),
            'window_size': self.window_size,
            'trajectory_length': self.trajectory_length,
            'num_autoreg': self.num_autoreg,
            # ========== 新增：保存scaler的完整拟合参数 ==========
            'x_scaler_mean_': self.x_scaler.mean_,
            'x_scaler_scale_': self.x_scaler.scale_,
            'x_scaler_var_': self.x_scaler.var_,
            'x_scaler_n_samples_seen_': self.x_scaler.n_samples_seen_,
        }
        joblib.dump(scalers, scaler_path)
        print(f"📌 归一化器已保存至独立文件: {scaler_path}")
        # 打印保存的关键参数，方便调试
        print(f"📝 保存的x_scaler均值（前3维）: {self.x_scaler.mean_[:3]}")
        print(f"📝 保存的x_scaler标准差（前3维）: {self.x_scaler.scale_[:3]}")
        
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 绘制损失曲线
        ax1.plot(self.train_history['train_loss'], label='Train Loss')
        ax1.plot(self.train_history['val_loss'], label='Validation Loss')
        # 计算并绘制加权损失曲线
        weighted_losses = [
            (self.train_history['train_loss'][i] * self.train_loss_weight) + 
            (self.train_history['val_loss'][i] * self.val_loss_weight) 
            for i in range(len(self.train_history['train_loss']))
        ]
        ax1.plot(weighted_losses, label='Weighted Loss', linestyle='--')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MAE Loss')
        ax1.set_title('Training and Validation Loss (20-Layer Autoregressive + Bernoulli Sampling)')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制MAE曲线
        ax2.plot(self.train_history['val_pos_mae'], label='Validation Position MAE', color='blue')
        ax2.plot(self.train_history['val_vel_mae'], label='Validation Velocity MAE', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Validation Position and Velocity MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_autoreg_20.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='挖掘机轨迹预测 - 20层自回归（高效版本）')
    parser.add_argument('--csv', type=str, default="data/data_med.csv", help='CSV数据文件路径')
    parser.add_argument('--data_dir', type=str, default=None, help='包含多个CSV文件的目录路径')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-3, help='学习率')
    parser.add_argument('--bon_p', type=float, default=0.5, help='伯努利采样概率')
    # 新增：允许通过命令行调整加权参数
    parser.add_argument('--train_weight', type=float, default=0.7, help='训练损失权重（默认0.7）')
    parser.add_argument('--val_weight', type=float, default=0.3, help='验证损失权重（默认0.3）')
    
    args = parser.parse_args()
    
    model = ExcavatorAccelerationPredictor()
    model.bon_p = args.bon_p
    # 设置加权参数
    model.train_loss_weight = args.train_weight
    model.val_loss_weight = args.val_weight
    # 校验权重和为1（可选）
    if abs(model.train_loss_weight + model.val_loss_weight - 1.0) > 1e-6:
        print(f"⚠️  警告：训练损失权重({model.train_loss_weight}) + 验证损失权重({model.val_loss_weight}) ≠ 1.0")
    
    if args.data_dir:
        model.load_data(data_dir=args.data_dir, batch_size=args.batch_size)
    else:
        model.load_data(csv_file=args.csv, batch_size=args.batch_size)
    
    model.train(epochs=args.epochs, lr=args.lr)
    
    # 训练完成后手动保存一次归一化器到独立文件（不影响模型参数文件）
    
    
    print("\n🎉 训练完成！")
    print(f"🏆 最佳加权损失: {model.train_history['best_weighted_loss']:.6f}")
    print(f"🏆 最佳验证损失: {model.train_history['best_val_loss']:.6f}")
    print("📄 模型参数文件: best_model_autoreg_20.pth (仅包含5个核心参数)")
    print("📄 归一化器文件: scalers_mae.pkl (独立保存，不混入模型参数)")

if __name__ == '__main__':
    main()