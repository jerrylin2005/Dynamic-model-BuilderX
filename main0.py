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
from torch.utils.data import TensorDataset, DataLoader
import warnings
import math

# 关闭警告
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    """固定所有随机种子，保证训练可复现"""
    # Python内置随机库
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU（单卡）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多卡场景
    # 关闭cuDNN的非确定性算法（关键：消除GPU计算的随机误差）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置Python哈希种子（避免字典/集合遍历顺序随机）
    os.environ['PYTHONHASHSEED'] = str(seed)

# 调用函数固定种子（seed选任意固定值，比如42）
set_seed(42)

# 神经网络模型定义
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
        # 设置计算设备
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型（稍后在load_data中根据数据确定输入维度）
        self.model = None
        self.plot = None
        # 初始化归一化器
        self.x_scaler = None
        self.y_scaler = None
        
        # 存储scaler参数为Tensor（用于保持梯度）
        self.y_scaler_mean = None
        self.y_scaler_scale = None
        self.x_scaler_mean_pos_vel = None
        self.x_scaler_scale_pos_vel = None
        
        # 数据存储
        self.X_seq = None
        self.y_seq = None
        self.df = None
        self.sampling_rate = None
        self.timestep = None  # 添加时间步长存储
        self.seq_length = 100  # 固定使用100个时间步的滑动窗口
        
        self.global_step = 0
        self.sched_params = {
            'p_start': 1.0,    # 初始 teacher forcing 概率（1.0 表示全 teacher）
            'p_end': 0.0,      # 最终下限（0.0 表示可衰减到完全自回归）
            'k': 7e-6,          # 衰减速率
            'unroll': 3,        # multi-step unroll 长度（保守推荐 3）
            'weights': [1.0, 0.5, 0.2]  # multi-step 损失权重
        }
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'boom_pos_mae': [],
            'arm_pos_mae': [],
            'bucket_pos_mae': [],
            'boom_vel_mae': [],
            'arm_vel_mae': [],
            'bucket_vel_mae': []
        }
        
        # 数据加载器
        self.train_loader = None
        self.test_loader = None
        self.free_run_loader = None
        
        # 存储下一个时间步的真实位置和速度（用于训练）
        self.next_positions = None
        self.next_velocities = None
    
    def load_data(self, csv_file=None, data_dir=None):
        """
        加载并预处理数据
        
        参数:
            csv_file: 单个CSV文件路径
            data_dir: 包含多个CSV文件的目录路径
        """
        if data_dir:
            # 从目录加载多个数据集
            X, y, next_pos, next_vel, self.df, self.sampling_rate, self.timestep = self._preprocessing_with_next_state(data_dir)
        else:
            # 加载单个数据集
            X, y, next_pos, next_vel, self.df, self.sampling_rate, self.timestep = self._preprocessing_with_next_state(csv_file)
        
        print(f"原始数据形状: X={X.shape}, y={y.shape}")
        print(f"采样率: {self.sampling_rate}Hz, 时间步长: {self.timestep:.6f}s")
        
        # 归一化数据
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X_normalized = self.x_scaler.fit_transform(X)
        y_normalized = self.y_scaler.fit_transform(y)
        
        # 保存scaler参数为Tensor（用于保持梯度）
        self.y_scaler_mean = torch.tensor(self.y_scaler.mean_, dtype=torch.float32).to(self.device)
        self.y_scaler_scale = torch.tensor(self.y_scaler.scale_, dtype=torch.float32).to(self.device)
        
        # 位置和速度的scaler参数（前6个特征）
        self.x_scaler_mean_pos_vel = torch.tensor(self.x_scaler.mean_[:6], dtype=torch.float32).to(self.device)
        self.x_scaler_scale_pos_vel = torch.tensor(self.x_scaler.scale_[:6], dtype=torch.float32).to(self.device)
        
        # 创建序列数据（使用100个时间步的滑动窗口）
        unroll = int(self.sched_params.get('unroll', 3))
        self.X_seq, self.y_seq, self.next_positions, self.next_velocities = self._create_sequences_with_next_state(
            X_normalized, y_normalized, next_pos, next_vel, self.seq_length, unroll=unroll
        )
        
        print(f"序列数据形状: X_seq={self.X_seq.shape}, y_seq={self.y_seq.shape}")
        print(f"下一时刻（multi-step）位置形状: {self.next_positions.shape}, 下一时刻速度形状: {self.next_velocities.shape}")
        # next_positions shape: (N, unroll, 3)
        
        # 分割数据集 (80/20)
        indices = np.arange(len(self.X_seq))
        X_train_seq, X_test_seq, y_train_seq, y_test_seq, next_pos_train, next_pos_test, next_vel_train, next_vel_test, idx_train, idx_test = train_test_split(
            self.X_seq, self.y_seq, self.next_positions, self.next_velocities, indices, 
            test_size=0.2, shuffle=False, random_state=42
        )
        
        # 转换为 PyTorch 张量（位置/速度变为多步）
        X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)  # step0 accel targets (normalized)
        next_pos_train_t = torch.tensor(next_pos_train, dtype=torch.float32)  # shape (N, unroll, 3) denorm
        next_vel_train_t = torch.tensor(next_vel_train, dtype=torch.float32)

        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_seq, dtype=torch.float32)
        next_pos_test_t = torch.tensor(next_pos_test, dtype=torch.float32)
        next_vel_test_t = torch.tensor(next_vel_test, dtype=torch.float32)
        
        # 创建数据集 - 训练集需要包含下一时刻的位置和速度
        train_dataset = TensorDataset(X_train_t, y_train_t, next_pos_train_t, next_vel_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t, next_pos_test_t, next_vel_test_t)
        
        # 创建数据加载器
        batch_size = 256
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型（输入维度 = 序列长度100 * 特征数9 = 900）
        input_dim = self.seq_length * 9
        output_dim = 3  # 仍然只输出加速度
        hidden_dim = 128  # 保持原论文指定的128节点
        
        self.model = PositionPredictor(input_dim, output_dim, hidden_dim).to(self.device)
        
        print("模型初始化完成:")
        print(self.model)
        summary(self.model, input_size=(input_dim,))
    
    def train(self, epochs=100, lr=5e-3):

        if self.model is None:
            raise ValueError("请先加载数据再训练模型")
        
        # 初始化优化器和损失函数 - 使用MAE损失
        criterion = nn.L1Loss()  # MAE损失
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        
        print(f"开始训练，共{epochs}个epochs，使用MAE损失和状态更新模拟")
        print(f"梯度保持机制已启用: 所有反归一化操作都保持梯度连接")
        print(f"使用参数： p_start={self.sched_params['p_start']}, p_end={self.sched_params['p_end']}, k={self.sched_params['k']}, unroll={self.sched_params['unroll']}, weights={self.sched_params['weights']}")
        
        for epoch in range(epochs):
            # 训练步骤 - 包含状态更新
            train_loss, train_mae_metrics = self._train_step_with_state_update(optimizer, criterion)
            
            # 验证步骤 - 包含状态更新
            val_loss, val_mae_metrics = self._validate_with_state_update(criterion)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 保存训练历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            
            # 保存MAE指标
            for key, value in train_mae_metrics.items():
                if key not in self.train_history:
                    self.train_history[key] = []
                self.train_history[key].append(value)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model_mae.pth')
            
            # 打印训练进度
            if (epoch+1) % 5 == 0 or epoch < 10:
                p_teacher = self.sched_params['p_end'] + (self.sched_params['p_start'] - self.sched_params['p_end']) * np.exp(-self.sched_params['k'] * self.global_step)
                print(f'Epoch [{(epoch+1):3d}/{epochs}]: p={p_teacher:.3f} Train MAE*1000 [{train_loss*1000:.3f}] Val MAE*1000 [{val_loss*1000:.3f}]')
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model_mae.pth'))
        
        # 绘制训练历史
        self._plot_training_history()
    
    def _train_step_with_state_update(self, optimizer, criterion):
        self.model.train()
        total_loss = 0
        total_pos_mae = torch.zeros(3, device=self.device)
        total_vel_mae = torch.zeros(3, device=self.device)
        total_samples = 0

        for inputs, accel_targets, next_pos_targets, next_vel_targets in self.train_loader:
            inputs = inputs.to(self.device)
            accel_targets = accel_targets.to(self.device)
            next_pos_targets = next_pos_targets.to(self.device)
            next_vel_targets = next_vel_targets.to(self.device)

            batch_size = inputs.shape[0]
            total_samples += batch_size

            current_window = inputs.clone()
            unroll = int(self.sched_params.get('unroll', 3))
            weights = self.sched_params.get('weights', [1.0]*unroll)
            assert len(weights) >= unroll, "weights 长度必须 >= unroll"

            # scheduled sampling
            t = float(self.global_step)
            p_start = self.sched_params['p_start']
            p_end = self.sched_params['p_end']
            k_decay = self.sched_params['k']
            p_teacher = p_end + (p_start - p_end) * math.exp(-k_decay * t)

            # 记录 multi-step predictions
            pred_pos_denorm_list = []
            pred_vel_denorm_list = []
            pred_accel_norm_list = []

            loss_total = 0.0

            for u in range(unroll):
                # flatten window -> model
                model_input = current_window.reshape(batch_size, -1)
                accel_pred_norm = self.model(model_input)  # (B,3)
                pred_accel_norm_list.append(accel_pred_norm)

                # double integration -> denorm next state
                accel_pred_denorm = accel_pred_norm * self.y_scaler_scale + self.y_scaler_mean
                last_pos_denorm = current_window[:, -1, :3] * self.x_scaler_scale_pos_vel[:3] + self.x_scaler_mean_pos_vel[:3]
                last_vel_denorm = current_window[:, -1, 3:6] * self.x_scaler_scale_pos_vel[3:] + self.x_scaler_mean_pos_vel[3:]
                next_vel_denorm = last_vel_denorm + accel_pred_denorm * self.timestep
                next_pos_denorm = last_pos_denorm + (last_vel_denorm + next_vel_denorm)/2.0 * self.timestep

                pred_pos_denorm_list.append(next_pos_denorm)
                pred_vel_denorm_list.append(next_vel_denorm)

                # normalize back
                next_pos_norm = (next_pos_denorm - self.x_scaler_mean_pos_vel[:3]) / self.x_scaler_scale_pos_vel[:3]
                next_vel_norm = (next_vel_denorm - self.x_scaler_mean_pos_vel[3:]) / self.x_scaler_scale_pos_vel[3:]
                pwm_norm = current_window[:, -1, 6:9]
                model_next_row_norm = torch.cat([next_pos_norm, next_vel_norm, pwm_norm], dim=1)

                # GT for step u
                if next_pos_targets.ndim == 3 and next_pos_targets.shape[1] > u:
                    gt_pos_u = next_pos_targets[:, u, :]
                    gt_vel_u = next_vel_targets[:, u, :]
                    gt_pos_norm_u = (gt_pos_u - self.x_scaler_mean_pos_vel[:3]) / self.x_scaler_scale_pos_vel[:3]
                    gt_vel_norm_u = (gt_vel_u - self.x_scaler_mean_pos_vel[3:]) / self.x_scaler_scale_pos_vel[3:]
                    gt_next_row_norm = torch.cat([gt_pos_norm_u, gt_vel_norm_u, pwm_norm], dim=1)
                else:
                    gt_next_row_norm = None

                # scheduled sampling
                if gt_next_row_norm is not None:
                    use_teacher = (torch.rand(batch_size, device=self.device) < p_teacher).float().unsqueeze(1).repeat(1,9)
                    chosen_next_row = use_teacher * gt_next_row_norm + (1.0 - use_teacher) * model_next_row_norm.detach()
                else:
                    chosen_next_row = model_next_row_norm.detach()

                # slide window
                current_window = torch.roll(current_window, shifts=-1, dims=1)
                current_window[:, -1, :] = chosen_next_row

                # multi-step weighted loss
                w = weights[u]
                if gt_next_row_norm is not None:
                    pos_loss_u = criterion(next_pos_denorm, gt_pos_u)
                    vel_loss_u = criterion(next_vel_denorm, gt_vel_u)
                    loss_total += w * (0.8 * pos_loss_u + 0.2 * vel_loss_u)

            # backward & optimize
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            total_loss += loss_total.item()

            # step0 MAE metrics
            pos_target_step0 = next_pos_targets[:, 0, :] if next_pos_targets.ndim==3 else next_pos_targets
            vel_target_step0 = next_vel_targets[:, 0, :] if next_vel_targets.ndim==3 else next_vel_targets
            first_next_positions_pred_denorm = pred_pos_denorm_list[0]
            first_next_velocities_pred_denorm = pred_vel_denorm_list[0]

            with torch.no_grad():
                for i in range(3):
                    total_pos_mae[i] += torch.mean(torch.abs(first_next_positions_pred_denorm[:, i] - pos_target_step0[:, i])) * batch_size
                    total_vel_mae[i] += torch.mean(torch.abs(first_next_velocities_pred_denorm[:, i] - vel_target_step0[:, i])) * batch_size

            self.global_step += 1

        avg_loss = total_loss / len(self.train_loader)
        avg_pos_mae = total_pos_mae / total_samples
        avg_vel_mae = total_vel_mae / total_samples

        mae_metrics = {
            'boom_pos_mae': avg_pos_mae[0].item(),
            'arm_pos_mae': avg_pos_mae[1].item(),
            'bucket_pos_mae': avg_pos_mae[2].item(),
            'boom_vel_mae': avg_vel_mae[0].item(),
            'arm_vel_mae': avg_vel_mae[1].item(),
            'bucket_vel_mae': avg_vel_mae[2].item()
        }

        return avg_loss, mae_metrics

    
    def _validate_with_state_update(self, criterion):
        """
        验证步骤 - 包含双重积分状态更新
        修改：使用Tensor操作保持梯度连续性
        
        参数:
            criterion: 损失函数（MAE）
            
        返回:
            total_loss: 总损失
            mae_metrics: MAE指标字典
        """
        self.model.eval()
        total_loss = 0
        total_pos_mae = torch.zeros(3, device=self.device)
        total_vel_mae = torch.zeros(3, device=self.device)
        total_samples = 0
        
        with torch.no_grad():
            for inputs, accel_targets, next_pos_targets, next_vel_targets in self.test_loader:
                inputs = inputs.to(self.device)
                accel_targets = accel_targets.to(self.device)
                next_pos_targets = next_pos_targets.to(self.device)
                next_vel_targets = next_vel_targets.to(self.device)
                
                batch_size = inputs.shape[0]
                total_samples += batch_size
                
                # 重塑输入以匹配模型期望的形状
                inputs_reshaped = inputs.reshape(batch_size, -1)
                
                # 前向传播 - 预测加速度
                accel_preds = self.model(inputs_reshaped)
                
                # === 修改：保持梯度的反归一化 ===
                accel_preds_denorm = accel_preds * self.y_scaler_scale + self.y_scaler_mean
                
                # 从输入中提取当前状态
                current_positions = inputs[:, -1, :3]
                current_velocities = inputs[:, -1, 3:6]
                
                # === 修改：保持梯度的状态反归一化 ===
                current_state = torch.cat([current_positions, current_velocities], dim=1)
                current_state_denorm = current_state * self.x_scaler_scale_pos_vel + self.x_scaler_mean_pos_vel
                
                current_positions_denorm = current_state_denorm[:, :3]
                current_velocities_denorm = current_state_denorm[:, 3:]
                
                # 使用预测的加速度进行双重积分更新状态
                next_velocities_pred = current_velocities_denorm + accel_preds_denorm * self.timestep
                next_positions_pred = current_positions_denorm + (current_velocities_denorm + next_velocities_pred) / 2 * self.timestep
                
                # 方案：validation 也只用 step0 的 GT
                pos_target_step0 = next_pos_targets[:, 0, :] if next_pos_targets.ndim == 3 else next_pos_targets
                vel_target_step0 = next_vel_targets[:, 0, :] if next_vel_targets.ndim == 3 else next_vel_targets
                
                assert next_positions_pred.shape == pos_target_step0.shape, \
                    f"[VAL] pos pred {next_positions_pred.shape} vs gt {pos_target_step0.shape}"

                assert next_velocities_pred.shape == vel_target_step0.shape, \
                    f"[VAL] vel pred {next_velocities_pred.shape} vs gt {vel_target_step0.shape}"

                pos_loss = criterion(next_positions_pred, pos_target_step0)
                vel_loss = criterion(next_velocities_pred, vel_target_step0)
                
                # 组合损失 - 使用与训练时相同的权重
                accel_loss = criterion(accel_preds, accel_targets)
                loss = 0 * accel_loss + 0.7 * pos_loss + 0.3 * vel_loss
                total_loss += loss.item()
                
                pos_target_step0 = next_pos_targets[:, 0, :] if next_pos_targets.ndim == 3 else next_pos_targets
                vel_target_step0 = next_vel_targets[:, 0, :] if next_vel_targets.ndim == 3 else next_vel_targets
                
                # 累加MAE指标
                for i in range(3):
                    total_pos_mae[i] += torch.mean(torch.abs(next_positions_pred[:, i] - pos_target_step0[:, i])) * batch_size
                    total_vel_mae[i] += torch.mean(torch.abs(next_velocities_pred[:, i] - vel_target_step0[:, i])) * batch_size
        
        # 计算平均指标
        avg_loss = total_loss / len(self.test_loader)
        avg_pos_mae = total_pos_mae / total_samples
        avg_vel_mae = total_vel_mae / total_samples
        
        mae_metrics = {
            'boom_pos_mae': avg_pos_mae[0].item(),
            'arm_pos_mae': avg_pos_mae[1].item(),
            'bucket_pos_mae': avg_pos_mae[2].item(),
            'boom_vel_mae': avg_vel_mae[0].item(),
            'arm_vel_mae': avg_vel_mae[1].item(),
            'bucket_vel_mae': avg_vel_mae[2].item()
        }
        
        return avg_loss, mae_metrics
    
    def predict(self, input_data, denormalize=True):
        """
        预测加速度（支持批次处理）
        
        参数:
            input_data: 输入数据，可以是单个样本或批次样本
            denormalize: 是否反归一化输出
            
        返回:
            加速度预测值
        """
        self.model.eval()
        
        # 确保输入数据是张量
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        
        input_data = input_data.to(self.device)
        
        with torch.no_grad():
            # 重塑输入以匹配模型期望的形状
            if input_data.dim() == 3:  # 批次数据: (batch_size, seq_len, features)
                batch_size = input_data.shape[0]
                input_data = input_data.reshape(batch_size, -1)
            elif input_data.dim() == 2:  # 单个序列: (seq_len, features)
                input_data = input_data.unsqueeze(0).reshape(1, -1)
            elif input_data.dim() == 1:  # 展平的数据
                input_data = input_data.unsqueeze(0)
            
            # 预测加速度
            output = self.model(input_data).cpu().numpy()
            
            # 反归一化
            if denormalize:
                output = self.y_scaler.inverse_transform(output)
            
            return output
    
    def run_free_run_experiments(self, num_simulations=85, steps=1000):
        """
        运行自由运行实验 - 使用MAE和RMSE评估
        
        参数:
            num_simulations: 模拟次数
            steps: 每次模拟步数
            
        返回:
            模拟结果
        """
        if self.model is None:
            raise ValueError("请先加载数据再运行模拟")
        
        print(f"开始自由运行模拟: {num_simulations}轮, 每轮{steps}步")
        
        # 运行详细自由运行实验，使用MAE和RMSE评估
        simulation_results = self._run_free_run_experiments_with_mae_rmse(
            num_simulations=num_simulations, steps=steps
        )
        
        return simulation_results
    
    def save_model(self, model_path='best_model_mae.pth', scaler_path='scalers_mae.pkl'):
        """
        保存模型和归一化器
        """
        if self.model is None:
            raise ValueError("没有模型可保存")
        
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.seq_length * 9,
            'output_dim': 3,
            'hidden_dim': 128,
            'timestep': self.timestep
        }, model_path)
        
        # 保存归一化器
        scalers = {
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler,
            'timestep': self.timestep,
            'y_scaler_mean': self.y_scaler_mean.cpu().numpy(),
            'y_scaler_scale': self.y_scaler_scale.cpu().numpy(),
            'x_scaler_mean_pos_vel': self.x_scaler_mean_pos_vel.cpu().numpy(),
            'x_scaler_scale_pos_vel': self.x_scaler_scale_pos_vel.cpu().numpy()
        }
        joblib.dump(scalers, scaler_path)
        
        print(f"模型已保存至: {model_path}")
        print(f"归一化器已保存至: {scaler_path}")
    
    def load_model(self, model_path='best_model_mae.pth', scaler_path='scalers_mae.pkl'):
        """
        加载模型和归一化器（修复NotFittedError问题）
        """
        # ========== 1. 加载模型（原有逻辑不变） ==========
        checkpoint = torch.load(model_path, map_location=self.device)
        input_dim = checkpoint['input_dim']
        output_dim = checkpoint['output_dim']
        hidden_dim = checkpoint['hidden_dim']
        self.timestep = checkpoint.get('timestep', 0.01)
        
        self.model = PositionPredictor(input_dim, output_dim, hidden_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # ========== 2. 加载归一化器（核心修复） ==========
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"归一化器文件 {scaler_path} 不存在！")
        
        scalers = joblib.load(scaler_path)
        
        # 方案1：优先恢复原有scaler实例（如果保存完整）
        self.x_scaler = scalers.get('x_scaler', StandardScaler())
        self.y_scaler = scalers.get('y_scaler', StandardScaler())
        
        # 方案2：强制手动恢复拟合参数（关键修复）
        # 即使scaler实例不完整，也手动设置拟合参数，确保可以调用inverse_transform
        # ---- 恢复x_scaler ----
        if hasattr(self.x_scaler, 'mean_') and len(self.x_scaler.mean_) > 0:
            pass  # 已有拟合参数，无需处理
        else:
            # 从保存的参数中恢复x_scaler的拟合值
            self.x_scaler.mean_ = scalers.get('x_scaler_mean_', np.zeros(9))
            self.x_scaler.scale_ = scalers.get('x_scaler_scale_', np.ones(9))
            self.x_scaler.var_ = scalers.get('x_scaler_var_', np.ones(9))
            self.x_scaler.n_samples_seen_ = scalers.get('x_scaler_n_samples_seen_', 1)
            self.x_scaler.n_features_in_ = 9  # 固定9个输入特征
        
        # ---- 恢复y_scaler（重点修复） ----
        if hasattr(self.y_scaler, 'mean_') and len(self.y_scaler.mean_) > 0:
            pass  # 已有拟合参数，无需处理
        else:
            # 从保存的参数中恢复y_scaler的拟合值
            self.y_scaler.mean_ = scalers.get('y_scaler_mean', np.array([0.0, 0.0, 0.0]))
            self.y_scaler.scale_ = scalers.get('y_scaler_scale', np.array([1.0, 1.0, 1.0]))
            self.y_scaler.var_ = self.y_scaler.scale_ **2  # 方差=标准差平方
            self.y_scaler.n_samples_seen_ = scalers.get('x_scaler_n_samples_seen_', 1)
            self.y_scaler.n_features_in_ = 3  # 固定3个输出特征
        
        # ========== 3. 恢复其他参数（原有逻辑） ==========
        self.timestep = scalers.get('timestep', self.timestep)
        
        # 恢复scaler参数为Tensor
        self.y_scaler_mean = torch.tensor(
            scalers.get('y_scaler_mean', np.array([0,0,0])), 
            dtype=torch.float32
        ).to(self.device)
        self.y_scaler_scale = torch.tensor(
            scalers.get('y_scaler_scale', np.array([1,1,1])), 
            dtype=torch.float32
        ).to(self.device)
        self.x_scaler_mean_pos_vel = torch.tensor(
            scalers.get('x_scaler_mean_pos_vel', np.zeros(6)), 
            dtype=torch.float32
        ).to(self.device)
        self.x_scaler_scale_pos_vel = torch.tensor(
            scalers.get('x_scaler_scale_pos_vel', np.ones(6)), 
            dtype=torch.float32
        ).to(self.device)
        
        # 设置序列长度
        self.seq_length = input_dim // 9
        
        # ========== 4. 验证scaler是否可用（新增） ==========
        try:
            # 测试y_scaler的inverse_transform
            test_pred = np.array([[0.0, 0.0, 0.0]])
            test_denorm = self.y_scaler.inverse_transform(test_pred)
            print(f"✅ y_scaler验证通过：逆归一化测试结果 = {test_denorm[0]}")
            
            # 测试x_scaler
            test_x = np.array([[0.0]*9])
            test_x_denorm = self.x_scaler.inverse_transform(test_x)
            print(f"✅ x_scaler验证通过：逆归一化测试结果 = {test_x_denorm[0][:3]} (前3维)")
        except Exception as e:
            raise RuntimeError(f"❌ 归一化器验证失败: {str(e)}")
        
        # ========== 5. 打印加载信息 ==========
        print(f"模型已从 {model_path} 加载")
        print(f"归一化器已从 {scaler_path} 加载")
        print(f"序列长度: {self.seq_length}, 时间步长: {self.timestep:.6f}s")
        print(f"成功加载预训练模型：{model_path}")
        print(f"成功加载归一化器：{scaler_path}")
    
    def _preprocessing_with_next_state(self, csv_file=None, data_dir=None):
        """
        数据预处理 - 包含下一时刻状态信息
        
        返回:
            X: 特征数据
            y: 加速度目标
            next_positions: 下一时刻位置
            next_velocities: 下一时刻速度
            df: 数据框
            sr: 采样率
            timestep: 时间步长
        """
        if data_dir:
            # 从目录加载多个数据集
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            if not csv_files:
                raise ValueError(f"在目录 {data_dir} 中没有找到CSV文件")
            
            all_X, all_y, all_next_pos, all_next_vel, all_dfs = [], [], [], [], []
            sampling_rates = []
            
            for csv_file in csv_files:
                try:
                    X, y, next_pos, next_vel, df, sr, timestep = self._preprocess_single_file(csv_file)
                    all_X.append(X)
                    all_y.append(y)
                    all_next_pos.append(next_pos)
                    all_next_vel.append(next_vel)
                    all_dfs.append(df)
                    sampling_rates.append(sr)
                    print(f"成功加载: {csv_file} (采样率: {sr}Hz, 样本数: {len(df)})")
                except Exception as e:
                    print(f"加载文件 {csv_file} 时出错: {str(e)}")
            
            if not all_X:
                raise ValueError("没有成功加载任何数据文件")
            
            # 合并所有数据
            X_combined = np.vstack(all_X)
            y_combined = np.vstack(all_y)
            next_pos_combined = np.vstack(all_next_pos)
            next_vel_combined = np.vstack(all_next_vel)
            df_combined = pd.concat(all_dfs, ignore_index=True)
            avg_sr = int(np.mean(sampling_rates))
            timestep = 1.0 / avg_sr
            
            return X_combined, y_combined, next_pos_combined, next_vel_combined, df_combined, avg_sr, timestep
        else:
            # 加载单个文件
            return self._preprocess_single_file(csv_file)
    
    def _preprocess_single_file(self, csv_file):
        """预处理单个文件"""
        df = pd.read_csv(csv_file)

        # 计算采样率和时间步长
        sr = self._calculate_sampling_rate(df)
        timestep = 1.0 / sr
        
        # 创建移位目标
        df['Next_boom_position'] = df['boom_position'].shift(-1)
        df['Next_arm_position'] = df['arm_position'].shift(-1)
        df['Next_bucket_position'] = df['bucket_position'].shift(-1)
        df['Next_boom_velocity'] = df['boom_velocity'].shift(-1)
        df['Next_arm_velocity'] = df['arm_velocity'].shift(-1)
        df['Next_bucket_velocity'] = df['bucket_velocity'].shift(-1)
        
        # 计算加速度 (通过速度差分)
        df['boom_acceleration'] = (df['Next_boom_velocity'] - df['boom_velocity']) / timestep
        df['arm_acceleration'] = (df['Next_arm_velocity'] - df['arm_velocity']) / timestep
        df['bucket_acceleration'] = (df['Next_bucket_velocity'] - df['bucket_velocity']) / timestep
        
        df = df.dropna().reset_index(drop=True)

        # 特征和目标
        X = df[['boom_position','arm_position', 'bucket_position', 
                "boom_velocity", "arm_velocity",  "bucket_velocity", 
                'pwm_boom','pwm_arm', 'pwm_bucket']].values
        y = df[['boom_acceleration', 'arm_acceleration', 'bucket_acceleration']].values
        
        # 下一时刻的位置和速度
        next_positions = df[['Next_boom_position', 'Next_arm_position', 'Next_bucket_position']].values
        next_velocities = df[['Next_boom_velocity', 'Next_arm_velocity', 'Next_bucket_velocity']].values

        return X, y, next_positions, next_velocities, df, sr, timestep
    
    def _calculate_sampling_rate(self, df, timestamp_col='timestamp'):
        """计算采样率"""
        if timestamp_col in df.columns:
            timestamps = pd.to_numeric(df[timestamp_col])
            time_diffs = np.diff(timestamps)
            avg_interval = np.mean(time_diffs)
            return int(np.ceil(1.0 / avg_interval))
        else:
            # 如果没有时间戳列，使用默认值
            return 100  # 默认100Hz
    
    def _create_sequences_with_next_state(self, data, labels, next_positions, next_velocities, seq_length=100, unroll=3):
        """
        创建序列数据，返回 multi-step 的 next state（长度为 unroll）。
        返回：
            X: shape (N, seq_length, feat)
            y: 加速度 targets for step0 (原来的 y)，shape (N, 3)
            next_pos_seq: shape (N, unroll, 3)  # denorm positions
            next_vel_seq: shape (N, unroll, 3)  # denorm velocities
        说明：next_positions/next_velocities 参数在原始实现是每条样本的下一时刻（denorm）。
        这里我们使用原始 data 的时间序列来获取连续的 future steps。
        """
        X_list, y_list = [], []
        next_pos_seq_list, next_vel_seq_list = [], []

        N = len(data)
        max_start = N - seq_length - (unroll - 1)  # 必须能向前看 unroll-1 步

        for i in range(max_start):
            X_list.append(data[i:i+seq_length])
            # y 对应原来第 seq_length-1 时刻的加速度（labels 数组）
            y_list.append(labels[i+seq_length-1])

            # 收集未来 unroll 步的 pos/vel（denorm，直接从 next_positions/next_velocities 的时间索引）
            # 注意：next_positions 原来是 next of each row, i.e., at index j it is the next of row j.
            # We want: for starting index i, the target for step u is at index (i + seq_length - 1) + u
            pos_seq_u = []
            vel_seq_u = []
            base_idx = i + seq_length - 1
            for u in range(unroll):
                idx = base_idx + u
                pos_seq_u.append(next_positions[idx])   # shape (3,)
                vel_seq_u.append(next_velocities[idx])
            next_pos_seq_list.append(np.stack(pos_seq_u, axis=0))  # (unroll,3)
            next_vel_seq_list.append(np.stack(vel_seq_u, axis=0))

        return np.array(X_list), np.array(y_list), np.array(next_pos_seq_list), np.array(next_vel_seq_list)
    
    def _run_free_run_experiments_with_mae_rmse(self, num_simulations=85, steps=1000):
        """
        详细自由运行实验 - 使用MAE和RMSE评估
        
        返回:
            模拟结果
        """
        plt.figure(figsize=(30, num_simulations * 9))
        simulation_results = []
        pos_mae_list = []
        vel_mae_list = []
        pos_rmse_list = []
        vel_rmse_list = []
        
        # 使用指数加权平均计算平滑MAE
        smoothed_pos_mae_list = []
        smoothed_vel_mae_list = []
        smoothed_pos_rmse_list = []
        smoothed_vel_rmse_list = []
        
        for sim in range(num_simulations):
            # 随机选择起始点
            max_start = len(self.df) - steps - self.seq_length - 2
            start_idx = random.randint(0, max_start)
            
            # 运行自由模拟
            predicted, true, pwm = self._free_run_simulation(start_idx, steps)
            
            # 检查数据有效性
            if np.any(~np.isfinite(predicted)) or np.any(~np.isfinite(true)):
                print(f"警告: 在模拟 {sim} 中发现无效值，跳过MAE和RMSE计算")
                continue
            
            # 计算MAE
            boom_pos_mae, arm_pos_mae, bucket_pos_mae = self._calc_mae_by_joint(true[:, :3], predicted[:, :3])
            boom_vel_mae, arm_vel_mae, bucket_vel_mae = self._calc_mae_by_joint(true[:, 3:], predicted[:, 3:])
            
            # 计算RMSE
            boom_pos_rmse, arm_pos_rmse, bucket_pos_rmse = self._calc_rmse_by_joint(true[:, :3], predicted[:, :3])
            boom_vel_rmse, arm_vel_rmse, bucket_vel_rmse = self._calc_rmse_by_joint(true[:, 3:], predicted[:, 3:])
            
            # 总MAE和RMSE
            pos_mae = np.mean([boom_pos_mae, arm_pos_mae, bucket_pos_mae])
            vel_mae = np.mean([boom_vel_mae, arm_vel_mae, bucket_vel_mae])
            pos_rmse = np.mean([boom_pos_rmse, arm_pos_rmse, bucket_pos_rmse])
            vel_rmse = np.mean([boom_vel_rmse, arm_vel_rmse, bucket_vel_rmse])
            
            simulation_results.append((predicted, true, start_idx, pwm))
            pos_mae_list.append(pos_mae)
            vel_mae_list.append(vel_mae)
            pos_rmse_list.append(pos_rmse)
            vel_rmse_list.append(vel_rmse)
            
            # 计算平滑MAE和RMSE（指数加权平均）
            if sim == 0:
                smoothed_pos_mae = pos_mae
                smoothed_vel_mae = vel_mae
                smoothed_pos_rmse = pos_rmse
                smoothed_vel_rmse = vel_rmse
            else:
                alpha = 0.1  # 平滑系数
                smoothed_pos_mae = alpha * pos_mae + (1 - alpha) * smoothed_pos_mae_list[-1]
                smoothed_vel_mae = alpha * vel_mae + (1 - alpha) * smoothed_vel_mae_list[-1]
                smoothed_pos_rmse = alpha * pos_rmse + (1 - alpha) * smoothed_pos_rmse_list[-1]
                smoothed_vel_rmse = alpha * vel_rmse + (1 - alpha) * smoothed_vel_rmse_list[-1]
            
            smoothed_pos_mae_list.append(smoothed_pos_mae)
            smoothed_vel_mae_list.append(smoothed_vel_mae)
            smoothed_pos_rmse_list.append(smoothed_pos_rmse)
            smoothed_vel_rmse_list.append(smoothed_vel_rmse)
            
            # 打印详细评估结果，对比MAE和RMSE
            print(f"模拟 {sim+1} (起始索引: {start_idx}):")
            print(f"  位置MAE: 大臂={boom_pos_mae:.6f}, 中臂={arm_pos_mae:.6f}, 铲斗={bucket_pos_mae:.6f} rad")
            print(f"  位置RMSE: 大臂={boom_pos_rmse:.6f}, 中臂={arm_pos_rmse:.6f}, 铲斗={bucket_pos_rmse:.6f} rad")
            print(f"  速度MAE: 大臂={boom_vel_mae:.6f}, 中臂={arm_vel_mae:.6f}, 铲斗={bucket_vel_mae:.6f} rad/s")
            print(f"  速度RMSE: 大臂={boom_vel_rmse:.6f}, 中臂={arm_vel_rmse:.6f}, 铲斗={bucket_vel_rmse:.6f} rad/s")
            print(f"  总位置: MAE={pos_mae:.6f} rad, RMSE={pos_rmse:.6f} rad (RMSE/MAE: {pos_rmse/pos_mae:.2f})")
            print(f"  总速度: MAE={vel_mae:.6f} rad/s, RMSE={vel_rmse:.6f} rad/s (RMSE/MAE: {vel_rmse/vel_mae:.2f})")
            print(f"  平滑位置: MAE={smoothed_pos_mae:.6f} rad, RMSE={smoothed_pos_rmse:.6f} rad")
            print(f"  平滑速度: MAE={smoothed_vel_mae:.6f} rad/s, RMSE={smoothed_vel_rmse:.6f} rad/s\n")

            if (self.plot == 'True'):
            # 绘制详细结果，在标题中同时显示MAE和RMSE
                self._plot_simulation_results_with_mae_rmse(
                    sim, num_simulations, steps, true, predicted, pwm,
                    boom_pos_mae, arm_pos_mae, bucket_pos_mae,
                    boom_vel_mae, arm_vel_mae, bucket_vel_mae,
                    boom_pos_rmse, arm_pos_rmse, bucket_pos_rmse,
                    boom_vel_rmse, arm_vel_rmse, bucket_vel_rmse,
                    start_idx
                )
        if (self.plot == 'True'): 
            plt.tight_layout()
            plt.savefig('free_run_simulations_with_mae_rmse.png')
            # plt.show()
        
        # 输出统计信息
        pos_mae_list = np.array(pos_mae_list)
        vel_mae_list = np.array(vel_mae_list)
        pos_rmse_list = np.array(pos_rmse_list)
        vel_rmse_list = np.array(vel_rmse_list)
        
        print("\n" + "="*80)
        print("自由运行模拟统计信息 (MAE vs RMSE 对比):")
        print("="*80)
        print("位置误差统计:")
        print(f"  MAE - 最小值: {np.min(pos_mae_list):.6f} rad, 最大值: {np.max(pos_mae_list):.6f} rad")
        print(f"  MAE - 平均值: {np.mean(pos_mae_list):.6f} rad, 中位数: {np.median(pos_mae_list):.6f} rad")
        print(f"  MAE - 标准差: {np.std(pos_mae_list):.6f} rad")
        print(f"  RMSE - 最小值: {np.min(pos_rmse_list):.6f} rad, 最大值: {np.max(pos_rmse_list):.6f} rad")
        print(f"  RMSE - 平均值: {np.mean(pos_rmse_list):.6f} rad, 中位数: {np.median(pos_rmse_list):.6f} rad")
        print(f"  RMSE - 标准差: {np.std(pos_rmse_list):.6f} rad")
        print(f"  RMSE/MAE 比率 - 平均值: {np.mean(pos_rmse_list/pos_mae_list):.2f}, 范围: [{np.min(pos_rmse_list/pos_mae_list):.2f}, {np.max(pos_rmse_list/pos_mae_list):.2f}]")
        
        print("\n速度误差统计:")
        print(f"  MAE - 最小值: {np.min(vel_mae_list):.6f} rad/s, 最大值: {np.max(vel_mae_list):.6f} rad/s")
        print(f"  MAE - 平均值: {np.mean(vel_mae_list):.6f} rad/s, 中位数: {np.median(vel_mae_list):.6f} rad/s")
        print(f"  MAE - 标准差: {np.std(vel_mae_list):.6f} rad/s")
        print(f"  RMSE - 最小值: {np.min(vel_rmse_list):.6f} rad/s, 最大值: {np.max(vel_rmse_list):.6f} rad/s")
        print(f"  RMSE - 平均值: {np.mean(vel_rmse_list):.6f} rad/s, 中位数: {np.median(vel_rmse_list):.6f} rad/s")
        print(f"  RMSE - 标准差: {np.std(vel_rmse_list):.6f} rad/s")
        print(f"  RMSE/MAE 比率 - 平均值: {np.mean(vel_rmse_list/vel_mae_list):.2f}, 范围: [{np.min(vel_rmse_list/vel_mae_list):.2f}, {np.max(vel_rmse_list/vel_mae_list):.2f}]")
        
        print(f"\n平滑误差统计(模拟次数{len(smoothed_pos_mae_list)}):")
        print(f"  平滑位置MAE - 最终值: {smoothed_pos_mae_list[-1]:.6f} rad")
        print(f"  平滑位置RMSE - 最终值: {smoothed_pos_rmse_list[-1]:.6f} rad")
        print(f"  平滑速度MAE - 最终值: {smoothed_vel_mae_list[-1]:.6f} rad/s")
        print(f"  平滑速度RMSE - 最终值: {smoothed_vel_rmse_list[-1]:.6f} rad/s")
        print("="*80)
        
        # 绘制MAE和RMSE趋势对比图
        # self._plot_mae_rmse_trends(
        #     pos_mae_list, vel_mae_list, pos_rmse_list, vel_rmse_list,
        #     smoothed_pos_mae_list, smoothed_vel_mae_list,
        #     smoothed_pos_rmse_list, smoothed_vel_rmse_list
        # )
        
        return simulation_results
    
    def _free_run_simulation(self, start_idx, steps=1000):
        """自由运行模拟"""
        # 准备存储数组
        predicted_traj = np.zeros((steps, 6))  # 位置和速度
        true_traj = np.zeros((steps, 6))  # 位置和速度
        pwm_traj = np.zeros((steps, 3))  # PWM输入信号
        
        # 获取起始状态
        if start_idx + self.seq_length >= len(self.X_seq):
            start_idx = len(self.X_seq) - self.seq_length - 1
        
        deno_input = self.x_scaler.inverse_transform(self.X_seq[start_idx])
        
        current_position = deno_input[:, :3]     # 形状是（100,3）
        current_velocity = deno_input[:, 3:6]         
        current_torque = deno_input[:, 6:]
        
        for i in range(steps):
            # 准备当前输入: [位置, 速度, 扭矩]
            current_input = np.column_stack((current_position, current_velocity, current_torque))
            
            # 归一化当前输入
            normalized_input = self.x_scaler.transform(current_input)
            tensor_input = torch.tensor(normalized_input, dtype=torch.float32).to(self.device)
            tensor_input = tensor_input.reshape(1, -1)  # 重塑为(1, seq_length*features)
            
            with torch.no_grad():
                # 预测下一时刻的加速度
                normalized_prediction = self.predict(tensor_input, denormalize=False)
                
            # 反归一化预测值（加速度）
            denormalized_accel = self.y_scaler.inverse_transform(normalized_prediction.reshape(1, -1))[0]
            
            # 使用加速度积分更新速度
            next_velocity_pred = current_velocity[-1] + denormalized_accel * self.timestep
            
            # 使用速度积分更新位置（梯形积分法）
            next_position_pred = current_position[-1] + (current_velocity[-1] + next_velocity_pred) / 2 * self.timestep
            
            # 存储预测结果
            predicted_traj[i, :3] = next_position_pred
            predicted_traj[i, 3:] = next_velocity_pred
            
            # 获取并存储真实值（下一时刻）
            true_idx = start_idx + i
            if true_idx < len(self.X_seq):
                true_position = self.x_scaler.inverse_transform(self.X_seq[true_idx])[-1, :3]
                true_velocity = self.x_scaler.inverse_transform(self.X_seq[true_idx])[-1, 3:6]
                true_traj[i, :3] = true_position
                true_traj[i, 3:] = true_velocity
            
            # 存储当前PWM输入
            pwm_traj[i] = current_torque[-1]
            
            # 更新状态变量
            current_position = np.roll(current_position, -1, axis=0)
            current_position[-1] = next_position_pred
            current_velocity = np.roll(current_velocity, -1, axis=0)
            current_velocity[-1] = next_velocity_pred
            
            # 获取下一个扭矩（使用真实扭矩）
            next_torque_idx = true_idx + 1
            current_torque = np.roll(current_torque, -1, axis=0)
            if next_torque_idx < len(self.X_seq):
                current_torque[-1] = self.x_scaler.inverse_transform(self.X_seq[next_torque_idx])[-1, 6:9]
            else:
                current_torque[-1] = self.x_scaler.inverse_transform(self.X_seq[-1])[-1, 6:9]
        
        return predicted_traj, true_traj, pwm_traj
    
    def _plot_simulation_results_with_mae_rmse(self, sim, num_simulations, steps, true, predicted, pwm,
                                              boom_pos_mae, arm_pos_mae, bucket_pos_mae,
                                              boom_vel_mae, arm_vel_mae, bucket_vel_mae,
                                              boom_pos_rmse, arm_pos_rmse, bucket_pos_rmse,
                                              boom_vel_rmse, arm_vel_rmse, bucket_vel_rmse,
                                              start_idx):
        """绘制单个模拟结果 - 同时显示MAE和RMSE指标"""
        # 位置结果 - 大臂
        plt.subplot(num_simulations, 9, 9*sim + 1)
        plt.plot(range(steps), true[:, 0], 'b-', label='Real Value', linewidth=2)
        plt.plot(range(steps), predicted[:, 0], 'r--', label='Predicted Value', linewidth=2, alpha=0.8)
        # 使用MAE作为填充区域
        plt.fill_between(range(steps), 
                         true[:, 0] - boom_pos_mae, 
                         true[:, 0] + boom_pos_mae, 
                         color='gray', alpha=0.2, label=f'±MAE ({boom_pos_mae:.4f})')
        # 在标题中同时显示MAE和RMSE
        plt.title(f'sim{sim+1}: Boom Position\nMAE={boom_pos_mae:.4f}, RMSE={boom_pos_rmse:.4f}')
        plt.ylabel('Position (rad)')
        plt.legend()
        plt.grid(True)
        
        # 位置结果 - 中臂
        plt.subplot(num_simulations, 9, 9*sim + 2)
        plt.plot(range(steps), true[:, 1], 'b-', label='Real Value', linewidth=2)
        plt.plot(range(steps), predicted[:, 1], 'r--', label='Predicted Value', linewidth=2, alpha=0.8)
        plt.fill_between(range(steps), 
                         true[:, 1] - arm_pos_mae, 
                         true[:, 1] + arm_pos_mae, 
                         color='gray', alpha=0.2, label=f'±MAE ({arm_pos_mae:.4f})')
        plt.title(f'sim{sim+1}: Arm Position\nMAE={arm_pos_mae:.4f}, RMSE={arm_pos_rmse:.4f}')
        plt.ylabel('Position (rad)')
        plt.legend()
        plt.grid(True)

        # 位置结果 - 铲斗
        plt.subplot(num_simulations, 9, 9*sim + 3)
        plt.plot(range(steps), true[:, 2], 'b-', label='Real Value', linewidth=2)
        plt.plot(range(steps), predicted[:, 2], 'r--', label='Predicted Value', linewidth=2, alpha=0.8)
        plt.fill_between(range(steps), 
                         true[:, 2] - bucket_pos_mae, 
                         true[:, 2] + bucket_pos_mae, 
                         color='gray', alpha=0.2, label=f'±MAE ({bucket_pos_mae:.4f})')
        plt.title(f'sim{sim+1}: Bucket Position\nMAE={bucket_pos_mae:.4f}, RMSE={bucket_pos_rmse:.4f}')
        plt.ylabel('Position (rad)')
        plt.legend()
        plt.grid(True)

        # 速度结果 - 大臂
        plt.subplot(num_simulations, 9, 9*sim + 4)
        plt.plot(range(steps), true[:, 3], 'b-', label='Real Value', linewidth=2)
        plt.plot(range(steps), predicted[:, 3], 'r--', label='Predicted Value', linewidth=2, alpha=0.8)
        plt.fill_between(range(steps), 
                         true[:, 3] - boom_vel_mae, 
                         true[:, 3] + boom_vel_mae, 
                         color='gray', alpha=0.2, label=f'±MAE ({boom_vel_mae:.4f})')
        plt.title(f'sim{sim+1}: Boom Velocity\nMAE={boom_vel_mae:.4f}, RMSE={boom_vel_rmse:.4f}')
        plt.ylabel('Velocity (rad/s)')
        plt.legend()
        plt.grid(True)
    
        # 速度结果 - 中臂
        plt.subplot(num_simulations, 9, 9*sim + 5)
        plt.plot(range(steps), true[:, 4], 'b-', label='Real Value', linewidth=2)
        plt.plot(range(steps), predicted[:, 4], 'r--', label='Predicted Value', linewidth=2, alpha=0.8)
        plt.fill_between(range(steps), 
                         true[:, 4] - arm_vel_mae, 
                         true[:, 4] + arm_vel_mae, 
                         color='gray', alpha=0.2, label=f'±MAE ({arm_vel_mae:.4f})')
        plt.title(f'sim{sim+1}: Arm Velocity\nMAE={arm_vel_mae:.4f}, RMSE={arm_vel_rmse:.4f}')
        plt.ylabel('Velocity (rad/s)')
        plt.legend()
        plt.grid(True)

        # 速度结果 - 铲斗
        plt.subplot(num_simulations, 9, 9*sim + 6)
        plt.plot(range(steps), true[:, 5], 'b-', label='Real Value', linewidth=2)
        plt.plot(range(steps), predicted[:, 5], 'r--', label='Predicted Value', linewidth=2, alpha=0.8)
        plt.fill_between(range(steps), 
                         true[:, 5] - bucket_vel_mae, 
                         true[:, 5] + bucket_vel_mae, 
                         color='gray', alpha=0.2, label=f'±MAE ({bucket_vel_mae:.4f})')
        plt.title(f'sim{sim+1}: Bucket Velocity\nMAE={bucket_vel_mae:.4f}, RMSE={bucket_vel_rmse:.4f}')
        plt.ylabel('Velocity (rad/s)')
        plt.legend()
        plt.grid(True)

        # PWM输入信号 - 大臂
        plt.subplot(num_simulations, 9, 9*sim + 7)
        plt.plot(range(steps), pwm[:, 0], 'g-', label='PWM Input', linewidth=2)
        plt.title(f'sim{sim+1}: Boom PWM Input')
        plt.ylabel('PWM Value')
        plt.legend()
        plt.grid(True)

        # PWM输入信号 - 中臂
        plt.subplot(num_simulations, 9, 9*sim + 8)
        plt.plot(range(steps), pwm[:, 1], 'g-', label='PWM Input', linewidth=2)
        plt.title(f'sim{sim+1}: Arm PWM Input')
        plt.ylabel('PWM Value')
        plt.legend()
        plt.grid(True)

        # PWM输入信号 - 铲斗
        plt.subplot(num_simulations, 9, 9*sim + 9)
        plt.plot(range(steps), pwm[:, 2], 'g-', label='PWM Input', linewidth=2)
        plt.title(f'sim{sim+1}: Bucket PWM Input')
        plt.ylabel('PWM Value')
        plt.xlabel('Time Step')
        plt.legend()
        plt.grid(True)
    
    def _plot_mae_rmse_trends(self, pos_mae_list, vel_mae_list, pos_rmse_list, vel_rmse_list,
                              smoothed_pos_mae_list, smoothed_vel_mae_list,
                              smoothed_pos_rmse_list, smoothed_vel_rmse_list):
        """绘制MAE和RMSE趋势对比图"""
        plt.figure(figsize=(20, 12))
        
        # 1. 位置MAE和RMSE趋势
        plt.subplot(2, 3, 1)
        plt.plot(pos_mae_list, 'b-', label='Position MAE', alpha=0.6, linewidth=1.5)
        plt.plot(pos_rmse_list, 'r-', label='Position RMSE', alpha=0.6, linewidth=1.5)
        plt.plot(smoothed_pos_mae_list, 'b--', label='Smoothed Position MAE', linewidth=2)
        plt.plot(smoothed_pos_rmse_list, 'r--', label='Smoothed Position RMSE', linewidth=2)
        plt.xlabel('Simulation Count')
        plt.ylabel('Error (rad)')
        plt.title('Position Error Trends (MAE vs RMSE)')
        plt.legend()
        plt.grid(True)
        
        # 2. 速度MAE和RMSE趋势
        plt.subplot(2, 3, 2)
        plt.plot(vel_mae_list, 'b-', label='Velocity MAE', alpha=0.6, linewidth=1.5)
        plt.plot(vel_rmse_list, 'r-', label='Velocity RMSE', alpha=0.6, linewidth=1.5)
        plt.plot(smoothed_vel_mae_list, 'b--', label='Smoothed Velocity MAE', linewidth=2)
        plt.plot(smoothed_vel_rmse_list, 'r--', label='Smoothed Velocity RMSE', linewidth=2)
        plt.xlabel('Simulation Count')
        plt.ylabel('Error (rad/s)')
        plt.title('Velocity Error Trends (MAE vs RMSE)')
        plt.legend()
        plt.grid(True)
        
        # 3. 位置误差分布对比
        plt.subplot(2, 3, 3)
        bins = 20
        plt.hist(pos_mae_list, bins=bins, alpha=0.5, label='Position MAE', color='blue', edgecolor='black')
        plt.hist(pos_rmse_list, bins=bins, alpha=0.5, label='Position RMSE', color='red', edgecolor='black')
        plt.xlabel('Error (rad)')
        plt.ylabel('Frequency')
        plt.title('Position Error Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 速度误差分布对比
        plt.subplot(2, 3, 4)
        plt.hist(vel_mae_list, bins=bins, alpha=0.5, label='Velocity MAE', color='blue', edgecolor='black')
        plt.hist(vel_rmse_list, bins=bins, alpha=0.5, label='Velocity RMSE', color='red', edgecolor='black')
        plt.xlabel('Error (rad/s)')
        plt.ylabel('Frequency')
        plt.title('Velocity Error Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. RMSE/MAE比率分布
        plt.subplot(2, 3, 5)
        rmse_mae_ratio_pos = pos_rmse_list / pos_mae_list
        rmse_mae_ratio_vel = vel_rmse_list / vel_mae_list
        plt.hist(rmse_mae_ratio_pos, bins=15, alpha=0.5, label='Position RMSE/MAE', color='blue', edgecolor='black')
        plt.hist(rmse_mae_ratio_vel, bins=15, alpha=0.5, label='Velocity RMSE/MAE', color='red', edgecolor='black')
        plt.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Ratio=1.0')
        plt.xlabel('RMSE/MAE Ratio')
        plt.ylabel('Frequency')
        plt.title('RMSE/MAE Ratio Distribution (Ratio>1 indicates outliers)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. 误差统计摘要
        plt.subplot(2, 3, 6)
        # 清空子图，用于文本显示
        plt.axis('off')
        stats_text = f"""
        Error Statistics Summary:
        
        Position Error:
        MAE - Mean: {np.mean(pos_mae_list):.6f} rad
        MAE - Standard Deviation: {np.std(pos_mae_list):.6f} rad
        RMSE - Mean: {np.mean(pos_rmse_list):.6f} rad
        RMSE - Standard Deviation: {np.std(pos_rmse_list):.6f} rad
        RMSE/MAE - Mean: {np.mean(rmse_mae_ratio_pos):.2f}
        
        Velocity Error:
        MAE - Mean: {np.mean(vel_mae_list):.6f} rad/s
        MAE - Standard Deviation: {np.std(vel_mae_list):.6f} rad/s
        RMSE - Mean: {np.mean(vel_rmse_list):.6f} rad/s
        RMSE - Standard Deviation: {np.std(vel_rmse_list):.6f} rad/s
        RMSE/MAE - Mean: {np.mean(rmse_mae_ratio_vel):.2f}
        
        Notes:
        • RMSE is always greater than or equal to MAE
        • RMSE/MAE ratio reflects the dispersion of error distribution
        • Ratio close to 1 indicates concentrated error distribution
        • Ratio much larger than 1 indicates large outliers
        """
        plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('mae_rmse_trends_comparison.png', dpi=300)
        # plt.show()
    
    def _plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(15, 10))
        
        # 1. 损失曲线
        plt.subplot(3, 3, 1)
        plt.plot(self.train_history['train_loss'], label='Train Loss')
        plt.plot(self.train_history['val_loss'], label='Validation Loss')
        plt.xlabel('Training Epochs')
        plt.ylabel('MAE Loss')
        plt.title('Training and Validation Loss (MAE)')
        plt.legend()
        plt.grid(True)
        
        # 2. 大臂位置MAE
        plt.subplot(3, 3, 2)
        plt.plot(self.train_history['boom_pos_mae'], color='blue')
        plt.xlabel('Training Epochs')
        plt.ylabel('MAE (rad)')
        plt.title('Boom Position MAE')
        plt.grid(True)
        
        # 3. 中臂位置MAE
        plt.subplot(3, 3, 3)
        plt.plot(self.train_history['arm_pos_mae'], color='red')
        plt.xlabel('Training Epochs')
        plt.ylabel('MAE (rad)')
        plt.title('Arm Position MAE')
        plt.grid(True)
        
        # 4. 铲斗位置MAE
        plt.subplot(3, 3, 4)
        plt.plot(self.train_history['bucket_pos_mae'], color='green')
        plt.xlabel('Training Epochs')
        plt.ylabel('MAE (rad)')
        plt.title('Bucket Position MAE')
        plt.grid(True)
        
        # 5. 大臂速度MAE
        plt.subplot(3, 3, 5)
        plt.plot(self.train_history['boom_vel_mae'], color='blue', linestyle='--')
        plt.xlabel('Training Epochs')
        plt.ylabel('MAE (rad/s)')
        plt.title('Boom Velocity MAE')
        plt.grid(True)
        
        # 6. 中臂速度MAE
        plt.subplot(3, 3, 6)
        plt.plot(self.train_history['arm_vel_mae'], color='red', linestyle='--')
        plt.xlabel('Training Epochs')
        plt.ylabel('MAE (rad/s)')
        plt.title('Arm Velocity MAE')
        plt.grid(True)
        
        # 7. 铲斗速度MAE
        plt.subplot(3, 3, 7)
        plt.plot(self.train_history['bucket_vel_mae'], color='green', linestyle='--')
        plt.xlabel('Training Epochs')
        plt.ylabel('MAE (rad/s)')
        plt.title('Bucket Velocity MAE')
        plt.grid(True)
        
        # 8. 综合位置MAE
        plt.subplot(3, 3, 8)
        plt.plot(self.train_history['boom_pos_mae'], label='Boom', alpha=0.7)
        plt.plot(self.train_history['arm_pos_mae'], label='Arm', alpha=0.7)
        plt.plot(self.train_history['bucket_pos_mae'], label='Bucket', alpha=0.7)
        plt.xlabel('Training Epochs')
        plt.ylabel('MAE (rad)')
        plt.title('Position MAE Comparison')
        plt.legend()
        plt.grid(True)
        
        # 9. 综合速度MAE
        plt.subplot(3, 3, 9)
        plt.plot(self.train_history['boom_vel_mae'], label='Boom', alpha=0.7, linestyle='--')
        plt.plot(self.train_history['arm_vel_mae'], label='Arm', alpha=0.7, linestyle='--')
        plt.plot(self.train_history['bucket_vel_mae'], label='Bucket', alpha=0.7, linestyle='--')
        plt.xlabel('Training Epochs')
        plt.ylabel('MAE (rad/s)')
        plt.title('Velocity MAE Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_mae.png')
        # plt.show()
    
    def _calc_mae_by_joint(self, y_true, y_pred):
        """计算每个关节的MAE"""
        mae_list = []
        for i in range(y_true.shape[1]):
            mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
            mae_list.append(mae)
        return mae_list[0], mae_list[1], mae_list[2]
    
    def _calc_rmse_by_joint(self, y_true, y_pred):
        """计算每个关节的RMSE"""
        rmse_list = []
        for i in range(y_true.shape[1]):
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            rmse_list.append(rmse)
        return rmse_list[0], rmse_list[1], rmse_list[2]
    
    def _calc_rmse(self, y_true, y_pred):
        """计算RMSE"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def _replace_over_threshold(self, numbers, threshold=10, fill_value=-1e-4):
        """将超过阈值的数替换为指定负数"""
        replaced_items = []
        result = numbers.copy()
        
        for index, value in enumerate(numbers):
            if value > threshold:
                result[index] = fill_value
                replaced_items.append((index, value))
        
        if replaced_items:
            print("被替换的元素（索引, 原数值）:")
            for index, value in replaced_items:
                print(f"  索引 {index}: {value} -> {fill_value}")
            print(f"共替换了 {len(replaced_items)} 个元素")
        else:
            print("没有元素被替换")
        
        return result
    
    
    


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='机械臂轨迹预测训练脚本 - MAE版本')
    parser.add_argument('--csv', type=str, default="data/data_med.csv", help='要训练的CSV数据文件路径')
    parser.add_argument('--data_dir', type=str, default=None, help='包含多个CSV数据文件的目录路径')
    parser.add_argument('--train', type=str, default="True", help='是否训练数据 (default: True)')
    parser.add_argument('--free', type=str, default="True", help='是否使用自由运行测试数据 (default: True)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--model_path', type=str, default="best_model_autoreg_20.pth", 
                        help='预训练模型文件路径 (default: best_model_autoreg_20.pth)')
    parser.add_argument('--scaler_path', type=str, default="scalers_mae.pkl", 
                        help='归一化器文件路径 (default: scalers_mae.pkl)')
    parser.add_argument('--nsims', type=int, default=85, help='自由运行模拟次数 (default: 85)')
    parser.add_argument('--plot', type=str, default="False")
    
    args = parser.parse_args()
    
    # ========== 关键1：先创建类的实例（必须！） ==========
    model = ExcavatorAccelerationPredictor()
    model.plot = args.plot
    
    # 加载数据（自由运行需要数据来获取PWM/初始状态，必须先加载）
    if args.data_dir:
        model.load_data(data_dir=args.data_dir)
    else:
        model.load_data(csv_file=args.csv)
    
    # 训练模型（可选）
    if args.train == "True":
        model.train(epochs=args.epochs)
        # 保存模型（使用指定路径）
        model.save_model(model_path=args.model_path, scaler_path=args.scaler_path)
        # print(f"模型已保存至: {args.model_path}")
        # print(f"归一化器已保存至: {args.scaler_path}")
        print("训练完成！")
    
    # 自由运行测试（可选）
    if args.free == "True":
        # ========== 关键2：当不训练时，通过实例调用load_model ==========
        if args.train == "False":
            try:
                # 正确方式：通过实例model调用load_model，而非类名
                model.load_model(model_path=args.model_path, scaler_path=args.scaler_path)
                print(f"\033[32m成功加载预训练模型：{args.model_path}\033[0m")
                print(f"\033[32m成功加载归一化器：{args.scaler_path}\033[0m")
            except FileNotFoundError as e:
                # 文件不存在时的友好提示
                print(f"\033[31m错误：找不到文件 {e.filename}！\033[0m")
                print("请先执行训练（--train True）生成模型文件，或检查路径是否正确。")
                sys.exit(1)
            except Exception as e:
                # 其他加载错误（如文件损坏、版本不匹配）
                print(f"\033[31m加载模型失败：{str(e)}\033[0m")
                sys.exit(1)
        
        # 运行自由运行模拟（此时model是加载了预训练参数的实例，而非随机初始化）
        simulation_results = model.run_free_run_experiments(num_simulations=args.nsims, steps=1000)
        print("\033[32m自由运行模拟完成！\033[0m")


if __name__ == '__main__':
    main()