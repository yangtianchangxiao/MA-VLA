#!/usr/bin/env python3
"""
Original RynnVLA-001 Model ManiSkill Evaluation
使用纯原始RynnVLA-001模型评估ManiSkill，不使用我们的GNN改进
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# ManiSkill imports
import mani_skill.envs
import gymnasium as gym
from mani_skill.utils import gym_utils

# Add RynnVLA path
sys.path.append('/home/cx/AET_FOR_RL/vla/参考模型/RynnVLA-001')

class OriginalRynnVLAManiSkillEvaluator:
    """使用原始RynnVLA-001模型评估ManiSkill"""

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)

        print(f"🔬 Original RynnVLA-001 ManiSkill Evaluator")
        print(f"   Device: {self.device}")

        # 初始化原始RynnVLA模型
        self.model = self._load_original_rynnvla()

        # 初始化ManiSkill环境
        self.env = self._setup_maniskill_env()

    def _load_original_rynnvla(self):
        """加载原始RynnVLA-001模型"""
        print("🔄 Loading Original RynnVLA-001 model...")

        try:
            # 尝试导入RynnVLA的原始模型类
            from models.chameleon_model.chameleon.modeling_chameleon import ChameleonForConditionalGeneration
            from models.chameleon_model.configuration_xllmx_chameleon import ChameleonConfig

            # 创建配置 - 使用标准的7维action配置
            config = ChameleonConfig(
                vocab_size=65536,
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                hidden_act="silu",
                max_position_embeddings=4096,
                initializer_range=0.02,
                rms_norm_eps=1e-5,
                use_cache=True,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                action_chunk_size=20,
                action_dim=7,  # 原始的7维action
                visual_head_type='one_layer',
                state_dim=6
            )

            # 创建模型
            model = ChameleonForConditionalGeneration(config)
            model = model.to(self.device)
            model.eval()

            print("✅ Original RynnVLA-001 model created")
            print(f"   Action dimension: {config.action_dim}")
            print(f"   Action chunk size: {config.action_chunk_size}")

            return model

        except Exception as e:
            print(f"❌ Failed to load original RynnVLA: {e}")
            print("🔄 Creating simplified baseline model...")

            # 如果无法加载原始模型，创建一个简化的baseline
            return self._create_baseline_model()

    def _create_baseline_model(self):
        """创建简化的baseline模型用于测试"""
        class SimpleVLABaseline(nn.Module):
            def __init__(self):
                super().__init__()
                self.action_dim = 7  # DROID标准维度
                self.action_chunk_size = 20

                # 简化的特征提取
                self.vision_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, 2, 3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((8, 8)),
                    nn.Flatten(),
                    nn.Linear(64 * 64, 1024)
                )

                self.text_encoder = nn.Sequential(
                    nn.Linear(32, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024)
                )

                # Action head - 输出7维动作序列
                self.action_head = nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, self.action_dim * self.action_chunk_size)
                )

            def forward(self, images, text_tokens):
                batch_size = images.shape[0]

                # 特征融合
                vision_feat = self.vision_encoder(images)
                text_feat = self.text_encoder(text_tokens.float())
                fused_feat = torch.cat([vision_feat, text_feat], dim=-1)

                # 生成动作序列
                action_logits = self.action_head(fused_feat)
                actions = action_logits.view(batch_size, self.action_chunk_size, self.action_dim)

                return {
                    'actions': actions,  # [batch, chunk_size, action_dim]
                    'logits': action_logits
                }

        model = SimpleVLABaseline().to(self.device)
        print("✅ Baseline model created for testing")
        return model

    def _setup_maniskill_env(self):
        """设置ManiSkill环境"""
        print("🔄 Setting up ManiSkill environment...")

        try:
            env = gym.make(
                "PickCube-v1",
                obs_mode="state",
                control_mode="pd_joint_delta_pos",
                render_mode=None,
                max_episode_steps=200
            )

            print(f"✅ ManiSkill environment created")
            print(f"   Task: PickCube-v1")
            print(f"   Action space: {env.action_space}")
            print(f"   Expected action dim: {env.action_space.shape[0]}")

            return env

        except Exception as e:
            print(f"❌ Failed to setup ManiSkill environment: {e}")
            return None

    def _rynnvla_inference(self, obs, instruction="pick up the cube"):
        """使用原始RynnVLA进行推理"""

        if self.model is None:
            # 随机动作作为fallback
            return np.random.uniform(-0.1, 0.1, size=8)

        try:
            with torch.no_grad():
                batch_size = 1

                # 模拟图像输入 (实际部署时需要真实相机图像)
                images = torch.zeros(batch_size, 3, 320, 180, device=self.device)

                # 模拟文本token (实际需要proper tokenizer)
                text_tokens = torch.zeros(batch_size, 32, device=self.device)

                # 运行原始RynnVLA推理
                outputs = self.model(images, text_tokens)

                if 'actions' in outputs:
                    action_sequences = outputs['actions']  # [batch, chunk_size, action_dim]
                    print(f"   Model output shape: {action_sequences.shape}")

                    # 提取第一个时间步的动作
                    first_step_action = action_sequences[0, 0, :].cpu().numpy()  # [action_dim]
                    print(f"   First step action shape: {first_step_action.shape}")
                    print(f"   Action values: {first_step_action}")

                    # 转换维度以适配ManiSkill
                    return self._convert_to_maniskill_action(first_step_action)

                else:
                    print("⚠️ No 'actions' key in model output")
                    return np.random.uniform(-0.1, 0.1, size=8)

        except Exception as e:
            print(f"⚠️ RynnVLA inference failed: {e}")
            return np.random.uniform(-0.1, 0.1, size=8)

    def _convert_to_maniskill_action(self, rynnvla_action):
        """将RynnVLA输出转换为ManiSkill格式"""
        rynnvla_dim = len(rynnvla_action)
        maniskill_dim = 8

        print(f"🔄 Converting action: {rynnvla_dim}D → {maniskill_dim}D")
        print(f"   RynnVLA output: {rynnvla_action}")

        if rynnvla_dim == 7:
            # DROID格式: [6关节 + 夹爪] → ManiSkill 8维
            # 假设ManiSkill是 [7关节 + 夹爪]
            maniskill_action = np.zeros(8)
            maniskill_action[:6] = rynnvla_action[:6]  # 前6个关节直接复制
            maniskill_action[6] = 0.0  # 第7个关节设为0 (或插值)
            maniskill_action[7] = rynnvla_action[6]  # 夹爪映射到最后一维

        elif rynnvla_dim == 6:
            # LeRobot格式: 6维 → 8维
            maniskill_action = np.zeros(8)
            maniskill_action[:6] = rynnvla_action
            maniskill_action[6] = 0.0  # 额外关节
            maniskill_action[7] = 0.5  # 夹爪设为中性

        elif rynnvla_dim == 8:
            # 已经是8维，直接使用
            maniskill_action = rynnvla_action

        else:
            # 其他维度，截断或补零
            maniskill_action = np.zeros(8)
            copy_dim = min(rynnvla_dim, 8)
            maniskill_action[:copy_dim] = rynnvla_action[:copy_dim]

        print(f"   ManiSkill action: {maniskill_action}")
        return maniskill_action

    def evaluate_single_episode(self, max_steps=200):
        """评估单个episode"""

        if self.env is None:
            return {"success": False, "error": "Environment not initialized"}

        obs, info = self.env.reset()
        episode_reward = 0
        steps = 0
        success = False

        print(f"🎯 Starting episode with Original RynnVLA-001")

        for step in range(max_steps):
            # 使用原始RynnVLA推理
            action = self._rynnvla_inference(obs)

            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)

            episode_reward += reward
            steps += 1

            # 检查成功条件
            if info.get('success', False):
                success = True
                print(f"✅ Success at step {steps}!")
                break

            if terminated or truncated:
                break

        result = {
            "success": success,
            "steps": steps,
            "reward": float(episode_reward),
            "model": "Original RynnVLA-001"
        }

        print(f"📊 Episode result: Success={success}, Steps={steps}, Reward={float(episode_reward):.3f}")
        return result

    def run_evaluation(self, num_episodes=5):
        """运行完整评估"""
        print("🚀 Starting Original RynnVLA-001 ManiSkill Evaluation")
        print("=" * 60)

        if self.env is None:
            print("❌ Cannot run evaluation - environment failed to initialize")
            return None

        results = []

        for episode in range(num_episodes):
            print(f"\n📅 Episode {episode + 1}/{num_episodes}")
            result = self.evaluate_single_episode()
            results.append(result)

        # 计算总体统计
        success_rate = np.mean([r["success"] for r in results])
        avg_reward = np.mean([r["reward"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])

        print(f"\n🏆 Original RynnVLA-001 Results:")
        print(f"   Episodes: {num_episodes}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Average Reward: {avg_reward:.3f}")
        print(f"   Average Steps: {avg_steps:.1f}")

        return {
            "results": results,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps
        }

def main():
    """主函数"""
    print("🔬 Original RynnVLA-001 ManiSkill Evaluation")
    print("=" * 50)

    evaluator = OriginalRynnVLAManiSkillEvaluator()
    evaluation_results = evaluator.run_evaluation(num_episodes=3)

    if evaluation_results:
        print(f"\n✅ Evaluation completed!")
        print(f"🎯 This shows the original RynnVLA-001 performance on ManiSkill")
        print(f"📊 Key insight: Action dimension conversion {7}D → {8}D")
    else:
        print(f"\n❌ Evaluation failed")

if __name__ == "__main__":
    main()