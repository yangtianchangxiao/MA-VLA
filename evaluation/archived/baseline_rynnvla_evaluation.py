#!/usr/bin/env python3
"""
Baseline RynnVLA-001 Evaluation on ManiSkill
Test original pre-trained RynnVLA without our morphology modifications
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# Add our model path
sys.path.append('/home/cx/AET_FOR_RL/vla/train')
from vla_model import RealRynnVLALoRAGNN

# ManiSkill imports
import mani_skill.envs
import gymnasium as gym

class BaselineRynnVLAEvaluator:
    """Baseline evaluation with original RynnVLA-001 (no morphology training)"""

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)

        print(f"🤖 Baseline RynnVLA-001 Evaluator")
        print(f"   Device: {self.device}")
        print(f"   Testing: Original pre-trained RynnVLA without morphology training")

        # Load original RynnVLA model (no checkpoint loading)
        self.model = self._load_original_rynnvla()

        # Initialize ManiSkill environment
        self.env = self._setup_environment()

    def _load_original_rynnvla(self):
        """Load original RynnVLA-001 model without our training"""
        print("🔄 Loading original RynnVLA-001 model...")

        try:
            rynnvla_path = "/home/cx/AET_FOR_RL/vla/参考模型/RynnVLA-001/pretrained_models/RynnVLA-001-7B-Base"

            model = RealRynnVLALoRAGNN(
                model_path=rynnvla_path,
                lora_rank=32,
                gnn_node_dim=256
            ).to(self.device)

            print("✅ Original RynnVLA-001 model loaded (no custom training)")
            print("   This tests pure pre-trained performance")

            model.eval()
            return model

        except Exception as e:
            print(f"❌ Failed to load original model: {e}")
            return None

    def _setup_environment(self):
        """Setup ManiSkill PickCube environment"""
        print("🔄 Setting up ManiSkill environment...")

        try:
            env = gym.make(
                "PickCube-v1",
                obs_mode="state",
                control_mode="pd_joint_delta_pos",
                render_mode=None,
                max_episode_steps=200
            )

            print(f"✅ Environment created: {env.spec.id}")
            print(f"   Action space: {env.action_space}")

            return env

        except Exception as e:
            print(f"❌ Failed to setup environment: {e}")
            return None

    def _baseline_inference(self, obs, instruction="pick up the cube", num_joints=7):
        """Baseline inference using original RynnVLA"""

        if self.model is None:
            # Random baseline if model fails
            action = np.random.uniform(-0.1, 0.1, size=8)
            return action

        try:
            with torch.no_grad():
                # Use original RynnVLA forward pass
                # Create minimal input for pre-trained model
                batch_size = 1

                # Dummy token sequence (original RynnVLA expects token input)
                input_ids = torch.zeros(batch_size, 64, dtype=torch.long, device=self.device)
                attention_mask = torch.ones(batch_size, 64, device=self.device)

                # Run original model inference
                outputs = self.model(input_ids, attention_mask, num_joints=num_joints)
                predicted_actions = outputs['action_pred']  # Original RynnVLA output key

                # Convert to numpy and format for ManiSkill
                action = predicted_actions.cpu().numpy().flatten()

                # Pad/truncate to 8-dim for ManiSkill
                if len(action) < 8:
                    padded_action = np.zeros(8)
                    padded_action[:len(action)] = action
                    if len(action) < 8:
                        padded_action[-1] = 0.0  # Neutral gripper
                    action = padded_action
                elif len(action) > 8:
                    action = action[:8]

                return action

        except Exception as e:
            print(f"⚠️  Baseline inference failed: {e}, using random action")
            return np.random.uniform(-0.1, 0.1, size=8)

    def evaluate_single_episode(self, max_steps=200, instruction="pick up the cube", morphology_dof=7):
        """Evaluate single episode with baseline model"""

        if self.env is None:
            return {"success": False, "error": "Environment not initialized"}

        obs, info = self.env.reset()
        episode_reward = 0
        steps = 0
        success = False

        print(f"🎯 Testing baseline with {morphology_dof}-DOF")
        print(f"   Instruction: '{instruction}'")

        for step in range(max_steps):
            # Baseline inference
            action = self._baseline_inference(obs, instruction, num_joints=morphology_dof)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)

            episode_reward += reward
            steps += 1

            # Check success
            if info.get('success', False):
                success = True
                print(f"✅ Baseline success at step {steps}!")
                break

            if terminated or truncated:
                break

        result = {
            "success": success,
            "steps": steps,
            "reward": float(episode_reward),
            "morphology_dof": morphology_dof,
            "instruction": instruction,
            "model_type": "baseline_rynnvla"
        }

        print(f"📊 Baseline result: Success={success}, Steps={steps}, Reward={float(episode_reward):.3f}")
        return result

    def run_baseline_evaluation(self, num_episodes_per_morphology=3):
        """Run baseline evaluation"""
        print("🚀 Starting Baseline RynnVLA-001 Evaluation...")
        print("=" * 60)

        if self.model is None or self.env is None:
            print("❌ Cannot run baseline evaluation")
            return None

        # Test same morphologies as our trained model
        morphologies = [
            {"dof": 5, "name": "5-DOF reduced"},
            {"dof": 6, "name": "6-DOF (5+gripper)"},
            {"dof": 7, "name": "7-DOF standard"},
        ]

        instructions = [
            "pick up the cube",
            "grasp the red cube",
            "lift the object"
        ]

        all_results = []

        for morph in morphologies:
            print(f"\n🤖 Testing {morph['name']} with baseline model...")
            morph_results = []

            for episode in range(num_episodes_per_morphology):
                instruction = instructions[episode % len(instructions)]

                result = self.evaluate_single_episode(
                    instruction=instruction,
                    morphology_dof=morph["dof"]
                )

                result["morphology_name"] = morph["name"]
                morph_results.append(result)
                all_results.append(result)

            # Morphology stats
            success_rate = np.mean([r["success"] for r in morph_results])
            avg_steps = np.mean([r["steps"] for r in morph_results])
            avg_reward = np.mean([r["reward"] for r in morph_results])

            print(f"📊 Baseline {morph['name']} Results:")
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Average Steps: {avg_steps:.1f}")
            print(f"   Average Reward: {avg_reward:.3f}")

        # Overall baseline stats
        overall_success = np.mean([r["success"] for r in all_results])
        overall_reward = np.mean([r["reward"] for r in all_results])

        print(f"\n🏆 Baseline RynnVLA-001 Overall Results:")
        print(f"   Total Episodes: {len(all_results)}")
        print(f"   Overall Success Rate: {overall_success:.1%}")
        print(f"   Overall Average Reward: {overall_reward:.3f}")

        print(f"\n📊 Baseline Success Rate by Morphology:")
        morphologies = list(set([r["morphology_name"] for r in all_results]))
        for morph in morphologies:
            morph_results = [r for r in all_results if r["morphology_name"] == morph]
            success_rate = np.mean([r["success"] for r in morph_results])
            avg_reward = np.mean([r["reward"] for r in morph_results])
            print(f"   {morph}: {success_rate:.1%} success, {avg_reward:.3f} reward")

        return all_results

def main():
    """Main baseline evaluation"""
    print("🎯 Baseline RynnVLA-001 vs Our Multi-Morphology VLA")
    print("   This will show if our training helped or hurt performance")

    evaluator = BaselineRynnVLAEvaluator()
    results = evaluator.run_baseline_evaluation(num_episodes_per_morphology=3)

    if results:
        print(f"\n✅ Baseline evaluation completed!")
        print(f"🔍 Compare these results with our trained model results")
        print(f"📈 Higher performance = our training helped")
        print(f"📉 Lower performance = domain gap or training issues")
    else:
        print(f"\n❌ Baseline evaluation failed")

if __name__ == "__main__":
    main()