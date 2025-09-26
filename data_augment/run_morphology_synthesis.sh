#!/bin/bash
# Morphology Synthesis Runner
# Runs DOF and Link scaling synthesis separately for 46 valid episodes

cd /home/cx/AET_FOR_RL/vla/data_augment

echo "🎯 Starting Morphology Synthesis for Valid Episodes"
echo "============================================================"

# Load conda environment
conda activate AET_FOR_RL

# Run DOF modification synthesis in existing smart session
echo "🔧 Starting DOF Modification Synthesis..."
tmux send-keys -t smart_dof_synthesis "cd /home/cx/AET_FOR_RL/vla/data_augment/synthesis_runners" Enter
tmux send-keys -t smart_dof_synthesis "python run_dof_modification_synthesis.py" Enter

echo "   ✅ DOF synthesis started in tmux session: smart_dof_synthesis"

# Run Link scaling synthesis in existing smart session  
echo "🔗 Starting Link Scaling Synthesis..."
tmux send-keys -t smart_link_synthesis "cd /home/cx/AET_FOR_RL/vla/data_augment/synthesis_runners" Enter
tmux send-keys -t smart_link_synthesis "python run_link_scaling_synthesis.py" Enter

echo "   ✅ Link synthesis started in tmux session: smart_link_synthesis"

echo ""
echo "📊 Expected Output:"
echo "   🔧 DOF variations: 460 (46 episodes × 10 variations)"
echo "   🔗 Link variations: 460 (46 episodes × 10 variations)" 
echo "   📈 Total: 920 morphology variations"
echo ""
echo "📱 Monitor sessions:"
echo "   tmux attach -t smart_dof_synthesis"
echo "   tmux attach -t smart_link_synthesis"
echo ""
echo "🎉 Both synthesis processes started in separate tmux sessions!"