#!/usr/bin/env python3
"""
RynnVLA Model Downloader with Real-time Storage Monitoring
"""
import os
import time
import shutil
import subprocess
from huggingface_hub import snapshot_download

def check_disk_space():
    """Check available disk space in GB"""
    stat = shutil.disk_usage('/home')
    free_gb = stat.free / (1024**3)
    return free_gb

def monitor_download():
    """Download RynnVLA model with storage monitoring"""
    model_name = "Alibaba-DAMO-Academy/RynnVLA-001-7B-Base"
    download_dir = "./RynnVLA-001-7B-Base"
    
    print("🚀 Starting RynnVLA-001-7B-Base Download")
    print("=" * 60)
    
    initial_space = check_disk_space()
    print(f"📊 Initial free space: {initial_space:.1f} GB")
    print(f"📦 Expected model size: ~13.7 GB")
    print(f"✅ Expected remaining: ~{initial_space - 13.7:.1f} GB")
    print()
    
    if initial_space < 20:
        print("⚠️  WARNING: Low disk space! Proceeding carefully...")
    
    try:
        print("🔄 Starting download...")
        start_time = time.time()
        
        # Download with progress monitoring
        snapshot_download(
            repo_id=model_name,
            local_dir=download_dir,
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
        )
        
        end_time = time.time()
        final_space = check_disk_space()
        download_time = end_time - start_time
        space_used = initial_space - final_space
        
        print()
        print("✅ Download Complete!")
        print("=" * 60)
        print(f"⏱️  Download time: {download_time:.1f} seconds")
        print(f"💾 Space used: {space_used:.1f} GB")
        print(f"📊 Remaining space: {final_space:.1f} GB")
        
        # Check actual model size
        if os.path.exists(download_dir):
            result = subprocess.run(['du', '-sh', download_dir], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                actual_size = result.stdout.split()[0]
                print(f"📁 Actual model size: {actual_size}")
        
        print()
        print("🎯 Model ready for use!")
        return True
        
    except Exception as e:
        current_space = check_disk_space()
        print(f"❌ Download failed: {e}")
        print(f"📊 Current free space: {current_space:.1f} GB")
        return False

if __name__ == "__main__":
    success = monitor_download()
    if success:
        print("🚀 Ready to test with your GNN adapter!")
    else:
        print("💡 Try cleaning up some space and retry")