import shutil
import os
from huggingface_hub import try_to_load_from_cache
from transformers.utils import clean_cache

def check_disk_space():
    """Check disk space in the current directory and cache directory"""
    # Check current directory space
    total, used, free = shutil.disk_usage("/")
    
    # Convert to GB for readable format
    total_gb = total // (2**30)
    used_gb = used // (2**30)
    free_gb = free // (2**30)
    
    print("\nDisk Space Information:")
    print(f"Total: {total_gb}GB")
    print(f"Used: {used_gb}GB")
    print(f"Free: {free_gb}GB")
    
    # Check Hugging Face cache directory
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        cache_usage = sum(os.path.getsize(os.path.join(dirpath,filename)) 
                         for dirpath, _, filenames in os.walk(cache_dir)
                         for filename in filenames)
        cache_usage_gb = cache_usage // (2**30)
        print(f"\nHugging Face Cache Usage: {cache_usage_gb}GB")
        print(f"Cache Directory: {cache_dir}")

if __name__ == "__main__":
    check_disk_space()

    # Clean the cache
    clean_cache()