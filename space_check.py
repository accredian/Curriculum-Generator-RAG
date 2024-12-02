import subprocess
import os

def get_directory_size(path):
    """Get directory size using du command"""
    try:
        result = subprocess.run(['du', '-sh', path], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            size = result.stdout.split()[0]
            return size
        return "Error"
    except Exception:
        return "Error"

def analyze_space():
    """Analyze disk space usage safely"""
    print("Analyzing disk space usage...\n")
    
    # Key directories to check
    directories = {
        "Python Environment": "/home/codespace/.python",
        "VS Code Server": "/home/codespace/.vscode-server",
        "Project Files": "/workspaces/Curriculum_Generator",
        "NPM Packages": "/home/codespace/.npm",
        "Node Modules": "/workspaces/node_modules",
        "Git": "/workspaces/.git"
    }
    
    print("Directory Sizes:")
    print("-" * 50)
    for name, path in directories.items():
        if os.path.exists(path):
            size = get_directory_size(path)
            print(f"{name:<20}: {size}")
    
    print("\nLargest Files:")
    print("-" * 50)
    try:
        # Find top 10 largest files, excluding symbolic links
        cmd = "find /workspaces /home/codespace -type f -not -type l -exec du -h {} + 2>/dev/null | sort -rh | head -n 10"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Error finding largest files: {e}")
    
    # Get overall disk usage
    total, used, free = os.statvfs('/')
    total_gb = (total * free.f_frsize) / (1024**3)
    used_gb = ((total - free) * free.f_frsize) / (1024**3)
    free_gb = (free * free.f_frsize) / (1024**3)
    
    print("\nOverall Disk Usage:")
    print("-" * 50)
    print(f"Total: {total_gb:.2f}GB")
    print(f"Used:  {used_gb:.2f}GB")
    print(f"Free:  {free_gb:.2f}GB")

if __name__ == "__main__":
    analyze_space()