#!/usr/bin/env python3
import os
import sys
import argparse
import time
from pathlib import Path

# Try to import dotenv for loading .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    parser = argparse.ArgumentParser(description="Launch a Vast.ai instance for SWAMLA training.")
    parser.add_argument("--api_key", type=str, help="Vast.ai API Key (or set VAST_AI_API_KEY env var)")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--gpu_name", type=str, default="H100 SXM", help="GPU model name (e.g., 'H100 SXM', 'RTX 4090')")
    parser.add_argument("--image", type=str, default="vastai/pytorch", help="Docker image to use")
    parser.add_argument("--disk_space", type=float, default=40.0, help="Disk space in GB")
    parser.add_argument("--region", type=str, help="Region filter (optional)")
    parser.add_argument("--ports", type=int, nargs='+', default=[6006, 8888, 8080], help="Ports to expose (default: 6006 8888 8080)")
    parser.add_argument("--repo_url", type=str, default="https://github.com/Orosius/swamla.git", help="Git repository URL")
    parser.add_argument("--install_script", type=str, default="scripts/install.sh", help="Path to install script (local path)")
    
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("VAST_AI_API_KEY")
    if not api_key:
        print("Error: Vast.ai API Key is required. Set VAST_AI_API_KEY env var or pass --api_key.")
        sys.exit(1)

    try:
        from vastai_launcher import VastLauncher
    except ImportError:
        print("Error: vastai_launcher library not found. Please install it first.")
        sys.exit(1)

    print(f"Initializing VastLauncher...")
    launcher = VastLauncher(api_key=api_key)

    print(f"Searching for {args.gpu_count}x {args.gpu_name}...")
    print(f"Exposing ports: {args.ports}")
    try:
        instance_id = launcher.launch_instance(
            gpu_count=args.gpu_count,
            gpu_name=args.gpu_name,
            image=args.image,
            disk_space=args.disk_space,
            region=args.region,
            ports=args.ports
        )
        print(f"Successfully launched instance ID: {instance_id}")
    except Exception as e:
        print(f"Failed to launch instance: {e}")
        sys.exit(1)

    print(f"Waiting for instance {instance_id} to be ready (this may take a few minutes)...")
    try:
        launcher.wait_for_ready(instance_id)
        print("Instance is ready!")
    except Exception as e:
        print(f"Error waiting for instance: {e}")
        # We don't exit here, we might still want to try setup or let user handle it
    
    print("Setting up project...")
    try:
        # Get env vars
        hf_token = os.getenv("HF_TOKEN", "")
        wandb_key = os.getenv("WANDB_API_KEY", "")
        
        # Inject env vars into .bashrc so they persist and are available for setup
        print("Configuring environment variables...")
        env_cmds = [
            f"echo 'export HF_TOKEN={hf_token}' >> ~/.bashrc",
            f"echo 'export WANDB_API_KEY={wandb_key}' >> ~/.bashrc",
            "source ~/.bashrc"
        ]
        launcher.execute_commands(instance_id, env_cmds)

        # Run setup script using the library's feature
        launcher.setup_project(
            instance_id=instance_id,
            repo_url=args.repo_url,
            setup_script=args.install_script
        )
        print("Setup complete!")
        
        # Get port info
        print("\n" + "="*50)
        print(f"Instance {instance_id} is ready for training.")
        
        try:
            port_info = launcher.get_ports(instance_id)
            print("\nExposed Ports:")
            for port, info in port_info.items():
                print(f"  - {port}: {info['url']}")
        except Exception:
            # Fallback if get_ports fails or isn't available yet
            print(f"To connect via SSH, check your Vast.ai console or use 'vastai ssh-url {instance_id}'")
            
        print("="*50 + "\n")

    except Exception as e:
        print(f"Error during project setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
