#!/bin/bash
# This script sets up the environment for the project
echo "Setting up the project environment..."
sudo apt update && sudo apt upgrade
sudo apt install -y neovim btop git curl vulkan-tools libegl1 libgl1 libglx0 libglvnd0 libgles2
curl https://raw.githubusercontent.com/yuvibirdi/dotfiles-backup/refs/heads/master/ssh/.config/fish/config.fish -o ~/.config/fish/config.fish
