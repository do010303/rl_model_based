#!/bin/bash

# 🚀 Robotarm-RL-4DoF Quick Setup Script
# This script helps you get started with the 4-DOF Robot Arm RL project

echo "🤖 Welcome to Robotarm-RL-4DoF Setup!"
echo "=====================================+"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}🔍 Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d" " -f2)
    echo -e "${GREEN}✅ Python ${PYTHON_VERSION} found${NC}"
else
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check if virtual environment should be created
echo ""
echo -e "${BLUE}🌐 Virtual Environment Setup${NC}"
read -p "Create virtual environment? (recommended) [Y/n]: " create_venv
create_venv=${create_venv:-Y}

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}📦 Creating virtual environment 'robotarm_env'...${NC}"
    python3 -m venv robotarm_env
    
    echo -e "${YELLOW}🔌 Activating virtual environment...${NC}"
    source robotarm_env/bin/activate
    
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
    echo -e "${YELLOW}📋 To activate later: source robotarm_env/bin/activate${NC}"
fi

# Install dependencies
echo ""
echo -e "${BLUE}📦 Installing Dependencies${NC}"
echo -e "${YELLOW}⏳ Installing core packages...${NC}"

# Install basic packages first
pip3 install numpy matplotlib

echo -e "${YELLOW}⏳ Installing ML/RL packages...${NC}"
pip3 install tensorflow gymnasium

echo -e "${GREEN}✅ Dependencies installed successfully${NC}"

# Run tests
echo ""
echo -e "${BLUE}🧪 Running Tests${NC}"
echo -e "${YELLOW}⏳ Testing project functionality...${NC}"

if python3 simple_test.py; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Some tests failed. Check the output above.${NC}"
fi

# Demo
echo ""
echo -e "${BLUE}🎮 Ready to Run Demo${NC}"
read -p "Run interactive demo? [Y/n]: " run_demo
run_demo=${run_demo:-Y}

if [[ $run_demo =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🚀 Launching demo...${NC}"
    python3 demo.py
fi

# Success message
echo ""
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "==================="
echo ""
echo -e "${BLUE}🚀 Quick Start Commands:${NC}"
echo -e "${YELLOW}  python3 demo.py                     ${NC}# Interactive demo"
echo -e "${YELLOW}  python3 examples/train_ddpg.py      ${NC}# Start training"
echo -e "${YELLOW}  python3 examples/train_curriculum.py${NC}# Curriculum learning"
echo -e "${YELLOW}  python3 test_project.py             ${NC}# Run full tests"
echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo -e "${YELLOW}  README.md                           ${NC}# Project documentation"
echo -e "${YELLOW}  PROJECT_COMPLETION_SUMMARY.md       ${NC}# Complete overview"
echo -e "${YELLOW}  CONTRIBUTING.md                     ${NC}# Development guidelines"
echo ""
echo -e "${GREEN}🌟 Happy Coding with Robotarm-RL-4DoF!${NC}"
