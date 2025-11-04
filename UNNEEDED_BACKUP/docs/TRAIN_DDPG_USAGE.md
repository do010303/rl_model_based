# 🚀 Updated Train DDPG Usage Guide

The `train_ddpg.py` script has been enhanced to support configurable training parameters!

## ✅ **Key Features Added:**
- **Configurable Episodes**: Set any number of training episodes
- **Multiple Training Methods**: Choose between DDPG and MBPO
- **Flexible Parameters**: Customize training settings
- **Better Output**: Enhanced progress reporting and statistics

## 📋 **Usage Examples:**

### **Basic Training with Custom Episodes:**
```bash
# Train for 50 episodes using DDPG
python3 examples/train_ddpg.py --episodes 50 --method ddpg

# Train for 100 episodes using MBPO (default method)
python3 examples/train_ddpg.py --episodes 100

# Short test run (10 episodes)
python3 examples/train_ddpg.py -e 10 -m ddpg
```

### **Advanced Configuration:**
```bash
# Custom training with all parameters
python3 examples/train_ddpg.py \
    --episodes 150 \
    --method ddpg \
    --max-steps 250 \
    --success-distance 0.03 \
    --render

# Long training session for better results
python3 examples/train_ddpg.py --episodes 500 --method mbpo
```

### **Quick Testing:**
```bash
# Very fast test (5 episodes)
python3 examples/train_ddpg.py -e 5 -m ddpg

# Medium test (25 episodes)  
python3 examples/train_ddpg.py -e 25 -m ddpg
```

## 🎛️ **Available Parameters:**

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--episodes` | `-e` | 200 | Number of training episodes |
| `--method` | `-m` | mbpo | Training method (ddpg/mbpo) |
| `--render` | `-r` | False | Show visualization during training |
| `--max-steps` | | 200 | Maximum steps per episode |
| `--success-distance` | | 0.05 | Success threshold in meters |

## 📊 **Expected Training Times:**
- **5 episodes**: ~10-15 seconds
- **25 episodes**: ~1-2 minutes  
- **100 episodes**: ~5-10 minutes
- **200 episodes**: ~15-20 minutes
- **500+ episodes**: ~30-60 minutes

## 🎯 **Recommendations:**
- **Quick Testing**: 5-10 episodes
- **Development**: 25-50 episodes
- **Good Results**: 100-200 episodes  
- **Best Performance**: 300-500+ episodes
- **Use MBPO method** for better results (default)
- **Use DDPG method** for faster training and debugging

Now you can easily experiment with different episode counts to find the optimal training duration for your needs! 🚀