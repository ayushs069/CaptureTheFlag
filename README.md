# 🎮 Capture the Flag - Q-Learning AI Game

A 2D grid-based Capture the Flag game with Reinforcement Learning AI using Q-Learning algorithm, built with Pygame.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.5.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Game Rules](#-game-rules)
- [Controls](#-controls)
- [Q-Learning Details](#-q-learning-details)
- [Project Structure](#-project-structure)
- [Training Guide](#-training-guide)
- [Architecture](#-architecture)
- [Customization](#-customization)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

- **Grid-based 2D Environment**: Clean pygame rendering with visual feedback
- **Two Game Modes**:
  - Human vs AI: Play against a Q-Learning AI agent
  - AI vs AI: Watch and train two AI agents playing against each other
- **Q-Learning Implementation**: Tabular Q-Learning with epsilon-greedy exploration
- **Persistent Learning**: Save and load Q-tables for continued training
- **Real-time Visualization**: See agents learn and compete in real-time
- **Modular Architecture**: Clean separation between game logic, AI, and rendering

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ayushs069/CaptureTheFlag.git
cd CaptureTheFlag

# Install dependencies
pip install -r requirements.txt

# Run the game
python main.py

# Run tests
python test_game.py

# Run examples
python examples.py
```

## 🎯 Game Rules

- **Objective**: Capture the enemy's flag and return it to your base
- **Teams**: 
  - Blue team (left side) - Human or AI
  - Red team (right side) - AI
- **Winning**: First team to capture the enemy flag and return to their base wins
- **Obstacles**: Gray blocks that cannot be passed through
- **Capturing**: If an agent reaches the enemy flag, they pick it up (yellow indicator shows carrier)
- **Defense**: If agents collide, the flag carrier drops the flag

### Visual Guide

```
[B] 🔵 🔷              🔶 🔴 [R]
│   │  └─Blue Flag    │  │  └─Red Base
│   └─Blue Agent      │  └─Red Agent
└─Blue Base           └─Red Flag
```

## 🎮 Controls

### Human vs AI Mode
- **Arrow Keys**: Move your agent (Blue team)
- **M**: Switch to AI vs AI mode
- **R**: Reset game
- **S**: Save AI Q-tables

### AI vs AI Training Mode
- **Space**: Pause/Resume training
- **M**: Switch to Human vs AI mode
- **R**: Reset game
- **S**: Save AI Q-tables

## Q-Learning Details

### State Representation
Each state is represented as a tuple containing:
- Agent position (row, col)
- Enemy position (row, col)
- Enemy flag position (row, col)
- Own flag position (row, col)
- Flag carrying status (0 or 1)

### Actions
- 0: Move Up
- 1: Move Down
- 2: Move Left
- 3: Move Right

### Rewards
- **+100**: Capture enemy flag and return to base
- **+20**: Defend own flag (catch enemy with your flag)
- **-10**: Get caught while carrying enemy flag
- **-1**: Each movement step (encourages efficiency)
- **+5**: Move toward enemy flag
- **-2**: Move away from objective

### Hyperparameters
- Learning Rate (α): 0.1
- Discount Factor (γ): 0.9
- Initial Epsilon (ε): 0.2
- Epsilon Decay: 0.995
- Minimum Epsilon: 0.05

## File Structure

```
CTF/
├── main.py                 # Main game loop and orchestration
├── game_environment.py     # Game environment and rendering
├── q_learning_agent.py     # Q-Learning agent implementation
├── config.py              # Configuration and constants
├── README.md              # This file
├── blue_agent_qtable.pkl  # Saved Q-table for blue agent (created after training)
└── red_agent_qtable.pkl   # Saved Q-table for red agent (created after training)
```

## How to Train the AI

1. Run the game: `python main.py`
2. Press **M** to switch to AI vs AI mode
3. Watch the agents learn through trial and error
4. Q-tables are automatically saved every 100 episodes
5. Press **S** at any time to manually save progress
6. The epsilon (exploration rate) will decay over time, making agents more greedy as they learn

## Tips for Playing

- The AI gets smarter over time through Q-Learning
- After training in AI vs AI mode, switch to Human vs AI to play against a trained agent
- Use obstacles strategically to avoid getting caught
- The yellow dot on an agent indicates they're carrying a flag

## Architecture

### Game Environment (`game_environment.py`)
- Manages game state, positions, and rules
- Handles rendering with Pygame
- Provides reward signals for Q-Learning
- Validates moves and detects win conditions

### Q-Learning Agent (`q_learning_agent.py`)
- Implements tabular Q-Learning algorithm
- Epsilon-greedy action selection
- Q-value updates using Bellman equation
- Persistent storage of learned Q-tables

### Main Game Loop (`main.py`)
- Orchestrates game modes
- Handles user input
- Manages training episodes
- Coordinates agent actions and environment updates

## 🎓 Training Guide

### Quick Training
1. Press **M** to switch to AI vs AI mode
2. Let it run for 1000+ episodes
3. Q-tables auto-save every 100 episodes

### Training Timeline
- **0-500 episodes** (30 min): Random exploration, learning basics
- **500-1000 episodes** (30 min): Emergent strategies
- **1000-2000 episodes** (1 hour): Optimized play
- **2000+ episodes** (1+ hour): Human-competitive performance

### Expected Results
After ~1000 episodes of training:
- Q-table size: 20,000-40,000 state-action pairs
- Average episode length: 50-100 steps
- Win rate: ~50% each team (balanced)
- Epsilon: ~0.05 (near minimum)

## 🏗️ Architecture

### Q-Learning Algorithm
```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]

Where:
  α (alpha) = Learning rate (0.1)
  γ (gamma) = Discount factor (0.9)
  r = Reward received
  s = Current state
  a = Action taken
  s' = Next state
```

### State Space
Each state is a 9-element tuple:
```python
state = (
    agent_row, agent_col,           # Agent position
    enemy_row, enemy_col,           # Enemy position
    enemy_flag_row, enemy_flag_col, # Target flag position
    own_flag_row, own_flag_col,     # Own flag position
    has_flag                        # Carrying flag? (0 or 1)
)
```

### Data Flow
```
User Input → Agent → Environment → Reward → Q-Learning Update
                ↓
           Update Q-table
                ↓
           Better Strategy
```

## 🔧 Customization

Edit `config.py` to modify:

### Learning Parameters
```python
LEARNING_RATE = 0.1        # How fast agent learns
DISCOUNT_FACTOR = 0.9      # Future reward importance
EPSILON = 0.2              # Exploration rate
EPSILON_DECAY = 0.995      # Exploration decay rate
```

### Rewards
```python
REWARD_CAPTURE_FLAG = 100  # Win the game
REWARD_DEFEND_FLAG = 20    # Catch enemy with your flag
REWARD_GET_CAUGHT = -10    # Caught while carrying flag
REWARD_STEP = -1           # Each movement
```

### Grid Settings
```python
GRID_ROWS = 12            # Number of rows
GRID_COLS = 16            # Number of columns
CELL_SIZE = 50            # Size of each cell in pixels
```

## 📝 Examples

### Example 1: Basic Environment Usage
```python
from game_environment import CTFEnvironment

env = CTFEnvironment()
state = env.get_state('blue')
next_state, reward, done = env.move_agent('blue', 3)  # Move right
```

### Example 2: Training an Agent
```python
from q_learning_agent import QLearningAgent

agent = QLearningAgent("My Agent")
state = env.get_state('red')
action = agent.choose_action(state, training=True)
next_state, reward, done = env.move_agent('red', action)
agent.update_q_value(state, action, reward, next_state)
```

### Example 3: Save/Load Q-tables
```python
# Save
agent.save_q_table("my_agent.pkl")

# Load
agent.load_q_table("my_agent.pkl")
```

Run `python examples.py` to see more detailed examples.

## 🐛 Troubleshooting

**Q: The AI doesn't seem to learn**
- A: Train for more episodes (1000+), check that epsilon is decaying

**Q: Game too fast/slow**
- A: Adjust `PLAYER_SPEED` and `AI_SPEED` in config.py

**Q: Want to reset AI learning**
- A: Delete `*.pkl` files to start fresh

**Q: One team always wins**
- A: Normal at first, balances out after more training (500+ episodes)

**Q: Installation issues**
- A: Make sure you have Python 3.7+ and run `pip install pygame`

## 📊 Performance Metrics

### Computational Complexity
- State lookup: O(1) (dictionary hash)
- Action selection: O(4) (small action space)
- Q-value update: O(4)
- Memory: ~1-5 MB per Q-table

### Training Efficiency
- Episodes for basic competence: ~500-1000
- Episodes for good performance: ~2000-5000
- Time per episode: 5-20 seconds
- Total training time: 1-2 hours for competitive AI

## 🚀 Advanced Features

Potential extensions:
- Multiple flags
- Larger teams (2v2, 3v3)
- Power-ups and special abilities
- Deep Q-Learning with neural networks
- Different map layouts
- Tournament mode

## 📄 License

MIT License - Free to use and modify for educational purposes.

## 👤 Author

Created as a demonstration of Reinforcement Learning principles in game development.

## 🙏 Acknowledgments

- Built with [Pygame](https://www.pygame.org/)
- Inspired by classic Capture the Flag games
- Educational project for learning Q-Learning

## 📚 Learn More

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [Pygame Documentation](https://www.pygame.org/docs/)
- [Q-Learning Tutorial](https://en.wikipedia.org/wiki/Q-learning)

---

**Enjoy playing and training your AI!** 🎮🤖

If you find this project helpful, please star ⭐ the repository!
