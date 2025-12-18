"""
Q-Learning Agent for CTF game
Implements tabular Q-Learning algorithm
"""

import random
import pickle
from collections import defaultdict
from config import *


class QLearningAgent:
    def __init__(self, name="Agent", learning_rate=LEARNING_RATE, 
                 discount_factor=DISCOUNT_FACTOR, epsilon=EPSILON):
        """
        Initialize Q-Learning agent
        
        Args:
            name: Agent name for identification
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate
        """
        self.name = name
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = MIN_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        
        # Q-table: dictionary of (state, action) -> Q-value
        self.q_table = defaultdict(float)
        
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.actions = [0, 1, 2, 3]
        
        # Training statistics
        self.total_episodes = 0
        self.total_rewards = 0
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair"""
        return self.q_table[(state, action)]
    
    def get_best_action(self, state):
        """Get the best action for a given state (greedy)"""
        q_values = [self.get_q_value(state, action) for action in self.actions]
        max_q = max(q_values)
        
        # If multiple actions have the same max Q-value, choose randomly
        best_actions = [action for action, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def choose_action(self, state, training=True):
        """
        Choose an action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            action: Selected action
        """
        if training and random.random() < self.epsilon:
            # Exploration: choose random action
            return random.choice(self.actions)
        else:
            # Exploitation: choose best action
            return self.get_best_action(state)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-Learning update rule
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.get_q_value(state, action)
        
        # Find max Q-value for next state
        next_q_values = [self.get_q_value(next_state, a) for a in self.actions]
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Q-Learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[(state, action)] = new_q
        self.total_rewards += reward
    
    def decay_epsilon(self):
        """Decay epsilon for less exploration over time"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self, filename):
        """Save Q-table to file"""
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {filename}")
    
    def load_q_table(self, filename):
        """Load Q-table from file"""
        try:
            with open(filename, 'rb') as f:
                loaded_table = pickle.load(f)
                self.q_table = defaultdict(float, loaded_table)
            print(f"Q-table loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"No saved Q-table found at {filename}")
            return False
    
    def get_statistics(self):
        """Get training statistics"""
        return {
            'total_episodes': self.total_episodes,
            'total_rewards': self.total_rewards,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table)
        }
    
    def reset_episode(self):
        """Reset for new episode"""
        self.total_episodes += 1
    
    def simple_heuristic_action(self, state):
        """
        Simple heuristic for decision making (used before Q-table is trained)
        Move towards enemy flag if not carrying flag
        Move towards own base if carrying flag
        """
        agent_row, agent_col = state[0], state[1]
        target_flag_row, target_flag_col = state[4], state[5]
        own_base_row, own_base_col = state[6], state[7]
        has_flag = state[8]
        
        if has_flag:
            # Move towards own base
            target_row, target_col = own_base_row, own_base_col
        else:
            # Move towards enemy flag
            target_row, target_col = target_flag_row, target_flag_col
        
        # Calculate direction
        row_diff = target_row - agent_row
        col_diff = target_col - agent_col
        
        # Prioritize larger difference
        if abs(row_diff) > abs(col_diff):
            if row_diff < 0:
                return 0  # Up
            else:
                return 1  # Down
        else:
            if col_diff < 0:
                return 2  # Left
            else:
                return 3  # Right


class HumanAgent:
    """Simple wrapper for human player"""
    def __init__(self, name="Human"):
        self.name = name
        self.current_action = None
    
    def set_action(self, action):
        """Set action from keyboard input"""
        self.current_action = action
    
    def get_action(self):
        """Get and clear current action"""
        action = self.current_action
        self.current_action = None
        return action
