"""
Example usage and demonstration of CTF Q-Learning components
This file shows how to use the game components programmatically
"""

from game_environment import CTFEnvironment
from q_learning_agent import QLearningAgent
from config import *
import time


def example_1_basic_environment():
    """Example 1: Basic environment usage"""
    print("=" * 60)
    print("Example 1: Basic Environment Usage")
    print("=" * 60)
    
    # Create environment
    env = CTFEnvironment()
    
    # Get initial state
    blue_state = env.get_state('blue')
    red_state = env.get_state('red')
    
    print(f"\nInitial Blue State: {blue_state}")
    print(f"Initial Red State: {red_state}")
    
    # Make some moves
    print("\nMoving blue agent right 3 times:")
    for i in range(3):
        state, reward, done = env.move_agent('blue', 3)  # Move right
        print(f"  Step {i+1}: Position={state[:2]}, Reward={reward}, Done={done}")
    
    print("\n" + "=" * 60 + "\n")


def example_2_basic_agent():
    """Example 2: Basic agent usage"""
    print("=" * 60)
    print("Example 2: Basic Agent Usage")
    print("=" * 60)
    
    # Create agent
    agent = QLearningAgent("Test Agent", epsilon=0.2)
    
    # Create a state
    state = (2, 2, 9, 13, 9, 13, 2, 2, 0)
    
    # Choose actions
    print(f"\nState: {state}")
    print("\nChoosing 5 actions with exploration (epsilon=0.2):")
    for i in range(5):
        action = agent.choose_action(state, training=True)
        action_name = ['Up', 'Down', 'Left', 'Right'][action]
        print(f"  Action {i+1}: {action} ({action_name})")
    
    # Update Q-values
    print("\nUpdating Q-values for different actions:")
    for action in range(4):
        next_state = (2, 3, 9, 13, 9, 13, 2, 2, 0)
        reward = -1
        agent.update_q_value(state, action, reward, next_state)
        q_val = agent.get_q_value(state, action)
        print(f"  Action {action}: Q-value = {q_val:.3f}")
    
    print(f"\nQ-table size: {len(agent.q_table)}")
    
    print("\n" + "=" * 60 + "\n")


def example_3_training_episode():
    """Example 3: Single training episode"""
    print("=" * 60)
    print("Example 3: Single Training Episode")
    print("=" * 60)
    
    # Create environment and agents
    env = CTFEnvironment()
    blue_agent = QLearningAgent("Blue Agent", epsilon=0.3)
    red_agent = QLearningAgent("Red Agent", epsilon=0.3)
    
    # Run one episode
    print("\nRunning one training episode (max 50 steps):")
    
    blue_total_reward = 0
    red_total_reward = 0
    
    for step in range(50):
        # Blue agent move
        blue_state = env.get_state('blue')
        blue_action = blue_agent.choose_action(blue_state, training=True)
        blue_next_state, blue_reward, blue_done = env.move_agent('blue', blue_action)
        blue_agent.update_q_value(blue_state, blue_action, blue_reward, blue_next_state)
        blue_total_reward += blue_reward
        
        # Red agent move
        red_state = env.get_state('red')
        red_action = red_agent.choose_action(red_state, training=True)
        red_next_state, red_reward, red_done = env.move_agent('red', red_action)
        red_agent.update_q_value(red_state, red_action, red_reward, red_next_state)
        red_total_reward += red_reward
        
        # Check if game over
        if env.game_over:
            print(f"\n  Episode ended at step {step+1}")
            print(f"  Winner: {env.winner.upper()}")
            print(f"  Blue total reward: {blue_total_reward}")
            print(f"  Red total reward: {red_total_reward}")
            break
    else:
        print(f"\n  Episode reached max steps (50)")
        print(f"  Blue total reward: {blue_total_reward}")
        print(f"  Red total reward: {red_total_reward}")
    
    print(f"\n  Blue Q-table size: {len(blue_agent.q_table)}")
    print(f"  Red Q-table size: {len(red_agent.q_table)}")
    
    print("\n" + "=" * 60 + "\n")


def example_4_multi_episode_training():
    """Example 4: Multiple episode training"""
    print("=" * 60)
    print("Example 4: Multi-Episode Training (10 episodes)")
    print("=" * 60)
    
    # Create environment and agents
    env = CTFEnvironment()
    blue_agent = QLearningAgent("Blue Agent", epsilon=0.3)
    red_agent = QLearningAgent("Red Agent", epsilon=0.3)
    
    # Statistics
    blue_wins = 0
    red_wins = 0
    
    print("\nTraining for 10 episodes:")
    
    for episode in range(10):
        # Reset environment
        env.reset()
        
        # Run episode
        for step in range(100):  # Max 100 steps per episode
            # Blue agent
            blue_state = env.get_state('blue')
            blue_action = blue_agent.choose_action(blue_state, training=True)
            blue_next_state, blue_reward, _ = env.move_agent('blue', blue_action)
            blue_agent.update_q_value(blue_state, blue_action, blue_reward, blue_next_state)
            
            # Red agent
            red_state = env.get_state('red')
            red_action = red_agent.choose_action(red_state, training=True)
            red_next_state, red_reward, _ = env.move_agent('red', red_action)
            red_agent.update_q_value(red_state, red_action, red_reward, red_next_state)
            
            if env.game_over:
                break
        
        # Record winner
        if env.winner == 'blue':
            blue_wins += 1
        elif env.winner == 'red':
            red_wins += 1
        
        # Decay epsilon
        blue_agent.decay_epsilon()
        red_agent.decay_epsilon()
        
        print(f"  Episode {episode+1}: Winner={env.winner or 'None':5s}, "
              f"Steps={step+1:3d}, "
              f"Blue ε={blue_agent.epsilon:.3f}, "
              f"Red ε={red_agent.epsilon:.3f}")
    
    print(f"\nResults after 10 episodes:")
    print(f"  Blue wins: {blue_wins}")
    print(f"  Red wins: {red_wins}")
    print(f"  Draws: {10 - blue_wins - red_wins}")
    print(f"  Blue Q-table size: {len(blue_agent.q_table)}")
    print(f"  Red Q-table size: {len(red_agent.q_table)}")
    
    print("\n" + "=" * 60 + "\n")


def example_5_save_load_qtable():
    """Example 5: Saving and loading Q-tables"""
    print("=" * 60)
    print("Example 5: Save and Load Q-Tables")
    print("=" * 60)
    
    # Create and train agent
    agent = QLearningAgent("Demo Agent")
    
    # Add some Q-values
    print("\nAdding some Q-values to agent...")
    for i in range(10):
        state = (i, i, 5, 5, 8, 8, 2, 2, 0)
        for action in range(4):
            agent.q_table[(state, action)] = float(i * action)
    
    print(f"Q-table size before save: {len(agent.q_table)}")
    
    # Save Q-table
    filename = "demo_qtable.pkl"
    agent.save_q_table(filename)
    
    # Create new agent and load Q-table
    new_agent = QLearningAgent("New Agent")
    print(f"\nNew agent Q-table size before load: {len(new_agent.q_table)}")
    
    new_agent.load_q_table(filename)
    print(f"New agent Q-table size after load: {len(new_agent.q_table)}")
    
    # Verify loaded correctly
    test_state = (5, 5, 5, 5, 8, 8, 2, 2, 0)
    test_action = 2
    original_q = agent.get_q_value(test_state, test_action)
    loaded_q = new_agent.get_q_value(test_state, test_action)
    
    print(f"\nVerification:")
    print(f"  Original Q-value: {original_q}")
    print(f"  Loaded Q-value: {loaded_q}")
    print(f"  Match: {original_q == loaded_q}")
    
    # Cleanup
    import os
    if os.path.exists(filename):
        os.remove(filename)
        print(f"\nCleaned up {filename}")
    
    print("\n" + "=" * 60 + "\n")


def example_6_agent_comparison():
    """Example 6: Compare random vs greedy agent"""
    print("=" * 60)
    print("Example 6: Random vs Greedy Agent Comparison")
    print("=" * 60)
    
    # Create environment
    env = CTFEnvironment()
    
    # Create two agents
    random_agent = QLearningAgent("Random Agent", epsilon=1.0)  # Always explore
    greedy_agent = QLearningAgent("Greedy Agent", epsilon=0.0)  # Never explore
    
    # Pre-train greedy agent with some knowledge
    print("\nPre-training greedy agent with heuristic knowledge...")
    # Add some basic Q-values favoring movement toward enemy base
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            state = (row, col, 9, 13, 9, 13, 2, 2, 0)
            # Favor actions that move toward enemy (bottom-right)
            if row < 9:
                greedy_agent.q_table[(state, 1)] = 1.0  # Down is good
            if col < 13:
                greedy_agent.q_table[(state, 3)] = 1.0  # Right is good
    
    print(f"Greedy agent Q-table size: {len(greedy_agent.q_table)}")
    
    # Compare action selection
    test_state = (2, 2, 9, 13, 9, 13, 2, 2, 0)
    
    print(f"\nTest state: {test_state}")
    print("\nRandom agent (epsilon=1.0) - 10 actions:")
    random_actions = [random_agent.choose_action(test_state, training=True) 
                      for _ in range(10)]
    print(f"  Actions: {random_actions}")
    
    print("\nGreedy agent (epsilon=0.0) - 10 actions:")
    greedy_actions = [greedy_agent.choose_action(test_state, training=False) 
                      for _ in range(10)]
    print(f"  Actions: {greedy_actions}")
    
    print("\n" + "=" * 60 + "\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("CTF Q-Learning - Component Examples")
    print("=" * 60 + "\n")
    
    try:
        example_1_basic_environment()
        time.sleep(0.5)
        
        example_2_basic_agent()
        time.sleep(0.5)
        
        example_3_training_episode()
        time.sleep(0.5)
        
        example_4_multi_episode_training()
        time.sleep(0.5)
        
        example_5_save_load_qtable()
        time.sleep(0.5)
        
        example_6_agent_comparison()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nNow you can:")
        print("  1. Run the full game: python main.py")
        print("  2. Read the documentation: README.md")
        print("  3. Check training guide: TRAINING_GUIDE.md")
        print("  4. Explore architecture: ARCHITECTURE.md")
        
    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
