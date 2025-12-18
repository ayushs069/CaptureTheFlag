"""
Quick test script to verify the CTF game components
"""

from game_environment import CTFEnvironment
from q_learning_agent import QLearningAgent
from config import *


def test_environment():
    """Test game environment"""
    print("Testing Game Environment...")
    env = CTFEnvironment()
    
    # Check initial state
    blue_state = env.get_state('blue')
    red_state = env.get_state('red')
    
    print(f"  Blue initial state: {blue_state}")
    print(f"  Red initial state: {red_state}")
    print(f"  Obstacles count: {len(env.obstacles)}")
    print(f"  Grid size: {env.grid_rows} x {env.grid_cols}")
    
    # Test a few moves
    print("\n  Testing blue agent movement...")
    for action in [3, 3, 1, 1]:  # Right, Right, Down, Down
        next_state, reward, done = env.move_agent('blue', action)
        print(f"    Action: {action}, Reward: {reward}, Done: {done}")
    
    print("  ✓ Environment test passed!\n")


def test_agent():
    """Test Q-Learning agent"""
    print("Testing Q-Learning Agent...")
    agent = QLearningAgent("Test Agent")
    
    # Create a dummy state
    state = (2, 2, 9, 13, 9, 13, 2, 2, 0)
    
    # Test action selection
    action = agent.choose_action(state, training=True)
    print(f"  Selected action (with exploration): {action}")
    
    # Test Q-value update
    next_state = (2, 3, 9, 13, 9, 13, 2, 2, 0)
    reward = -1
    agent.update_q_value(state, action, reward, next_state)
    
    q_value = agent.get_q_value(state, action)
    print(f"  Q-value after update: {q_value}")
    
    # Test best action
    best_action = agent.get_best_action(state)
    print(f"  Best action (greedy): {best_action}")
    
    print(f"  Q-table size: {len(agent.q_table)}")
    print("  ✓ Agent test passed!\n")


def test_integration():
    """Test environment and agent integration"""
    print("Testing Integration (Environment + Agent)...")
    env = CTFEnvironment()
    agent = QLearningAgent("Test Agent")
    
    # Run a short episode
    state = env.get_state('red')
    total_reward = 0
    
    for step in range(10):
        action = agent.choose_action(state, training=True)
        next_state, reward, done = env.move_agent('red', action)
        agent.update_q_value(state, action, reward, next_state)
        
        total_reward += reward
        state = next_state
        
        if done:
            print(f"  Episode ended at step {step}")
            break
    
    print(f"  Completed 10 steps")
    print(f"  Total reward: {total_reward}")
    print(f"  Q-table size: {len(agent.q_table)}")
    print("  ✓ Integration test passed!\n")


def main():
    print("=" * 50)
    print("CTF Game Component Tests")
    print("=" * 50)
    print()
    
    try:
        test_environment()
        test_agent()
        test_integration()
        
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        print("\nYou can now run the main game with: python main.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
