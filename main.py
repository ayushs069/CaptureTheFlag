"""
Main game loop for Capture the Flag with Q-Learning
"""

import pygame
import sys
from game_environment import CTFEnvironment
from q_learning_agent import QLearningAgent, HumanAgent
from config import *


class CTFGame:
    def __init__(self):
        """Initialize the game"""
        pygame.init()
        
        # Create window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Capture the Flag - Q-Learning AI")
        
        # Clock for FPS
        self.clock = pygame.time.Clock()
        
        # Font
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        
        # Create environment
        self.env = CTFEnvironment()
        
        # Create agents
        self.blue_agent = HumanAgent("Blue Player")
        self.red_agent = QLearningAgent("Red AI", epsilon=EPSILON)
        
        # Try to load pre-trained Q-table
        self.red_agent.load_q_table("red_agent_qtable.pkl")
        
        # Game state
        self.mode = MODE_HUMAN_VS_AI
        self.running = True
        self.paused = False
        self.episode = 0
        
        # Timing
        self.last_move_time = 0
        self.move_delay = PLAYER_SPEED
        
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                # Mode switching
                if event.key == pygame.K_m:
                    self.switch_mode()
                
                # Reset game
                elif event.key == pygame.K_r:
                    self.reset_game()
                
                # Pause (only in AI vs AI mode)
                elif event.key == pygame.K_SPACE and self.mode == MODE_AI_VS_AI:
                    self.paused = not self.paused
                
                # Save Q-table
                elif event.key == pygame.K_s:
                    self.save_agents()
                
                # Human player controls (only in Human vs AI mode)
                elif self.mode == MODE_HUMAN_VS_AI and not self.env.game_over:
                    if event.key == pygame.K_UP:
                        self.blue_agent.set_action(0)
                    elif event.key == pygame.K_DOWN:
                        self.blue_agent.set_action(1)
                    elif event.key == pygame.K_LEFT:
                        self.blue_agent.set_action(2)
                    elif event.key == pygame.K_RIGHT:
                        self.blue_agent.set_action(3)
    
    def switch_mode(self):
        """Switch between game modes"""
        if self.mode == MODE_HUMAN_VS_AI:
            self.mode = MODE_AI_VS_AI
            # Create blue AI agent for AI vs AI mode
            self.blue_agent = QLearningAgent("Blue AI", epsilon=EPSILON)
            self.blue_agent.load_q_table("blue_agent_qtable.pkl")
            print("Switched to AI vs AI mode")
        else:
            self.mode = MODE_HUMAN_VS_AI
            # Create human agent for Human vs AI mode
            self.blue_agent = HumanAgent("Blue Player")
            print("Switched to Human vs AI mode")
        
        self.reset_game()
    
    def reset_game(self):
        """Reset the game"""
        self.env.reset()
        self.episode += 1
        
        if isinstance(self.blue_agent, QLearningAgent):
            self.blue_agent.reset_episode()
        if isinstance(self.red_agent, QLearningAgent):
            self.red_agent.reset_episode()
    
    def save_agents(self):
        """Save agent Q-tables"""
        if isinstance(self.blue_agent, QLearningAgent):
            self.blue_agent.save_q_table("blue_agent_qtable.pkl")
        self.red_agent.save_q_table("red_agent_qtable.pkl")
        print("Agent Q-tables saved!")
    
    def update_human_vs_ai(self):
        """Update game in Human vs AI mode"""
        current_time = pygame.time.get_ticks()
        
        if current_time - self.last_move_time < self.move_delay:
            return
        
        self.last_move_time = current_time
        
        if self.env.game_over:
            return
        
        # Human player move
        human_action = self.blue_agent.get_action()
        if human_action is not None:
            state = self.env.get_state('blue')
            next_state, reward, done = self.env.move_agent('blue', human_action)
        
        # AI agent move
        red_state = self.env.get_state('red')
        red_action = self.red_agent.choose_action(red_state, training=False)
        red_next_state, red_reward, red_done = self.env.move_agent('red', red_action)
        
        # Check if game is over
        if self.env.game_over:
            print(f"Game Over! Winner: {self.env.winner}")
    
    def update_ai_vs_ai(self):
        """Update game in AI vs AI mode (training)"""
        if self.paused or self.env.game_over:
            return
        
        current_time = pygame.time.get_ticks()
        
        if current_time - self.last_move_time < AI_SPEED:
            return
        
        self.last_move_time = current_time
        
        # Blue AI move
        blue_state = self.env.get_state('blue')
        blue_action = self.blue_agent.choose_action(blue_state, training=True)
        blue_next_state, blue_reward, blue_done = self.env.move_agent('blue', blue_action)
        self.blue_agent.update_q_value(blue_state, blue_action, blue_reward, blue_next_state)
        
        # Red AI move
        red_state = self.env.get_state('red')
        red_action = self.red_agent.choose_action(red_state, training=True)
        red_next_state, red_reward, red_done = self.env.move_agent('red', red_action)
        self.red_agent.update_q_value(red_state, red_action, red_reward, red_next_state)
        
        # Check if episode is done
        if self.env.game_over:
            print(f"Episode {self.episode}: Winner: {self.env.winner}, "
                  f"Blue Score: {self.env.blue_score}, Red Score: {self.env.red_score}")
            
            # Decay epsilon
            self.blue_agent.decay_epsilon()
            self.red_agent.decay_epsilon()
            
            # Auto-reset after a short delay
            pygame.time.wait(500)
            self.reset_game()
            
            # Save Q-tables periodically
            if self.episode % 100 == 0:
                self.save_agents()
                print(f"Q-tables saved at episode {self.episode}")
                print(f"Blue Agent - Q-table size: {len(self.blue_agent.q_table)}, "
                      f"Epsilon: {self.blue_agent.epsilon:.3f}")
                print(f"Red Agent - Q-table size: {len(self.red_agent.q_table)}, "
                      f"Epsilon: {self.red_agent.epsilon:.3f}")
    
    def render(self):
        """Render the game"""
        # Render environment
        self.env.render(self.screen)
        
        # Render UI
        episode_num = self.episode if self.mode == MODE_AI_VS_AI else None
        self.env.render_ui(self.screen, self.font, self.mode, episode_num)
        
        # Show paused text
        if self.paused:
            paused_text = self.title_font.render("PAUSED", True, RED)
            paused_rect = paused_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            
            # Draw background rectangle
            bg_rect = paused_rect.inflate(20, 20)
            pygame.draw.rect(self.screen, WHITE, bg_rect)
            pygame.draw.rect(self.screen, BLACK, bg_rect, 2)
            
            self.screen.blit(paused_text, paused_rect)
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        print("=== Capture the Flag - Q-Learning AI ===")
        print("Controls:")
        print("  Arrow Keys - Move (Human vs AI mode)")
        print("  M - Switch mode (Human vs AI / AI vs AI)")
        print("  R - Reset game")
        print("  S - Save Q-tables")
        print("  Space - Pause/Resume (AI vs AI mode)")
        print("  ESC/Close - Quit")
        print("\nStarting in Human vs AI mode...")
        
        while self.running:
            # Handle events
            self.handle_events()
            
            # Update game logic
            if self.mode == MODE_HUMAN_VS_AI:
                self.update_human_vs_ai()
            else:
                self.update_ai_vs_ai()
            
            # Render
            self.render()
            
            # Control FPS
            self.clock.tick(FPS)
        
        # Save before quitting
        print("\nSaving Q-tables before exit...")
        self.save_agents()
        
        # Print statistics
        if isinstance(self.blue_agent, QLearningAgent):
            stats = self.blue_agent.get_statistics()
            print(f"\nBlue Agent Statistics:")
            print(f"  Episodes: {stats['total_episodes']}")
            print(f"  Total Rewards: {stats['total_rewards']}")
            print(f"  Q-table Size: {stats['q_table_size']}")
            print(f"  Final Epsilon: {stats['epsilon']:.3f}")
        
        stats = self.red_agent.get_statistics()
        print(f"\nRed Agent Statistics:")
        print(f"  Episodes: {stats['total_episodes']}")
        print(f"  Total Rewards: {stats['total_rewards']}")
        print(f"  Q-table Size: {stats['q_table_size']}")
        print(f"  Final Epsilon: {stats['epsilon']:.3f}")
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = CTFGame()
    game.run()
