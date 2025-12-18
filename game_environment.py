"""
Game Environment for Capture the Flag
Handles game state, rendering, and game logic
"""

import pygame
import random
from config import *


class CTFEnvironment:
    def __init__(self):
        """Initialize the game environment"""
        self.grid_rows = GRID_ROWS
        self.grid_cols = GRID_COLS
        self.cell_size = CELL_SIZE
        
        # Initialize positions
        self.reset()
        
        # Create obstacles
        self.obstacles = self._create_obstacles()
        
    def _create_obstacles(self):
        """Create obstacles on the map"""
        obstacles = set()
        
        # Add border walls
        for i in range(self.grid_cols):
            obstacles.add((0, i))
            obstacles.add((self.grid_rows - 1, i))
        for i in range(self.grid_rows):
            obstacles.add((i, 0))
            obstacles.add((i, self.grid_cols - 1))
        
        # Add some random obstacles in the middle
        middle_obstacles = [
            (3, 5), (3, 6), (3, 7),
            (8, 5), (8, 6), (8, 7),
            (5, 3), (6, 3), (7, 3),
            (5, 12), (6, 12), (7, 12),
            (5, 8), (6, 8)
        ]
        
        for obs in middle_obstacles:
            if 0 < obs[0] < self.grid_rows - 1 and 0 < obs[1] < self.grid_cols - 1:
                obstacles.add(obs)
        
        return obstacles
    
    def reset(self):
        """Reset the game to initial state"""
        # Blue team (left side) - Player/AI
        self.blue_base = (2, 2)
        self.blue_flag_home = (2, 2)
        self.blue_agent = (2, 2)
        self.blue_flag_pos = (2, 2)
        self.blue_has_flag = False
        
        # Red team (right side) - AI
        self.red_base = (self.grid_rows - 3, self.grid_cols - 3)
        self.red_flag_home = (self.grid_rows - 3, self.grid_cols - 3)
        self.red_agent = (self.grid_rows - 3, self.grid_cols - 3)
        self.red_flag_pos = (self.grid_rows - 3, self.grid_cols - 3)
        self.red_has_flag = False
        
        # Game state
        self.game_over = False
        self.winner = None
        self.blue_score = 0
        self.red_score = 0
        
        return self.get_state('blue'), self.get_state('red')
    
    def get_state(self, team):
        """Get the current state for a team"""
        if team == 'blue':
            agent_pos = self.blue_agent
            enemy_pos = self.red_agent
            target_flag = self.red_flag_pos
            own_flag = self.blue_flag_pos
            has_flag = self.blue_has_flag
        else:
            agent_pos = self.red_agent
            enemy_pos = self.blue_agent
            target_flag = self.blue_flag_pos
            own_flag = self.red_flag_pos
            has_flag = self.red_has_flag
        
        # Return a tuple representing the state
        return (
            agent_pos[0], agent_pos[1],
            enemy_pos[0], enemy_pos[1],
            target_flag[0], target_flag[1],
            own_flag[0], own_flag[1],
            int(has_flag)
        )
    
    def is_valid_position(self, pos):
        """Check if a position is valid (not obstacle)"""
        return pos not in self.obstacles
    
    def move_agent(self, team, action):
        """
        Move an agent based on action
        Actions: 0=up, 1=down, 2=left, 3=right
        Returns: new_state, reward, done
        """
        if team == 'blue':
            current_pos = self.blue_agent
        else:
            current_pos = self.red_agent
        
        # Calculate new position
        new_pos = current_pos
        if action == 0:  # Up
            new_pos = (current_pos[0] - 1, current_pos[1])
        elif action == 1:  # Down
            new_pos = (current_pos[0] + 1, current_pos[1])
        elif action == 2:  # Left
            new_pos = (current_pos[0], current_pos[1] - 1)
        elif action == 3:  # Right
            new_pos = (current_pos[0], current_pos[1] + 1)
        
        # Check if move is valid
        if not self.is_valid_position(new_pos):
            # Invalid move, stay in place
            return self.get_state(team), REWARD_STEP, False
        
        # Update agent position
        if team == 'blue':
            self.blue_agent = new_pos
        else:
            self.red_agent = new_pos
        
        # Calculate reward and check game conditions
        reward = REWARD_STEP
        done = False
        
        # Check if agent captures enemy flag
        if team == 'blue':
            # Check if blue agent reached red flag
            if self.blue_agent == self.red_flag_pos and not self.blue_has_flag:
                self.blue_has_flag = True
                self.red_flag_pos = self.blue_agent  # Flag moves with agent
                reward += REWARD_MOVE_TOWARD_FLAG
            
            # If carrying flag, update flag position
            if self.blue_has_flag:
                self.red_flag_pos = self.blue_agent
                
                # Check if returned to base with flag
                if self.blue_agent == self.blue_base:
                    reward += REWARD_CAPTURE_FLAG
                    self.blue_score += 1
                    done = True
                    self.game_over = True
                    self.winner = 'blue'
            
            # Check if caught by red agent
            if self.blue_agent == self.red_agent:
                if self.blue_has_flag:
                    # Blue caught with red flag
                    reward += REWARD_GET_CAUGHT
                    self.blue_has_flag = False
                    self.red_flag_pos = self.red_flag_home
                if self.red_has_flag:
                    # Blue caught red with blue flag
                    reward += REWARD_DEFEND_FLAG
                    self.red_has_flag = False
                    self.blue_flag_pos = self.blue_flag_home
        
        else:  # Red team
            # Check if red agent reached blue flag
            if self.red_agent == self.blue_flag_pos and not self.red_has_flag:
                self.red_has_flag = True
                self.blue_flag_pos = self.red_agent
                reward += REWARD_MOVE_TOWARD_FLAG
            
            # If carrying flag, update flag position
            if self.red_has_flag:
                self.blue_flag_pos = self.red_agent
                
                # Check if returned to base with flag
                if self.red_agent == self.red_base:
                    reward += REWARD_CAPTURE_FLAG
                    self.red_score += 1
                    done = True
                    self.game_over = True
                    self.winner = 'red'
            
            # Check if caught by blue agent
            if self.red_agent == self.blue_agent:
                if self.red_has_flag:
                    # Red caught with blue flag
                    reward += REWARD_GET_CAUGHT
                    self.red_has_flag = False
                    self.blue_flag_pos = self.blue_flag_home
                if self.blue_has_flag:
                    # Red caught blue with red flag
                    reward += REWARD_DEFEND_FLAG
                    self.blue_has_flag = False
                    self.red_flag_pos = self.red_flag_home
        
        return self.get_state(team), reward, done
    
    def render(self, screen):
        """Render the game state using Pygame"""
        screen.fill(WHITE)
        
        # Draw grid
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(screen, BLACK, rect, 1)
        
        # Draw bases
        blue_base_rect = pygame.Rect(
            self.blue_base[1] * self.cell_size,
            self.blue_base[0] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(screen, LIGHT_BLUE, blue_base_rect)
        
        red_base_rect = pygame.Rect(
            self.red_base[1] * self.cell_size,
            self.red_base[0] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(screen, LIGHT_RED, red_base_rect)
        
        # Draw obstacles
        for obs in self.obstacles:
            obs_rect = pygame.Rect(
                obs[1] * self.cell_size,
                obs[0] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(screen, DARK_GRAY, obs_rect)
        
        # Draw flags
        if not self.blue_has_flag:
            blue_flag_rect = pygame.Rect(
                self.blue_flag_pos[1] * self.cell_size + self.cell_size // 4,
                self.blue_flag_pos[0] * self.cell_size + self.cell_size // 4,
                self.cell_size // 2,
                self.cell_size // 2
            )
            pygame.draw.rect(screen, BLUE, blue_flag_rect)
        
        if not self.red_has_flag:
            red_flag_rect = pygame.Rect(
                self.red_flag_pos[1] * self.cell_size + self.cell_size // 4,
                self.red_flag_pos[0] * self.cell_size + self.cell_size // 4,
                self.cell_size // 2,
                self.cell_size // 2
            )
            pygame.draw.rect(screen, RED, red_flag_rect)
        
        # Draw agents
        blue_agent_center = (
            self.blue_agent[1] * self.cell_size + self.cell_size // 2,
            self.blue_agent[0] * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(screen, BLUE, blue_agent_center, self.cell_size // 3)
        
        red_agent_center = (
            self.red_agent[1] * self.cell_size + self.cell_size // 2,
            self.red_agent[0] * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(screen, RED, red_agent_center, self.cell_size // 3)
        
        # Draw indicator if agent has flag
        if self.blue_has_flag:
            pygame.draw.circle(screen, YELLOW, blue_agent_center, self.cell_size // 6)
        if self.red_has_flag:
            pygame.draw.circle(screen, YELLOW, red_agent_center, self.cell_size // 6)
    
    def render_ui(self, screen, font, mode, episode=None):
        """Render UI elements (score, mode, etc.)"""
        ui_y = self.grid_rows * self.cell_size + 10
        
        # Mode text
        mode_text = "Mode: Human vs AI" if mode == MODE_HUMAN_VS_AI else "Mode: AI vs AI Training"
        mode_surface = font.render(mode_text, True, BLACK)
        screen.blit(mode_surface, (10, ui_y))
        
        # Score
        score_text = f"Blue: {self.blue_score}  Red: {self.red_score}"
        score_surface = font.render(score_text, True, BLACK)
        screen.blit(score_surface, (10, ui_y + 30))
        
        # Episode (for AI vs AI mode)
        if episode is not None:
            episode_text = f"Episode: {episode}"
            episode_surface = font.render(episode_text, True, BLACK)
            screen.blit(episode_surface, (10, ui_y + 60))
        
        # Instructions
        if mode == MODE_HUMAN_VS_AI:
            inst_text = "Arrow Keys to move | M: Switch Mode | R: Reset"
        else:
            inst_text = "M: Switch Mode | R: Reset | Space: Pause/Resume"
        inst_surface = font.render(inst_text, True, GRAY)
        screen.blit(inst_surface, (10, ui_y + 90))
        
        # Winner
        if self.game_over and self.winner:
            winner_text = f"{self.winner.upper()} WINS!"
            winner_surface = font.render(winner_text, True, GREEN)
            winner_rect = winner_surface.get_rect(center=(WINDOW_WIDTH // 2, ui_y + 60))
            screen.blit(winner_surface, winner_rect)
