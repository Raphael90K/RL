import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame

# Anpassen der Grid-Umgebung mit Pygame
class CoinMazeEnv(gym.Env):
    def __init__(self, grid_size=5, num_coins=3, cell_size=50):
        super(CoinMazeEnv, self).__init__()

        # Pygame-Initialisierung
        pygame.init()
        self.screen_size = grid_size * cell_size
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Coin Maze")

        # Setze das Grid (z.B. 5x5)
        self.grid_size = grid_size
        self.num_coins = num_coins

        # Definiere die Aktionsräume (4 Richtungen: oben, unten, links, rechts)
        self.action_space = spaces.Discrete(4)

        # Definiere den Beobachtungsraum: Ein 2D-Gitter
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.uint8)

        # Initialisieren des Grids, Positionen und der Coins
        self.reset()

    def reset(self):
        # Setze das Gitter zurück und platziere den Agenten, Coins und den Ausgang
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.agent_pos = [0, 0]
        self.exit_pos = [self.grid_size-1, self.grid_size-1]
        self.coins = []

        # Setze Coins an zufällige Positionen
        for _ in range(self.num_coins):
            coin_pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            while coin_pos == self.agent_pos or coin_pos == self.exit_pos or coin_pos in self.coins:
                coin_pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            self.coins.append(coin_pos)
            self.grid[coin_pos[0], coin_pos[1]] = 1  # Markiere Position als Coin

        self.steps_taken = 0
        self.total_reward = 0
        self.render()  # Initiale Anzeige
        return self.grid

    def step(self, action):
        # Definiere Bewegungen: Oben, Unten, Links, Rechts
        if action == 0:  # Oben
            if self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
        elif action == 1:  # Unten
            if self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos[0] += 1
        elif action == 2:  # Links
            if self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
        elif action == 3:  # Rechts
            if self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[1] += 1

        # Check, ob der Agent auf einem Coin ist
        reward = 0
        if self.agent_pos in self.coins:
            reward = 1
            self.coins.remove(self.agent_pos)  # Coin einsammeln
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 0  # Coin aus dem Grid entfernen
            self.total_reward += reward

        # Check, ob der Agent den Ausgang erreicht hat
        done = False
        if self.agent_pos == self.exit_pos:
            reward = 10  # Bonus für den Ausgang
            done = True
            self.total_reward += reward

        # Anzahl der Schritte erhöhen
        self.steps_taken += 1

        # Gebe die Umgebung zurück und rendere das Grid
        self.render()
        return self.grid, reward, done, {}

    def render(self):
        # Bildschirm hintergrund
        self.screen.fill((255, 255, 255))

        # Zeichne das Gitter
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                if [i, j] == self.exit_pos:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)  # Ausgang in grün
                elif [i, j] == self.agent_pos:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Agent in blau
                elif [i, j] in self.coins:
                    pygame.draw.rect(self.screen, (255, 223, 0), rect)  # Coin in gelb
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)  # Gitterlinien

        # Text für Anzahl der Schritte und Belohnung
        font = pygame.font.SysFont("Arial", 18)
        step_text = font.render(f"Steps: {self.steps_taken}", True, (0, 0, 0))
        reward_text = font.render(f"Total Reward: {self.total_reward}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, self.screen_size - 40))
        self.screen.blit(reward_text, (10, self.screen_size - 20))

        # Bildschirm aktualisieren
        pygame.display.flip()

    def close(self):
        pygame.quit()

