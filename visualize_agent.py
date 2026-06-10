import gymnasium as gym
import numpy as np
import time
from frozen_lake_agent import train_agent

def watch_trained_agent():
    print("Training agent in the background...")
    # Get the student's trained Q-table
    q_table = train_agent()
    print("Training complete! Opening the visualizer...")

    # Initialize environment with human render mode
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
    state, _ = env.reset()
    done = False

    # Add a slight delay before starting so the human can see the window
    time.sleep(1)

    step_count = 0
    max_steps = 100 # Stop after 100 moves

    while not done and step_count < max_steps:
        # Tie-breaking action selection
        max_value = np.max(q_table[state])
        best_actions = np.where(q_table[state] == max_value)[0]
        action = np.random.choice(best_actions)
        
        # Take the step
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        step_count += 1
        time.sleep(0.5)

    if step_count >= max_steps:
        print("The Elf wandered aimlessly and got lost.")

    if reward == 1.0:
        print("The Elf reached the treasure!")
    else:
        print("The Elf fell in a hole.")
        
    # Keep the window open for a second before closing
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    watch_trained_agent()
