import gymnasium as gym
import numpy as np
import sys
from frozen_lake_agent import train_agent


def test_frozen_lake():
    print("Running Autograder: Best out of 5 Attempts...")
    
    max_attempts = 5
    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt} of {max_attempts} ---")
        
        # 1. Train a fresh agent
        try:
            q_table = train_agent()
        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)
            
        # 2. Test this specific agent
        state, _ = env.reset()
        reward = 0
        
        for step in range(20):
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
                
        # 3. Evaluate the attempt
        if reward == 1.0:
            print("Success! The agent navigated the ice perfectly.")
            sys.exit(0) # Exit immediately with a PASS
        else:
            print("Agent failed on this attempt.")
            
    # If the loop finishes without hitting sys.exit(0), all 5 attempts failed
    print("\nAll 5 attempts failed. Check your Bellman equation or exploration logic.")
    sys.exit(1)

if __name__ == "__main__":
    test_frozen_lake()
