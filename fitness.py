import numpy as np

def fitness_function(fitness, reward, observation, terminated):
    # Adjust the fitness function
    fitness += reward
    
    # Reward based on the proximity to the landing pad
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation
    distance_penalty = np.sqrt(x**2 + y**2)
    fitness -= distance_penalty
    
    # Penalize large velocities
    velocity_penalty = np.sqrt(vx**2 + vy**2)
    fitness -= velocity_penalty
    
    # Reward for stable landing (both legs touching the ground)
    if left_leg and right_leg:
        fitness += 100
    
    # Penalize for crashing
    if terminated and reward == -100:
        fitness -= 100

    # if terminated or truncated:
    #     break
    return fitness