import gymnasium as gym
import neat
import numpy as np
import pickle

def eval_genomes(genomes, config):
    env = gym.make("LunarLander-v2", render_mode="human")
    for genome_id, genome in genomes:
        observation, info = env.reset(seed=42)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0.0
        for _ in range(1000):
            action = np.argmax(net.activate(observation))
            observation, reward, terminated, truncated, info = env.step(action)
            
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

            if terminated or truncated:
                break
        
        # Set fitness threshold to terminate training
        if fitness >= 1000:
            break
        
        genome.fitness = fitness
    env.close()

def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, 50)
    
    # Save the winner
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    print('\nBest genome:\n{!s}'.format(winner))

def run_saved_model(config_file, model_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    
    with open(model_path, 'rb') as f:
        winner = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)
    fitness = 0.0
    for _ in range(1000):
        action = np.argmax(net.activate(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        fitness += reward
        if terminated or truncated:
            break
    env.close()
    
    print('Fitness of the loaded model:', fitness)

if __name__ == '__main__':
    config_path = "config-feedforward"  # Update this path
    # run_neat(config_path)
    
    # To run an existing saved model, uncomment the following line and provide the path to the saved model
    run_saved_model(config_path, 'winner.pkl')
