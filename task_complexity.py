import torch
import numpy as np
import scipy.stats as stats

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache() 

def compute_complexity(untrained_model_architecture, n_instances, batched_data, loss_function, n_bins, device = "cuda", trained = False, autoencoder = False, network_architecture_parameters = []):
    
    conditional_rewards = []
    for instance in range(n_instances):
        # Instantiate the conditional model
        if trained:
            model = untrained_model_architecture
        else:
            model = untrained_model_architecture(*network_architecture_parameters).to(device)

        # Instantiate the binned local (this conditional model) rewards/ Each sample in `local_rewards` os r_{ij} ~ p(r|theta_i)
        local_rewards = []
        for batch in batched_data:
            samples, labels = batch
            samples, labels = samples.to(device), labels.to(device)

            if not autoencoder:
                labels = labels.double()
            
            with torch.no_grad():
                scores = model(samples)

            if autoencoder:
                reward = loss_function(scores, samples).detach().cpu().numpy()
            else:
                reward = loss_function(scores, labels).detach().cpu().numpy()
                
            if reward.ndim == 0:
                reward = np.expand_dims(reward, axis = 0)
            local_rewards.extend(reward)

        # Append the local rewards to the conditional list
        conditional_rewards.append(local_rewards)
        

    # Collect all rewards and form their distribution
    all_rewards = sum(conditional_rewards, [])
    min_reward, max_reward = np.min(all_rewards), np.max(all_rewards)
    bins = np.linspace(min_reward, max_reward, n_bins + 1)
    rewards_distribution = np.histogram(all_rewards, bins)[0]/len(all_rewards)

    # Compute the rewards entropy
    rewards_entropy = stats.entropy(rewards_distribution)

    # Approximate p(r|theta_i) with the conditional rewards
    conditional_reward_distributions = [np.histogram(reward, bins)[0]/len(reward) for reward in conditional_rewards]

    # Compute the average over the conditionals
    average_conditional_rewards_entropy = np.mean([stats.entropy(dist) for dist in conditional_reward_distributions])

    # Compute the final score
    complexity_score = rewards_entropy - average_conditional_rewards_entropy

    return complexity_score