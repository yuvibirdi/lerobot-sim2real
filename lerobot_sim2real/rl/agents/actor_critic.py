from dataclasses import dataclass
from typing import List
from torch.distributions.normal import Normal
import torch
import numpy as np
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# note this is a very basic config setup more for example use than very deep RL research
@dataclass
class ActorCriticConfig:
    actor_features: int = 512
    critic_features: int = 512

class ActorCritic(nn.Module):
    def __init__(self, envs, sample_obs, feature_net: nn.Module, config: ActorCriticConfig = ActorCriticConfig()):
        super().__init__()
        self.feature_net = feature_net
        with torch.no_grad():
            latent_size = self.feature_net(sample_obs).shape[1]
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, config.critic_features)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(config.critic_features, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, config.actor_features)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(config.actor_features, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)
    def get_features(self, x):
        return self.feature_net(x)
    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)