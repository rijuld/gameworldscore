import torch
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from oasis_verl.workers import OasisWorker, OasisCriticWorker, OasisRewardWorker
from verl import DataProto
import ray

class OasisRayPPOTrainer(RayPPOTrainer):
    def __init__(self, config):
        # Skip super().__init__ because it initializes standard workers.
        # We will manually initialize our workers.
        self.config = config
        self._init_workers()

    def _init_workers(self):
        # Initialize OasisWorker (Actor + Rollout)
        self.actor_rollout_wg = ray.remote(OasisWorker).remote(self.config, role='actor_rollout')
        
        # Initialize OasisCriticWorker
        self.critic_wg = ray.remote(OasisCriticWorker).remote(self.config)
        
        # Initialize OasisRewardWorker
        self.rm_wg = ray.remote(OasisRewardWorker).remote(self.config)

    def fit(self):
        # Custom training loop
        print("Starting Oasis PPO Training...")
        
        for epoch in range(self.config.trainer.epochs):
            # 1. Generate Rollouts
            # We pass a dummy DataProto to trigger generation
            dummy_prompt = DataProto.from_dict({'dummy': torch.zeros(self.config.trainer.batch_size, 1)})
            rollout_data = ray.get(self.actor_rollout_wg.generate_sequences.remote(dummy_prompt))
            
            # 2. Compute Rewards (using RewardWorker)
            reward_data = ray.get(self.rm_wg.compute_rm_score.remote(rollout_data))
            rollout_data.batch['rewards'] = reward_data.batch['rewards']
            
            # 3. Compute Values
            # We need to pass the states from rollout to critic
            # rollout_data contains 'states', 'rewards'
            values_data = ray.get(self.critic_wg.compute_values.remote(rollout_data))
            
            # 4. Compute Advantages (GAE)
            # We can do this on the driver or inside a worker. Let's do it here for simplicity.
            rewards = rollout_data.batch['rewards']
            values = values_data.batch['values']
            
            advantages, returns = self.compute_gae(rewards, values)
            
            # Add advantages and returns to data
            rollout_data.batch['advantages'] = advantages
            rollout_data.batch['returns'] = returns
            
            # 5. Update Actor
            actor_metrics = ray.get(self.actor_rollout_wg.update_actor.remote(rollout_data))
            
            # 6. Update Critic
            critic_metrics = ray.get(self.critic_wg.update_critic.remote(rollout_data))
            
            print(f"Epoch {epoch}: Actor Loss: {actor_metrics.batch['metrics']['actor_loss']}, Critic Loss: {critic_metrics.batch['metrics']['critic_loss']}")

    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        # Simple GAE implementation
        # rewards: (B, T)
        # values: (B, T)
        
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(rewards.shape[1])):
            next_val = values[:, t+1] if t+1 < values.shape[1] else 0.0
            delta = rewards[:, t] + gamma * next_val - values[:, t]
            last_gae_lam = delta + gamma * lam * last_gae_lam
            advantages[:, t] = last_gae_lam
            
        returns = advantages + values
        return advantages, returns
