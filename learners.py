from utils import get_network, get_optimizer, get_lr_scheduler, get_loss_functions, set_all_seeds, abstract_loss_func
from mcts import MCTS, Node
from logger import Logger
from copy import deepcopy
import numpy as np
import datetime
import torch
import time
import pytz
import ray
import os


@ray.remote
class Learner(Logger):

  def __init__(self, config, storage, replay_buffer, state=None):
    set_all_seeds(config.seed)

    self.run_tag = config.run_tag
    self.group_tag = config.group_tag
    self.worker_id = 'learner'
    self.replay_buffer = replay_buffer
    self.storage = storage
    self.config = deepcopy(config)

    if "learner" in self.config.use_gpu_for:
      if torch.cuda.is_available():
        if self.config.learner_gpu_device_id is not None:
          device_id = self.config.learner_gpu_device_id
          self.device = torch.device("cuda:{}".format(device_id))
        else:
          self.device = torch.device("cuda")
      else:
        raise RuntimeError("GPU was requested but torch.cuda.is_available() is False.")
    else:
      self.device = torch.device("cpu")
    
    self.network = get_network(config, self.device)
    self.network.to(self.device)
    self.network.train()

    self.optimizer = get_optimizer(config, self.network.parameters())
    self.lr_scheduler = get_lr_scheduler(config, self.optimizer)
    self.scalar_loss_fn, self.policy_loss_fn = get_loss_functions(config)
    self.abstract_loss_fn = abstract_loss_func()

    self.training_step = 0
    self.losses_to_log = {'reward': 0., 'value': 0., 'policy': 0., 'abstract': 0., 'aggregation_times': 0.}

    self.throughput = {'total_frames': 0, 'total_games': 0, 'training_step': 0, 'time': {'ups': 0, 'fps': 0}}

    if self.config.norm_obs:
      self.obs_min = np.array(self.config.obs_range[::2], dtype=np.float32)
      self.obs_max = np.array(self.config.obs_range[1::2], dtype=np.float32)
      self.obs_range = self.obs_max - self.obs_min

    if state is not None:
      self.load_state(state)

    Logger.__init__(self)

  def load_state(self, state):
    self.run_tag = os.path.join(self.run_tag, 'resumed', '{}'.format(state['training_step']))
    self.network.load_state_dict(state['weights'])
    self.optimizer.load_state_dict(state['optimizer'])

    self.replay_buffer.add_initial_throughput.remote(state['total_frames'], state['total_games'])
    self.throughput['total_frames'] = state['total_frames']
    self.throughput['training_step'] = state['training_step']
    self.training_step = state['training_step'] 

  def save_state(self):
    actor_games = ray.get(self.storage.get_stats.remote('actor_games'))
    state = {'dirs': self.dirs,
             'config': self.config,
    		 		 'weights': self.network.get_weights(),
    		     'optimizer': self.optimizer.state_dict(),
             'training_step': self.training_step,
             'total_games': self.throughput['total_games'],
             'total_frames': self.throughput['total_frames'],
             'actor_games': actor_games}
    path = os.path.join(self.dirs['saves'], str(state['training_step']))
    torch.save(state, path)

  def send_weights(self):
    self.storage.store_weights.remote(self.network.get_weights(), self.training_step)

  def log_throughput(self):
    data = ray.get(self.replay_buffer.get_throughput.remote())

    self.throughput['total_games'] = data['games']
    self.log_scalar(tag='games/finished', value=data['games'], i=self.training_step)

    new_frames = data['frames'] - self.throughput['total_frames']
    if new_frames > self.config.frames_before_fps_log:

      current_time = time.time()
      new_updates = self.training_step - self.throughput['training_step']
      ups = new_updates / (current_time - self.throughput['time']['ups'])
      fps = new_frames / (current_time - self.throughput['time']['fps'])
      replay_ratio = ups / fps
      sample_ratio = self.config.batch_size * replay_ratio

      self.throughput['total_frames'] = data['frames']
      self.throughput['training_step'] = self.training_step
      self.throughput['time']['ups'] = current_time
      self.throughput['time']['fps'] = current_time

      self.log_scalar(tag='throughput/frames_per_second', value=fps, i=self.training_step)
      self.log_scalar(tag='throughput/updates_per_second', value=ups, i=self.training_step)
      self.log_scalar(tag='throughput/replay_ratio', value=replay_ratio, i=self.training_step)
      self.log_scalar(tag='throughput/sample_ratio', value=sample_ratio, i=self.training_step)
      self.log_scalar(tag='throughput/total_frames', value=data['frames'], i=self.training_step)

  def learn(self):
    self.send_weights()

    self.throughput['time']['fps'] = time.time() 
    while ray.get(self.replay_buffer.size.remote()) < self.config.stored_before_train:
      time.sleep(1)

    self.throughput['time']['ups'] = time.time() 
    while self.training_step < self.config.training_steps:
      not_ready_batches = [self.replay_buffer.sample_batch.remote() for _ in range(self.config.batches_per_fetch)]
      while len(not_ready_batches) > 0:
        ready_batches, not_ready_batches = ray.wait(not_ready_batches, num_returns=1)

        batch = ray.get(ready_batches[0])
        self.update_weights(batch)
        self.training_step += 1

        if self.training_step % self.config.send_weights_frequency == 0:
          self.send_weights()

        if self.training_step % self.config.save_state_frequency == 0:
          self.save_state()

        if self.training_step % self.config.learner_log_frequency == 0:
          reward_loss = self.losses_to_log['reward'] / self.config.learner_log_frequency
          value_loss = self.losses_to_log['value'] / self.config.learner_log_frequency
          policy_loss = self.losses_to_log['policy'] / self.config.learner_log_frequency
          abstract_loss = self.losses_to_log['abstract'] / self.config.learner_log_frequency
          aggregation_times = self.losses_to_log['aggregation_times'] / self.config.learner_log_frequency

          self.losses_to_log['reward'] = 0
          self.losses_to_log['value'] = 0
          self.losses_to_log['policy'] = 0
          self.losses_to_log['abstract_loss'] = 0
          self.losses_to_log['aggregation_times'] = 0

          self.log_scalar(tag='loss/reward', value=reward_loss, i=self.training_step)
          self.log_scalar(tag='loss/value', value=value_loss, i=self.training_step)
          self.log_scalar(tag='loss/policy', value=policy_loss, i=self.training_step)
          self.log_scalar(tag='loss/abstract_loss', value=abstract_loss, i=self.training_step)
          self.log_scalar(tag='loss/aggregation_times', value=aggregation_times, i=self.training_step)
          self.log_throughput()

          if self.lr_scheduler is not None:
            self.log_scalar(tag='loss/learning_rate', value=self.lr_scheduler.lr, i=self.training_step)

          if self.config.debug:
            total_grad_norm = 0
            for name, weights in self.network.named_parameters():
              self.log_histogram(weights.grad.data.cpu().numpy(), 'gradients' + '/' + name + '_grad', self.training_step)
              self.log_histogram(weights.data.cpu().numpy(), 'network_weights' + '/' + name, self.training_step)
              total_grad_norm += weights.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            self.log_scalar(tag='total_gradient_norm', value=total_grad_norm, i=self.training_step)

  def update_weights(self, batch):
    batch, idxs, is_weights = batch
    observations, actions, targets,  aggregation_times_batch = batch

    target_rewards, target_values, target_policies = targets

    if self.config.norm_obs:
      observations = (observations - self.obs_min) / self.obs_range
    observations = torch.tensor(observations).to(self.device)

    value, reward, policy_logits, hidden_state = self.network.initial_inference(observations)

    with torch.no_grad():
      target_policies = torch.tensor(target_policies).to(self.device)
      target_values = torch.tensor(target_values).to(self.device)
      target_rewards = torch.tensor(target_rewards).to(self.device)
      is_weights = torch.tensor(is_weights).to(self.device)

      init_value = self.config.inverse_value_transform(value) if not self.config.no_support else value
      new_errors = (init_value.squeeze() - target_values[:, 0]).cpu().numpy()
      self.replay_buffer.update.remote(idxs, new_errors)

      if not self.config.no_target_transform:
        target_values = self.config.scalar_transform(target_values)
        target_rewards = self.config.scalar_transform(target_rewards)

      if not self.config.no_support:
        target_values = self.config.value_phi(target_values)
        target_rewards = self.config.reward_phi(target_rewards)

    reward_loss = 0
    value_loss = self.scalar_loss_fn(value.squeeze(), target_values[:, 0])
    policy_loss = self.policy_loss_fn(policy_logits.squeeze(), target_policies[:, 0])
    abstract_loss = 0
    abstract_v_loss = 0

    aggregation_times = 0

    for k in aggregation_times_batch:
      aggregation_times += np.mean(k)
    aggregation_times /= len(aggregation_times_batch)
    if self.config.num_sample_action == 0:
      step_error = self.config.max_transitive_error / self.config.action_space
    else:
      step_error = self.config.max_transitive_error / self.config.num_sample_action

    if step_error > 0:
      Abstract_node = {}

      action_space = np.array(range(self.config.action_space))

      node_aggregation_times = 0


      for i in range(len(action_space)):
        action = action_space[i]

        next_hidden_state, _ = self.network.dynamics(hidden_state[0].unsqueeze(0),[action])

        policy, predict_V = self.network.prediction(next_hidden_state)

        policy = torch.softmax(
      torch.tensor([policy[0][a].item() for a in action_space]), dim=0
    ).numpy().astype('float64')

        predict_V = self.config.inverse_value_transform(predict_V)

        predict_V = predict_V.item()

        Abstract_node[action] = [next_hidden_state,  predict_V, np.argmax(policy)]

      sorted_Abstract_node = sorted(Abstract_node.items(), key=lambda x: x[1][1])

      for k in range(len(sorted_Abstract_node) - 1):
        a1, v1, abstract_r1 = sorted_Abstract_node[k][1][2], sorted_Abstract_node[k][1][1], sorted_Abstract_node[k][1][0]
        a2, v2, abstract_r2 = sorted_Abstract_node[k + 1][1][2], sorted_Abstract_node[k + 1][1][1], sorted_Abstract_node[k + 1][1][0]
        if abs(v2 - v1) < (step_error * (abs(v1) + abs(v2)) / 2) and a1 == a2:
          node_aggregation_times += 1
          abstract_loss += self.abstract_loss_fn(abstract_r1, abstract_r2)

      if node_aggregation_times > 0:

        abstract_loss /= torch.tensor(node_aggregation_times).to(self.device)

    for i, action in enumerate(zip(*actions), 1):
      next_hidden_state, reward = self.network.dynamics(hidden_state, action)
      abstract_representation, predict_V = self.network.abstract_embed(next_hidden_state)
      policy_logits, value = self.network.prediction(abstract_representation)
      next_hidden_state.register_hook(lambda grad: grad * 0.5)

      reward_loss += self.scalar_loss_fn(reward.squeeze(), target_rewards[:, i])

      value_loss += self.scalar_loss_fn(value.squeeze(), target_values[:, i])
      
      policy_loss += self.policy_loss_fn(policy_logits.squeeze(), target_policies[:, i])

      abstract_v_loss += self.scalar_loss_fn(predict_V.squeeze(), target_values[:, i])


    reward_loss = (is_weights * reward_loss).mean()
    value_loss = (is_weights * value_loss).mean()
    policy_loss = (is_weights * policy_loss).mean()
    abstract_loss = (is_weights * abstract_loss).mean()
    abstract_v_loss = (is_weights * abstract_v_loss).mean()

    full_weighted_loss = reward_loss + value_loss + policy_loss + abstract_loss * self.config.abstract_loss_weight + abstract_v_loss


    full_weighted_loss.register_hook(lambda grad: grad * (1/self.config.num_unroll_steps))

    self.optimizer.zero_grad()

    full_weighted_loss.backward()

    if self.config.clip_grad:
      torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.clip_grad)

    self.optimizer.step()

    if self.lr_scheduler is not None:
      self.lr_scheduler.step()

    value_loss += abstract_v_loss
    self.losses_to_log['reward'] += reward_loss.detach().cpu().item()
    self.losses_to_log['value'] += value_loss.detach().cpu().item()
    self.losses_to_log['policy'] += policy_loss.detach().cpu().item()
    self.losses_to_log['abstract'] += abstract_loss.detach().cpu().item()
    self.losses_to_log['aggregation_times'] += aggregation_times

  def launch(self):
    print("Learner is online on {}.".format(self.device))
    self.learn()

