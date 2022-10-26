from pickletools import read_long1
from curling_simulator.config import R
from utils import get_network, get_environment, set_all_seeds
from collections import defaultdict
from mcts import MCTS, Node
from logger import Logger
from copy import deepcopy
import numpy as np
import datetime
import pytz
import time
import torch
import ray
import random
import os

c_elo=1.0/400.0
elo_k1=128.0
elo_k2=128.0

def estimate_win_probability(ra, rb) -> float:
    """Returns the estimated probability of winning from 'player A' perspective.
    Args:
        ra: elo rating for player A.
        rb: elo rating for player B.
        c_elo: the constant for elo, default 1 / 400.
    Returns:
        the win probability of player A.
    """

    return 1.0 / (1 + 10 ** ((rb - ra) * c_elo))


def compute_elo_rating(winner: int, ra=0, rb=0) -> float:
    """Returns the Elo rating from 'player A' perspective.
    Args:
        winner: who won the game, `0` for player A, `1` for player B.
        ra: current elo rating for player A, default 2500.
        rb: current elo rating for player B, default 2500.
        k: the factor, default 32.
    Returns:
        A tuple contains new estimated Elo ratings for player A and player B.
        format (elo_player_A, elo_player_B)
    """
    if winner is None:
        return (ra, rb)
    if not isinstance(winner, int) or winner not in [0, 1]:
        raise ValueError(f'Expect input argument `winner` to be [0, 1], got {winner}')

    # Compute the winning probability of player A
    prob_a = estimate_win_probability(ra, rb)

    # Compute the winning probability of player B
    prob_b = estimate_win_probability(rb, ra)

    # Updating the Elo Ratings
    k1 = elo_k1
    k2 = elo_k2
    if winner == 0:
        new_ra = ra + k1 * (1 - prob_a)
        new_rb = rb + k2 * (0 - prob_b)
        # new_rb = max(0, new_rb)
    else:
        new_ra = ra + k1 * (0 - prob_a)
        new_rb = rb + k2 * (1 - prob_b)
        # new_ra = max(0, new_ra)

    return (new_ra, new_rb)



@ray.remote(num_gpus=0.25)
class ELO_evaluator(Logger):

  def __init__(self, eval_key, config, storage, replay_buffer, state=None):
    set_all_seeds(config.seed + eval_key if config.seed is not None else None)

    self.run_tag = config.run_tag
    self.group_tag = config.group_tag
    self.config = deepcopy(config)
    self.eval_key = eval_key
    self.storage = storage
    # self.replay_buffer = replay_buffer
    # self.config.render = False

    self.environment = get_environment(self.config)
    self.environment.seed(self.config.seed)
    self.mcts = MCTS(self.config)
    
    self.ra_offen = self.config.default_score
    self.ra_defen = self.config.default_score
    self.rb_offen = self.config.default_score
    self.rb_defen = self.config.default_score

    self.record_ra_offen = self.ra_offen
    self.record_ra_defen = self.ra_defen
    self.record_rb_defen = self.rb_defen
    self.record_rb_offen = self.rb_offen
    
    if self.config.fixed_temperatures:
      self.temperature = self.config.fixed_temperatures[self.eval_key]
      self.worker_id = 'evaluators/temp={}'.format(round(self.temperature, 1))
    else:
      self.worker_id = 'evaluator-{}'.format(self.eval_key)

    if "eval" in self.config.use_gpu_for:
      if torch.cuda.is_available():
        if self.config.learner_gpu_device_ids is not None:
          device_id = self.config.learner_gpu_device_ids
          self.device = torch.device("cuda:{}".format(device_id))
        else:
          self.device = torch.device("cuda")
      else:
        raise RuntimeError("GPU was requested but torch.cuda.is_available() is False.")
    else:
      self.device = torch.device("cpu")

    self.curr_network = get_network(self.config, self.device)
    self.curr_network.to(self.device)
    self.curr_network.eval()
    self.old_network = get_network(self.config, self.device)
    self.old_network.to(self.device)
    self.old_network.eval()

    if self.config.norm_obs:
      self.obs_min = np.array(self.config.obs_range[::2], dtype=np.float32)
      self.obs_max = np.array(self.config.obs_range[1::2], dtype=np.float32)
      self.obs_range = self.obs_max - self.obs_min

    if self.config.two_players:
      self.stats_to_log = defaultdict(int)

    self.experiences_collected = 0
    self.training_step = 0
    self.games_played = 0
    self.return_to_log = 0
    self.length_to_log = 0
    self.value_to_log = {'avg': 0, 'max': 0}

    if state is not None:
      self.load_state(state)

    Logger.__init__(self)
    
  def get_value(self):
    return self.ra_offen, self.rb_offen, self.ra_defen, self.rb_defen

  def load_state(self, state):
    self.run_tag = os.path.join(self.run_tag, 'resumed', '{}'.format(state['training_step']))
    self.curr_network.load_state_dict(state['weights'])
    self.old_network.load_state_dict(state['weights'])
    self.training_step = state['training_step']

  def sync_weights(self, force=False):
    weights, training_step = ray.get(self.storage.get_elo_weights.remote())
    if training_step != self.training_step or force:
      self.curr_network.load_weights(weights)
      self.training_step = training_step

  def run_selfplay(self):

    while not ray.get(self.storage.is_ready.remote()):
      time.sleep(1)

    self.sync_weights(force=True)

    game = self.config.new_game(self.environment)
    _ = game.environment.reset()
    
    ra_offen_list = []
    rb_offen_list = []
    ra_defen_list = []
    rb_defen_list = []
    for i in range(self.config.elo_eval_game_num):
      ra_0, rb_0, ra_1, rb_1 = self.play_game(game)
      ra_offen_list.append(ra_0)
      rb_offen_list.append(rb_0)
      ra_defen_list.append(ra_1)
      rb_defen_list.append(rb_1)
    self.ra_offen = sum(ra_offen_list) / len(ra_offen_list)
    self.rb_offen = sum(rb_offen_list) / len(rb_offen_list)
    self.ra_defen = sum(ra_defen_list) / len(ra_defen_list)
    self.rb_defen = sum(rb_defen_list) / len(rb_defen_list)
    
    # if self.ra_offen > self.record_ra_offen:
    self.old_network.load_state_dict(self.curr_network.state_dict())
    self.record_ra_offen = self.ra_offen
    self.record_rb_offen = self.rb_offen
    self.record_ra_defen = self.ra_defen
    self.record_rb_defen = self.rb_defen
    
    # if game.history.rewards[-1] > 0:
    #   self.ra, self.rb = compute_elo_rating(winner=1, ra=self.ra, rb=self.rb)
    # else:
    #   self.ra, self.rb = compute_elo_rating(winner=0, ra=self.ra, rb=self.rb)
    # self.old_network.load_state_dict(self.curr_network.state_dict())
      # print('elo_A:{} ;elo_B:{}'.format(self.ra,self.rb))

    self.sync_weights(force=True)

  def play_game(self, game):
    ra_0 = 0
    ra_1 = 0
    rb_0 = 0
    rb_1 = 0
    
    if not self.config.fixed_temperatures:
      self.temperature = self.config.visit_softmax_temperature(self.training_step)

    if game.environment.curr_shot == 15:
      use_net = self.curr_network
      root = Node(0)

      current_observation = np.float32(game.get_observation(-1))
      if self.config.norm_obs:
        current_observation = (current_observation - self.obs_min) / self.obs_range
      current_observation = torch.from_numpy(current_observation).to(self.device)

      initial_inference = use_net.initial_inference(current_observation.unsqueeze(0))

      legal_actions = game.environment.legal_actions()
      root.expand(initial_inference, game.to_play, legal_actions, self.config)
      root.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

      self.mcts.run(root, use_net)

      error = root.value() - initial_inference.value.item()
      game.history.errors.append(error)

      action = self.config.select_action(root, self.temperature)

      game.apply(action)
      game.store_search_statistics(root)
      
      if game.history.rewards[-1] > 0: #player_B wins
        ra_1, rb_1 = compute_elo_rating(winner=0, ra=self.ra_defen, rb=self.rb_defen)
        # self.record_ra = self.ra
        # self.record_rb = self.rb
      else:
        ra_1, rb_1 = compute_elo_rating(winner=1, ra=self.ra_defen, rb=self.rb_defen)
        # self.old_network.load_state_dict(self.curr_network.state_dict())
        # self.record_ra = self.ra
        # self.record_rb = self.rb
    
    else:
      ra_list = []
      rb_list = []
      for i in range(2):
        _ = game.environment.reset()
        while not game.terminal:
          if game.environment.curr_player.value == i:
            # print('curr')
            use_net = self.curr_network
          else:
            # print('old')
            use_net = self.old_network
          root = Node(0)

          current_observation = np.float32(game.get_observation(-1))
          if self.config.norm_obs:
            current_observation = (current_observation - self.obs_min) / self.obs_range
          current_observation = torch.from_numpy(current_observation).to(self.device)

          initial_inference = use_net.initial_inference(current_observation.unsqueeze(0))

          legal_actions = game.environment.legal_actions()
          root.expand(initial_inference, game.to_play, legal_actions, self.config)
          # root.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

          self.mcts.run(root, use_net)

          error = root.value() - initial_inference.value.item()
          game.history.errors.append(error)

          action = self.config.select_action(root, self.temperature)

          game.apply(action)
          game.store_search_statistics(root)

          # self.experiences_collected += 1

          # if self.experiences_collected % self.config.weight_sync_frequency == 0:
          #   self.sync_weights()
          
        if i==0:#offensive
          if game.history.rewards[-1] > 0: #player_B wins
            ra, rb = compute_elo_rating(winner=1, ra=self.record_ra_defen, rb=self.record_rb_defen)
            # self.record_ra = self.ra
            # self.record_rb = self.rb
          else:
            ra, rb = compute_elo_rating(winner=0, ra=self.record_ra_defen, rb=self.record_rb_defen)
            # self.old_network.load_state_dict(self.curr_network.state_dict())
            # self.record_ra = self.ra
            # self.record_rb = self.rb
        
        else:#defensive
          if game.history.rewards[-1] > 0: #player_B wins
            ra, rb = compute_elo_rating(winner=0, ra=self.record_ra_offen, rb=self.record_rb_offen)
            # self.record_ra = self.ra
            # self.record_rb = self.rb
          else:
            ra, rb = compute_elo_rating(winner=1, ra=self.record_ra_offen, rb=self.record_rb_offen)
            # self.old_network.load_state_dict(self.curr_network.state_dict())
            # self.record_ra = self.ra
            # self.record_rb = self.rb
        
        if i==0:
          ra_0, rb_0 = ra, rb
        else:
          ra_1, rb_1 = ra, rb
        # ra_list.append(ra)
        # rb_list.append(rb) 

      # ra = sum(ra_list) / len(ra_list)
      # rb = sum(rb_list) / len(rb_list)
        
    if self.config.two_players:
      self.stats_to_log[game.info["result"]] += 1
      
    return ra_0, rb_0, ra_1, rb_1

  def launch(self):
    with torch.inference_mode():
      self.run_selfplay()

