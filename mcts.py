import datetime

import numpy as np
import torch
import math


class MinMaxStats(object):

  def __init__(self, minimum_bound=None, maximum_bound=None):
    self.minimum = float('inf')  if minimum_bound is None else minimum_bound
    self.maximum = -float('inf') if maximum_bound is None else maximum_bound

  def update(self, value: float):
    self.minimum = min(self.minimum, value)
    self.maximum = max(self.maximum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      return (value - self.minimum) / (self.maximum - self.minimum)
    elif self.maximum == self.minimum:
      return 1.0
    return value

  def reset(self, minimum_bound=None, maximum_bound=None):
    self.minimum = float('inf')  if minimum_bound is None else minimum_bound
    self.maximum = -float('inf') if maximum_bound is None else maximum_bound


class Node(object):

  def __init__(self, prior):
    self.hidden_state = None
    self.visit_count = 0
    self.value_sum = 0
    self.reward = 0
    self.children = {}
    self.prior = prior
    self.to_play = 1
    self.abstract_V = 0
    self.aggregation_times = 0
    self.abstract_loss = 0
    self.Q = 0

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

  def expand(self, hidden_state, policy_logits, to_play, actions, config):
    self.to_play = to_play
    self.hidden_state = hidden_state

    actions = np.array(actions)

    sample_num = config.num_sample_action

    policy_values = torch.softmax(
      torch.tensor([policy_logits[0][a].item() for a in actions]), dim=0
    ).numpy().astype('float64')

    policy_values /= policy_values.sum()

    if sample_num > 0:

      regret = config.max_r * np.ones(shape=actions.shape) - self.reward * policy_values
      v_a = ((actions - (actions * policy_values) ** 2) ** 2) * policy_values
      uniform_policy = np.ones(actions.shape) / len(actions)

      best_alpha, best_dis = self.golden_selection(regret, v_a, uniform_policy, policy_values)


      if len(actions) > sample_num:
        sample_action = np.random.choice(actions, size=sample_num, replace=False, p=best_dis)
      else:
        sample_action = actions

      sample_policy_values = torch.softmax(
        torch.tensor([policy_logits[0][a] for a in sample_action]), dim=0
      ).numpy().astype('float64')

      for i in range(len(sample_action)):
        action = sample_action[i]
        p = sample_policy_values[i]
        self.children[action] = Node(p)


    else:
      policy = {a: policy_values[i] for i, a in enumerate(actions)}

      for action, p in policy.items():
        self.children[action] = Node(p)


  def add_exploration_noise(self, dirichlet_alpha, frac):
    actions = list(self.children.keys())
    noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
    for a, n in zip(actions, noise):
      self.children[a].prior = self.children[a].prior*(1-frac) + n*frac

  def golden_selection(self, regret, v_a, uniform_policy, policy_values):
    R = 0.618033989
    C = 1.0 - R
    a = 0
    b = 1
    # First telescoping
    x1 = R * a + C * b
    x2 = C * a + R * b

    policy1 = x1 * uniform_policy + (1 - x1) * policy_values
    policy2 = x2 * uniform_policy + (1 - x2) * policy_values

    ratio1 = (np.dot(policy1, regret) ** 2 / (np.dot(policy1, v_a)))
    ratio2 = (np.dot(policy2, regret) ** 2 / (np.dot(policy2, v_a)))

    e = 1e-5
    # Main loop

    while b - a > e:
      if ratio2 > ratio1:
        a = x1
        x1 = x2
        ratio1 = ratio2
        x2 = a + C * (b - a)
        policy2 = x2 * uniform_policy + (1 - x2) * policy_values
        ratio2 = (np.dot(policy2, regret) ** 2 / (np.dot(policy2, v_a)))
      else:
        b = x2
        x2 = x1
        ratio2 = ratio1
        x1 = a + R * (b - a)
        policy1 = x1 * uniform_policy + (1 - x1) * policy_values
        ratio1 = (np.dot(policy1, regret) ** 2 / (np.dot(policy1, v_a)))
    best_a = (a + b) / 2
    policy_out = best_a * uniform_policy + (1 - best_a) * policy_values

    return best_a, policy_out


class MCTS(object):

  def __init__(self, config):
    self.num_simulations = config.num_simulations
    self.config = config
    self.discount = config.discount
    self.pb_c_base = config.pb_c_base
    self.pb_c_init = config.pb_c_init
    self.init_value_score = config.init_value_score
    self.action_space = range(config.action_space)
    self.two_players = config.two_players
    self.known_bounds = config.known_bounds

    self.min_max_stats = MinMaxStats(*config.known_bounds)

  def run(self, root, network):
    self.min_max_stats.reset(*self.known_bounds)
    step_error = self.config.max_transitive_error / self.num_simulations
    min_max_v = MinMaxStats()

    search_paths = []
    for _ in range(self.num_simulations):
      # s2 = datetime.datetime.now()
      node = root
      search_path = [node]
      to_play = root.to_play

      while node.expanded():
        action, node = self.select_child(node)

        search_path.append(node)

        if self.two_players:
          to_play *= -1

      parent = search_path[-2]

      next_hidden_state = network.dynamics(parent.hidden_state, [action])
      abstract_representastion , abstarct_V = network.abstract_embed(next_hidden_state)
      node.abstract_v = abstarct_V.item()
      min_max_v.update(abstarct_V.item())

      policy_logits, value = network.prediction(abstract_representastion)

      node.expand(abstract_representastion, policy_logits, to_play, self.action_space, self.config)

      self.backpropagate(search_path, value.item(), to_play)

      for a in parent.children.keys():
        if parent.children[a].abstract_v != 0 and a != action:
          v1 = min_max_v.normalize(node.abstract_v)
          v2 = min_max_v.normalize(parent.children[a].abstract_v)
          if abs(v1 - v2) < step_error:
            parent.aggregation_times += 1
            if v1 > v2:
              parent.children.pop(a)
            else:
              parent.children.pop(action)


      search_paths.append(search_path)

      # e2 = datetime.datetime.now()
      #
      # print(f'sim time:{(e2 - s2).microseconds}')

    return search_paths

  def select_child(self, node):
    if node.visit_count == 0:
      _, action, child = max(
          (child.prior, action, child)
          for action, child in node.children.items())
    else:
      _, action, child = max(
          (self.ucb_score(node, child), action, child)
          for action, child in node.children.items())

    return action, child

  def ucb_score(self, parent, child):
    pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    if child.visit_count > 0:
      value = -child.value() if self.two_players else child.value()
      value_score = self.min_max_stats.normalize(child.reward + self.discount*value)

    else:
      value_score = self.init_value_score


    return prior_score + value_score

  def backpropagate(self, search_path, value, to_play):

    for idx, node in enumerate(reversed(search_path)):
      node.value_sum += value if node.to_play == to_play else -value
      node.visit_count += 1

      if self.two_players and node.to_play == to_play:
        reward = -node.reward
      else:
        reward = node.reward

      if idx < len(search_path) - 1:
        if self.two_players:
          new_q = node.reward - self.discount*node.value()
        else:
          new_q = node.reward + self.discount*node.value()
        self.min_max_stats.update(new_q)

      value = reward + self.discount*value


