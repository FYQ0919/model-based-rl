import numpy as np
import torch
import math
import datetime


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
    self.aggregation_times = 0
    self.last_policy = None
    self.parent = None
    self.best_a = None

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

  def expand(self, network_output, to_play, actions, config):
    self.to_play = to_play
    self.hidden_state = network_output.hidden_state
    actions = np.array(actions)

    if network_output.reward:
      self.reward = network_output.reward.item()

    sample_num = config.num_sample_action

    policy_logits = network_output.policy_logits

    policy_values = torch.softmax(
      torch.tensor([policy_logits[0][a].item() for a in actions]), dim=0
    ).numpy().astype('float64')

    policy_values /= policy_values.sum()

    policy_values = 0.999 * policy_values + 1e-3 * np.ones(actions.shape) / len(actions)

    self.best_a = actions[np.argmax(policy_values)]

    if sample_num > 0:

      if self.last_policy != None:


        n_r = (self.reward - config.min_r)/ (config.max_r - config.min_r)

        regret = config.max_r * np.ones(shape=actions.shape) - self.reward * self.last_policy
        v_a = (n_r * np.ones(shape=actions.shape) - (n_r * np.ones(shape=actions.shape)* self.last_policy) ** 2)
        uniform_policy = np.ones(actions.shape) / len(actions)

        best_alpha, best_dis = self.golden_selection(regret, v_a, uniform_policy, policy_values)
      else:
        best_dis = policy_values

      if len(actions) > sample_num:
        sample_action = np.random.choice(actions, size=sample_num, replace=False, p=best_dis)
      else:
        sample_action = actions

      sample_policy_values = torch.softmax(
        torch.tensor([policy_logits[0][a] for a in sample_action]), dim=0
      ).numpy().astype('float64')

      for i in range(len(sample_action)):
        a = sample_action[i]
        p = sample_policy_values[i]
        self.children[a] = Node(p)

    else:
      policy = {a: policy_values[i] for i, a in enumerate(actions)}

      for action, p in policy.items():
        self.children[action] = Node(p)
        self.children[action].last_policy = policy_values



  def add_exploration_noise(self, dirichlet_alpha, frac):
    actions = list(self.children.keys())
    noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
    for a, n in zip(actions, noise):
      self.children[a].prior = self.children[a].prior*(1-frac) + n*frac

  def golden_selection(self, regret, v_a, uniform_policy, policy_values):
    R = 0.618033989
    C = 1 - R
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
    self.step_error = self.config.step_error

    search_paths = []
    for _ in range(self.num_simulations):

      node = root
      search_path = [node]
      to_play = root.to_play

      while node.expanded():
        action, node = self.select_child(node)
        search_path.append(node)

        if self.two_players:
          to_play *= -1

      parent = search_path[-2]

      network_output = network.recurrent_inference(parent.hidden_state, [action])
      node.expand(network_output, to_play, self.action_space, self.config)
      for child in node.children.values():
        child.parent = node
      self.backpropagate(search_path, network_output.value.item(), to_play)

      if search_path not in search_paths:

        search_paths.append(search_path)

      if self.step_error > 0 and len(search_paths) > 1:

        delet_paths = []

        for i in range(len(search_paths)-1):
          branch1 = search_path
          branch2 = search_paths[i]
          different_nodes = [[],[]]
          if len(branch1) == len(branch2):

            branch_value_loss = 0
            aggregation = True

            for j in range(1,len(branch1)):
              if branch1[j] != branch2[j]:
                is_aggregation, value_loss = self.abstract(branch1[j], branch2[j], type=self.config.abstract_type)
                if not is_aggregation:
                  aggregation = False
                  break
                branch_value_loss += value_loss
                different_nodes[0].append(branch1[j])
                different_nodes[1].append(branch2[j])
              else:
                continue

            if aggregation:

              root.aggregation_times += len(branch1)
              if branch_value_loss >= 0:

                 delet_node = different_nodes[1][0]
                 visit_count = delet_node.visit_count
                 value_sum = delet_node.value_sum
                 abstract_node = different_nodes[0][0]
                 abstract_node.visit_count += visit_count
                 abstract_node.value_sum += value_sum

                 delet_paths.append(branch2)
                 for a, n in delet_node.parent.children.items():
                   if n == delet_node:
                     delet_key = a

                 if delet_key in delet_node.parent.children.keys():
                   delet_node.parent.children.pop(delet_key)


              else:

                delet_node = different_nodes[0][0]
                visit_count = delet_node.visit_count
                value_sum = delet_node.value_sum
                abstract_node = different_nodes[1][0]
                abstract_node.visit_count += visit_count
                abstract_node.value_sum += value_sum
                delet_paths.append(branch1)
                for a, n in delet_node.parent.children.items():
                  if n == delet_node:
                    delet_key = a
                if delet_key in delet_node.parent.children.keys():
                  delet_node.parent.children.pop(delet_key)
                break
        for path in delet_paths:
          search_paths.remove(path)

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

  def abstract(self, node1, node2, type):
    if type == 1:
      if node1.value() == node2.value() and node1.best_a == node2.best_a:
        value_loss = node1.value() - node2.value()
        return True, value_loss
      else:
        return False, 0

    elif type == 2:
      if abs(node1.value() - node2.value()) < self.step_error and node1.best_a == node2.best_a:
        value_loss = node1.value() - node2.value()
        return True, value_loss
      else:
        return False, 0

    elif type == 3:
      value_loss = node1.value() - node2.value()
      if len(node1.children.keys()) > 0 and len(node2.children.keys()) > 0:

        for a in node1.children.keys():
          if a in node2.children.keys():
           Q_sa = abs(node1.children[a].value() - node2.children[a].value())
           if Q_sa > 0:
             return False, 0
        return True, value_loss
      else:
        return True, value_loss

    elif type == 4:

      value_loss = node1.value() - node2.value()

      if len(node1.children.keys()) > 0 and len(node2.children.keys()) > 0:
        for a in node1.children.keys():
          if a in node2.children.keys():
            Q_sa = abs(node1.children[a].value() - node2.children[a].value())
            if Q_sa > self.step_error:
              return False, 0

        return True, value_loss
      else:
        return True, value_loss

    elif type == 5:
      value_loss = node1.value() - node2.value()

      if len(node1.children.keys()) > 0 and len(node2.children.keys()) > 0:
        for a in node1.children.keys():
          if a in node2.children.keys():
            
            if round(node1.children[a].value()/self.step_error) != round(node2.children[a].value()/self.step_error):
              return False, 0

        return True, value_loss
      else:
        return True, value_loss
    elif type == 6:
      if abs(node1.value() - node2.value()) < self.step_error:
        value_loss = node1.value() - node2.value()
        return True, value_loss
      else:
        return False, 0
    else:
      return False, 0


