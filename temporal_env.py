import gym
import numpy as np
from pylogics.parsers.ltl import parse_ltl
from pylogics.parsers.ldl import parse_ldl
from temprl.wrapper import TemporalGoal, TemporalGoalWrapper
from flloat.parser.ltlf import LTLfParser
from flloat.parser.ldlf import LDLfParser
from reward_monitor import create_monitor

class BoolGoalCartpole:
    def __init__(self, goal_string, goal_type, goal_pos=0, logic="ltlf"):
        if logic == "ltlf":
            self.goal = LTLfParser()(goal_string)
            automaton = self.goal.to_automaton()
        else:
            self.goal = LDLfParser()(goal_string)
            automaton = self.goal.to_automaton()
        self.goal_pos = goal_pos
        self.goal_type = goal_type
        self.tg = TemporalGoal(automaton=automaton, reward=1)

    def extract_fluents(self, obs, action):
        fluents = []
        if abs(self.goal_pos - obs[0]) <= 0.1:
            fluents.append("reach_goal")
        if abs(obs[2]) <= 0.209:
            fluents.append("balanced")
        return fluents

    def reward(self, automaton_state):
        return self.tg.automaton.reward if automaton_state[0] in self.tg.automaton.accepting_states else 0

    def performance(self, obs):
        if self.goal_type == "balance":
            return 1 if abs(obs[2]) <= 0.209 else 0
        else:
            return 1 if abs(self.goal_pos - obs[0]) <= 0.1 else 0


class QuantGoalCartpole:
    def __init__(self, goal_string, goal_type, goal_pos=0, logic="ltlf"):
        if logic == "ltlf":
            self.tg = parse_ltl(goal_string)
            self.monitor = create_monitor(self.tg)
        else:
            self.tg = parse_ldl(goal_string)
        self.goal_pos = goal_pos
        self.goal_type = goal_type

    def extract_fluents(self, obs):
        goal_dist = abs(self.goal_pos - obs[0])
        reach_goal = (self.goal_pos - goal_dist)/self.goal_pos if goal_dist <= self.goal_pos else 0
        angle_val = abs(obs[2])
        balanced = (0.209 - angle_val)/0.209 if angle_val <= 0.209 else 0
        return {"reach_goal": reach_goal, "balanced": balanced}

    def step(self, obs):
        fluents = self.extract_fluents(obs)
        self.monitor.step(fluents)
        return self.monitor.reward()

    def reset(self):
        self.monitor.reset()

    def performance(self, obs):
        if self.goal_type == "balance":
            return 1 if abs(obs[2]) <= 0.209 else 0
        else:
            return 1 if abs(self.goal_pos - obs[0]) <= 0.1 else 0


class BoolEnv(gym.Wrapper):
    def __init__(self, env, bool_goal):
        wrapped_env = TemporalGoalWrapper(env=env, temp_goals=[bool_goal.tg],
                                          fluent_extractor=bool_goal.extract_fluents)
        wrapped_env.observation_space.shape = (env.observation_space.shape[0] +
                                               wrapped_env.observation_space.spaces[1].shape[0],)
        super().__init__(wrapped_env)
        self.bool_goal = bool_goal
        self.obs = []

    def step(self, action):
        (obs, automaton_state), _, done, info = self.env.step(action)
        reward = self.bool_goal.reward(automaton_state)
        self.obs = obs
        return np.append(obs, automaton_state), reward, done, info

    def reset(self):
        obs, automaton_state = self.env.reset()
        return np.append(obs, automaton_state)

    def performance(self):
        return self.bool_goal.performance(self.obs)


class QuantEnv(gym.Wrapper):
    def __init__(self, env, quant_goal):
        super().__init__(env)
        self.quant_goal = quant_goal
        # Env shape + one value for the monitor state + one value per register
        self.observation_space.shape = (self.observation_space.shape[0] + 1 + len(quant_goal.monitor.V),)
        self.obs = []

    def step(self, action):
        obs, _, done, info = super().step(action)
        reward = self.quant_goal.step(obs)
        self.obs = obs
        extended_obs = np.append(obs,
                                 [self.quant_goal.monitor.current_state] + list(self.quant_goal.monitor.V.values()))
        return extended_obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self.obs = obs
        self.quant_goal.reset()
        extended_obs = np.append(obs,
                                 [self.quant_goal.monitor.current_state] + list(self.quant_goal.monitor.V.values()))
        return extended_obs

    def performance(self):
        return self.quant_goal.performance(self.obs)
