import gym
import numpy as np
import math
from pylogics.parsers.ltl import parse_ltl
from pylogics.parsers.ldl import parse_ldl
from temprl.wrapper import TemporalGoal, TemporalGoalWrapper
from flloat.parser.ltlf import LTLfParser
from flloat.parser.ldlf import LDLfParser
from reward_monitor import create_monitor

from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper, VecEnv, VecEnvStepReturn
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


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
        reach_goal = (self.goal_pos - goal_dist) / self.goal_pos if goal_dist <= self.goal_pos else 0
        angle_val = abs(obs[2])
        balanced = (0.209 - angle_val) / 0.209 if angle_val <= 0.209 else 0
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


class LDLCartPoleWrapper(VecEnvWrapper):
    def __init__(self, spec_string, semantics):
        venv = VecFrameStack(DummyVecEnv([lambda: gym.make('CartPole-v1')]), 500)
        super().__init__(venv)
        # self.goal_spec = LDLfParser()(spec_string)
        self.spec_string = spec_string
        self.trace_bool = []
        self.trace_quant = []
        if semantics in ("bool", "quant"):
            self.semantics = semantics
        else:
            raise ValueError("Semantics must be either 'bool' or 'quant'.")

    def extract_bool_fluents(self, obs):
        reach_goal = abs(0.5 - obs[0]) <= 0.1
        reach_start = abs(obs[0]) <= 0.1
        balanced = abs(obs[2]) <= 0.209
        return {"reach_goal": reach_goal, "reach_start": reach_start, "balanced": balanced}

    def extract_quant_fluents(self, obs):
        reach_goal = max(0, 1 - abs(0.5 - obs[0]))
        reach_start = max(0, 1 - abs(obs[0]))
        balanced = max(0, (0.209 - abs(obs[2])) / 0.209)
        return {"reach_goal": reach_goal, "reach_start": reach_start, "balanced": balanced}

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self):
        obs, _, done, info = self.venv.step_wait()

        if done[0]:
            # Episode is over, so we need to give final reward
            # obs represents the first observation of the reset environment
            observations = info[0]['terminal_observation'][-4:]
        else:
            observations = obs[0, -4:]

        self.trace_bool.append(self.extract_bool_fluents(observations))
        bool_reward = self.bool_value()
        # Success is always the boolean reward, even in the quantitative case
        info[0]['is_success'] = bool_reward
        if self.semantics == "bool":
            # In boolean semantics, reward and success are the same
            reward = bool_reward
        else:
            self.trace_quant.append(self.extract_quant_fluents(observations))
            reward = self.quant_value()

        # print(len(self.trace_bool))
        if done[0]:
            # print(done[0])
            # Now we are getting the fluents for the reset environment's first observation
            fluents_bool = self.extract_bool_fluents(obs[0, -4:])
            self.trace_bool = [fluents_bool]

            if self.semantics == "quant":
                fluents_quant = self.extract_quant_fluents(obs[0, -4:])
                self.trace_quant = [fluents_quant]

        return obs, np.array([reward]), done, info

    def reset(self):
        obs = self.venv.reset()
        fluents_bool = self.extract_bool_fluents(obs[0, -4:])
        self.trace_bool = [fluents_bool]
        if self.semantics == "quant":
            fluents_quant = self.extract_quant_fluents(obs[0, -4:])
            self.trace_quant = [fluents_quant]
        return obs

    def bool_value(self):
        if self.spec_string == "<balanced* ; reach_goal & balanced ; balanced* ; reach_start & balanced>tt":
            return int(any([all([self.trace_bool[i]['reach_start']] +
                                [all([self.trace_bool[j]['balanced'] for j in range(0, i + 1)])] +
                                [any([self.trace_bool[j]['reach_goal'] for j in range(0, i)])])
                            for i in range(1, len(self.trace_bool))]))
        elif self.spec_string == "FG reach_goal & G balanced":
            return int(all([self.trace_bool[i]['balanced'] for i in range(0, len(self.trace_bool))] +
                       [self.trace_bool[-1]['reach_goal']]))
        elif self.spec_string == "G balanced":
            return int(all([self.trace_bool[i]['balanced'] for i in range(0, len(self.trace_bool))]))

    def quant_value(self):
        if self.spec_string == "<balanced* ; reach_goal & balanced ; balanced* ; reach_start & balanced>tt":
            return max([min(self.trace_quant[i]['reach_start'],
                            min([self.trace_quant[j]['balanced'] for j in range(0, i + 1)]),
                            max([self.trace_quant[j]['reach_goal'] for j in range(0, i)]))
                        for i in range(1, len(self.trace_quant))], default=0)
        elif self.spec_string == "FG reach_goal & G balanced":
            return min([self.trace_quant[i]['balanced'] for i in range(0, len(self.trace_quant))] +
                       [self.trace_quant[-1]['reach_goal']])
        elif self.spec_string == "G balanced":
            return min([self.trace_quant[i]['balanced'] for i in range(0, len(self.trace_quant))])
