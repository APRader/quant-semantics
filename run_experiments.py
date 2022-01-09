import argparse

import gym
import torch
import ddqn
import matplotlib.pyplot as plt
import numpy as np
import random
import string

from temporal_env import BoolGoalCartpole, BoolEnv, QuantGoalCartpole, QuantEnv

NUM_RUNS = 5
NUM_EPISODES = 2000
MEASURE_STEP = 50


def setup_cartpole(goal_type, goal_string, goal_string_flloat):
    """
    Sets up CartPole gym environment with appropriate temporal goals.
    :return: Gym environment for boolean and quantitative goal.
    """

    if goal_type == "balance":
        # Creating boolean temporal environment
        tg_bool = BoolGoalCartpole(goal_string_flloat, goal_type)
        # Creating quantitative temporal environment
        tg_quant = QuantGoalCartpole(goal_string, goal_type)
    elif goal_type == "reach_goal_near":
        goal_pos = 0.5
        print(f"Goal position: {goal_pos:0.2f}")
        tg_bool = BoolGoalCartpole(goal_string_flloat, goal_type, goal_pos)
        # Creating quantitative temporal environment
        tg_quant = QuantGoalCartpole(goal_string, goal_type, goal_pos)
    elif goal_type == "reach_goal_far":
        goal_pos = 1
        print(f"Goal position: {goal_pos:0.2f}")
        tg_bool = BoolGoalCartpole(goal_string_flloat, goal_type, goal_pos)
        # Creating quantitative temporal environment
        tg_quant = QuantGoalCartpole(goal_string, goal_type, goal_pos)
    else:
        goal_pos = 2
        print(f"Goal position: {goal_pos:0.2f}")
        tg_bool = BoolGoalCartpole(goal_string_flloat, goal_type, goal_pos)
        # Creating quantitative temporal environment
        tg_quant = QuantGoalCartpole(goal_string, goal_type, goal_pos)

    env_bool = BoolEnv(gym.make('CartPole-v1'), tg_bool)
    env_quant = QuantEnv(gym.make('CartPole-v1'), tg_quant)

    return env_bool, env_quant


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type",
                        choices=["balance", "reach_goal_near", "reach_goal_far", "reach_goal_very_far"],
                        help="choose which type of experiment to run")
    parser.add_argument("-s", "--seed", type=int, help="the seed for the random number generator")
    args = parser.parse_args()

    characters = string.ascii_letters + string.digits
    random_string = ''.join((random.choice(characters) for _ in range(5)))
    print(f"File id: {random_string}")
    env_bool = None
    env_quant = None

    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = random.randint(0, 1000)

    performances_bool = []
    performances_quant = []
    rewards_bool = []
    rewards_quant = []

    if args.type == "balance":
        goal_string = "G(balanced)"
        goal_string_flloat = goal_string
    elif args.type in ("reach_goal_near", "reach_goal_far", "reach_goal_very_far"):
        goal_string = "F(reach_goal & X false)"
        goal_string_flloat = "F(reach_goal & WX false)"
        # goal_string = "F(reach_goal & X false) & G balanced"
        # goal_string_flloat = "F(reach_goal & WX false) & G balanced"
    else:
        exit()

    for i in range(NUM_RUNS):
        print(f"Run {i + 1}")

        env_bool, env_quant = setup_cartpole(args.type, goal_string, goal_string_flloat)

        print("Boolean")
        performance_bool, reward_bool = ddqn.main(env_bool, num_episodes=NUM_EPISODES, measure_step=MEASURE_STEP,
                                                  measure_repeats=5, eps_decay=0.99, seed=seed, render=False)
        performances_bool.append(performance_bool)
        rewards_bool.append(reward_bool)

        print("Quantitative")
        performance_quant, reward_quant = ddqn.main(env_quant, num_episodes=NUM_EPISODES, measure_step=MEASURE_STEP,
                                                    measure_repeats=5, eps_decay=0.99, seed=seed, render=False)
        performances_quant.append(performance_quant)
        rewards_quant.append(reward_quant)

    performances_bool = np.array(performances_bool)
    performances_quant = np.array(performances_quant)
    rewards_bool = np.array(rewards_bool)
    rewards_quant = np.array(rewards_quant)

    mean_performances_bool = np.mean(performances_bool, axis=0)
    mean_performances_quant = np.mean(performances_quant, axis=0)
    mean_rewards_bool = np.mean(rewards_bool, axis=0)
    mean_rewards_quant = np.mean(rewards_quant, axis=0)
    std_performances_bool = np.std(performances_bool, axis=0)
    std_performances_quant = np.std(performances_quant, axis=0)
    std_rewards_bool = np.std(rewards_bool, axis=0)
    std_rewards_quant = np.std(rewards_quant, axis=0)

    # Save numpy arrays
    np.savez(f"output/{random_string}_{goal_string}.npz",
             mean_performances_bool=mean_performances_bool, mean_performances_quant=mean_performances_quant,
             std_performances_bool=std_performances_bool, std_performances_quant=std_performances_quant,
             mean_rewards_bool=mean_rewards_bool, mean_rewards_quant=mean_rewards_quant,
             std_rewards_bool=std_rewards_bool, std_rewards_quant=std_rewards_quant)

    x_values = range(0, NUM_EPISODES, MEASURE_STEP)
    fig, ax = plt.subplots()
    ax.plot(x_values, mean_performances_bool, '--', label="Boolean")
    ax.plot(x_values, mean_performances_quant, label="Quantitative")
    ax.fill_between(x_values, mean_performances_bool - std_performances_bool,
                    mean_performances_bool + std_performances_bool, alpha=0.3)
    ax.fill_between(x_values, mean_performances_quant - std_performances_quant,
                    mean_performances_quant + std_performances_quant, alpha=0.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Performance")
    ax.legend()
    ax.set_title(f"CartPole {args.type}")
    plt.savefig(f"output/{random_string}_{goal_string}_{args.type}_performances.pdf")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x_values, mean_rewards_bool, '--', label="Boolean")
    ax.plot(x_values, mean_rewards_quant, label="Quantitative")
    ax.fill_between(x_values, mean_rewards_bool - std_rewards_bool,
                    mean_rewards_bool + std_rewards_bool, alpha=0.3)
    ax.fill_between(x_values, mean_rewards_quant - std_rewards_quant,
                    mean_rewards_quant + std_rewards_quant, alpha=0.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rewards")
    ax.legend()
    ax.set_title(f"CartPole {args.type}")
    plt.savefig(f"output/{random_string}_{goal_string}_{args.type}_rewards.pdf")

    print(f"DONE - All {NUM_RUNS} runs completed")
    print(f"Figure saved with id {random_string}")
