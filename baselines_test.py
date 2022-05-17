import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import ARS
from temporal_env import LDLCartPoleWrapper


def run():
    spec_string = "<balanced* ; reach_goal & balanced ; balanced* ; reach_start & balanced>tt"
    # spec_string = "FG reach_goal & G balanced"
    # spec_string = "G balanced"

    env = LDLCartPoleWrapper(spec_string, semantics="quant")
    # env = LDLCartPoleWrapper(spec_string, semantics="bool")

    eval_env = LDLCartPoleWrapper(spec_string, semantics="bool")
    eval_callback = EvalCallback(eval_env, log_path="./output", eval_freq=5_000, n_eval_episodes=10)

    # model = PPO('MlpPolicy', env, verbose=1).learn(1_000_000, callback=eval_callback)
    # model = ARS('MlpPolicy', env, verbose=1).learn(500_000, callback=eval_callback)
    model = ARS('LinearPolicy', env, verbose=1).learn(500_000, callback=eval_callback)

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


def visualise_results():
    evaluations_bool = np.load("output/cartpole_balance_ars_lin_bool.npz")
    evaluations_quant = np.load("output/cartpole_balance_ars_lin_quant.npz")
    performances_bool = evaluations_bool["results"]
    performances_quant = evaluations_quant["results"]#[:100]
    x_values = evaluations_bool["timesteps"]
    mean_performances_quant = np.mean(performances_quant, axis=1)
    # std_performances_quant = np.std(performances_quant, axis=1)
    mean_performances_bool = np.mean(performances_bool, axis=1)
    # std_performances_bool = np.std(performances_bool, axis=1)

    fig, ax = plt.subplots()
    ax.plot(x_values, mean_performances_bool, '--', label="Boolean")
    ax.plot(x_values, mean_performances_quant, label="Quantitative")
    # ax.fill_between(x_values, mean_performances_bool - std_performances_bool,
    #                 mean_performances_bool + std_performances_bool, alpha=0.3)
    # ax.fill_between(x_values, mean_performances_quant - std_performances_quant,
    #                 mean_performances_quant + std_performances_quant, alpha=0.3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Performance")
    ax.legend()
    # plt.show()
    ax.set_title(f"CartPole balance")
    plt.savefig(f"output/performances_both.pdf")
    # plt.close(fig)


# run()
visualise_results()
