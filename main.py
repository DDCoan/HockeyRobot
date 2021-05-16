import torch
import numpy as np
import random
from TD3 import TD3
import os
import laserhockey.hockey_env as h_env
from evaluate_models import eval_policy


def run(env: h_env.HockeyEnv, player1_file="", player2_list=None, num_play=4, save_model=True,
        save_dir="self_play",
        **user_config):
    if player2_list is None:
        player2_list = ["weak", "strong"]
    _config = {
        "seed": 0,
        "start_episodes": int(1e2),
        "eval_freq": int(2e2),
        "max_episodes": int(2e3),
        "batch_size": 256,
        "expl_noise": 0.1,
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2
    }
    _config.update(user_config)

    policy_class = "TD3"
    file_name = f"{policy_class}_HockeyEnv_{_config['seed']}"

    env.seed(_config["seed"])
    torch.manual_seed(_config["seed"])
    np.random.seed(_config["seed"])

    state_dim = env.observation_space.shape[0]  # 18
    action_dim = int(env.action_space.shape[0] / 2)  # env.action_space.shape[0]  # 8 for both, 4 for 1
    max_action = float(env.action_space.high[0])

    kwargs = {"state_dim": state_dim if env.keep_mode else state_dim - 2, "action_dim": action_dim,
              "max_action": max_action,
              "policy_noise": _config["policy_noise"] * max_action,
              "noise_clip": _config["noise_clip"] * max_action}
    kwargs.update(_config)

    # construct both players
    player1 = TD3(**kwargs)

    if player1_file != "":
        player1.load(f"./models/{player1_file}")

    def _create_player2(player2_file):
        if player2_file == "weak":
            player2 = h_env.BasicOpponent(weak=True)
        elif player2_file == "strong":
            player2 = h_env.BasicOpponent(weak=False)
        else:
            player2 = TD3(**kwargs)
            player2.load(f"./models/{player2_file}")
        return player2

    def _run_episode(action_func, train=False):
        episode_reward = 0
        state, done = env.reset(), False
        state_agent2 = env.obs_agent_two()
        for t in range(env.max_timesteps):
            action = action_func(state,
                                 state_agent2)
            next_state, reward, done, _ = env.step(action)
            state_agent2 = env.obs_agent_two()
            done_bool = float(done)

            # Store data in replay buffer
            player1.store_transition((state, action, reward, next_state, done_bool))
            state = next_state
            episode_reward += reward
            if train:
                player1.train()
            if done: break
        print(f"Episode Num: {i + 1}, T: {t}, Reward: {episode_reward}")
        return episode_reward

    action_random = lambda obs1, obs2: env.action_space.sample()

    # collect initial data
    for i in range(_config["start_episodes"]):
        _run_episode(action_random, train=False)

    # run trials for
    for trial_num in range(num_play):
        if not os.path.exists(f"./results/{save_dir}/trial_{trial_num}"):
            os.makedirs(f"./results/{save_dir}/trial_{trial_num}")

        if save_model and not os.path.exists(f"./models/{save_dir}/trial_{trial_num}"):
            os.makedirs(f"./models/{save_dir}/trial_{trial_num}")

        n_opponents = len(player2_list)
        weights = [n_opponents / 2, n_opponents / 2] + list(range(1, n_opponents - 1))
        player2_file = random.choices(player2_list, weights)[0]

        print(f"opponent: {player2_file}")
        player2 = _create_player2(player2_file)

        player2.keep_mode = env.keep_mode

        # Evaluate untrained player1
        evaluations = [eval_policy(env, player1, player2, _config["seed"])]

        action_play = lambda obs1, obs2: np.hstack([(player1.act(np.array(obs1))
                                                     + np.random.normal(0, max_action * _config["expl_noise"],
                                                                        size=action_dim)).clip(
            -max_action, max_action),
            (player2.act(np.array(obs2))
             + np.random.normal(0, max_action * _config["expl_noise"], size=action_dim)).clip(
                -max_action, max_action)
        ])

        for i in range(_config["max_episodes"]):
            _run_episode(action_play, train=True)

            # Evaluate episode
            if (i + 1) % _config["eval_freq"] == 0:
                evaluations.append(eval_policy(env, player1, player2, _config["seed"]))
                np.save(f"./results/{save_dir}/trial_{trial_num}/{file_name}", evaluations)
                if save_model:
                    player1.save(
                        f"./models/{save_dir}/trial_{trial_num}/{file_name}_{int((i + 1) // _config['eval_freq'])}")
                    # add trained model to opponent list
                    player2_list.append(
                        f"{save_dir}/trial_{trial_num}/{file_name}_{int((i + 1) // _config['eval_freq'])}")


if __name__ == "__main__":
    opponents = ["weak", "strong"]  # initial opponent buffer
    env = h_env.HockeyEnv(mode=0)
    run(env=env, player1_file="", save_dir=f"normal", start_episodes=500, max_episodes=int(1e3), num_play=20)
