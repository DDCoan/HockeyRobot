from TD3 import TD3
import numpy as np
import laserhockey.hockey_env as h_env


def eval_policy(env, player1, player2, seed, eval_episodes=50):
    env.seed(seed + 100)
    num_win = 0
    obs_buffer = []
    reward_buffer = []

    for i in range(eval_episodes):
        obs = env.reset()
        obs_agent2 = env.obs_agent_two()
        for _ in range(env.max_timesteps):
            # env.render()
            a1 = player1.act(obs)
            a2 = player2.act(obs_agent2)
            obs, r, d, info = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()
            obs_buffer.append(obs)
            reward_buffer.append(r)
            if d: break
        if info["winner"] == 1: num_win += 1
        continue

    obs_buffer = np.asarray(obs_buffer)
    reward_buffer = np.asarray(reward_buffer)
    print("---------------------------------------")
    print(
        f"Evaluation over {eval_episodes} episodes: {np.mean(reward_buffer):.3f} +- {np.std(reward_buffer):.3f}, success rate {num_win / eval_episodes}")
    print("---------------------------------------")
    return np.mean(obs_buffer), np.std(obs_buffer), np.mean(reward_buffer), np.std(
        reward_buffer), num_win / eval_episodes


def eval_models(player1_file="", player2_file="weak"):
    env = h_env.HockeyEnv(mode=0)

    state_dim = env.observation_space.shape[0]  # 18
    action_dim = int(env.action_space.shape[0] / 2)  # env.action_space.shape[0]  # 8 for both, 4 for 1
    max_action = float(env.action_space.high[0])

    player1 = TD3(state_dim, action_dim, max_action)

    player1.load(f"{player1_file}")

    if player2_file == "weak":
        player2 = h_env.BasicOpponent(weak=True)
    elif player2_file == "strong":
        player2 = h_env.BasicOpponent(weak=False)
    else:
        raise ValueError

    return eval_policy(env, player1, player2, 0, eval_episodes=100)


if __name__ == "__main__":
    eval_models(player1_file="model/TD3_Hockey-v0")  # evaluate trained model against the weak opponent
