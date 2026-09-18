from environment import GridWorld
from random_agent import RandomAgent


def run_random_baseline(num_episodes=100):
    env = GridWorld(max_steps=50)
    agent = RandomAgent(seed=42)

    episode_rewards = []
    episode_steps = []
    successes = 0

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()

        total_reward = 0.0

        while True:
            action = agent.select_action(state)

            state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(info["steps"])

        if terminated:
            successes += 1

        if episode <= 5:
            print(
                f"Episode {episode:3d}"
                f" | Reward: {total_reward:.2f}"
                f" | Steps: {info['steps']:2d}"
                f" | Reached goal: {terminated}"
            )

    success_rate = (successes / num_episodes) * 100
    average_reward = sum(episode_rewards) / num_episodes
    average_steps = sum(episode_steps) / num_episodes

    print("\n--- Random Agent Baseline ---")
    print(f"Episodes:           {num_episodes}")
    print(f"Successful episodes: {successes}")
    print(f"Success rate:        {success_rate:.2f}%")
    print(f"Average reward:      {average_reward:.3f}")
    print(f"Average steps:       {average_steps:.2f}")

    return episode_rewards, episode_steps, successes


if __name__ == "__main__":
    run_random_baseline()