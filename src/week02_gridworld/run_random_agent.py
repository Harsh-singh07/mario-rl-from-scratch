import matplotlib.pyplot as plt
from pathlib import Path
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
def save_baseline_plots(episode_rewards, episode_steps):
    output_dir = Path("artifacts/week02")
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = range(1, len(episode_rewards) + 1)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(episodes, episode_rewards, color="steelblue")
    axes[0].set_title("Random Agent: Episode Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes, episode_steps, color="darkorange")
    axes[1].set_title("Random Agent: Steps Per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()

    plot_path = output_dir / "random_baseline_results.png"
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")

    print(f"\nGraph saved to: {plot_path}")


if __name__ == "__main__":
    episode_rewards, episode_steps, successes = run_random_baseline()

    save_baseline_plots(
        episode_rewards,
        episode_steps,
    )