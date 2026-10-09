from src.week02_gridworld.environment import GridWorld
from src.week03_qlearning.train_q_learning import train_q_learning


def evaluate_agent(agent, num_episodes=100):
    env = GridWorld(max_steps=50)

    episode_rewards = []
    episode_steps = []
    successes = 0

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        total_reward = 0.0

        while True:
            # Evaluation uses learned knowledge only.
            action = agent.best_action(state)

            state, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(info["steps"])

        if terminated:
            successes += 1

    success_rate = (successes / num_episodes) * 100
    average_reward = sum(episode_rewards) / num_episodes
    average_steps = sum(episode_steps) / num_episodes

    print("\n--- Q-Learning Evaluation ---")
    print(f"Episodes:           {num_episodes}")
    print(f"Successful episodes: {successes}")
    print(f"Success rate:        {success_rate:.2f}%")
    print(f"Average reward:      {average_reward:.3f}")
    print(f"Average steps:       {average_steps:.2f}")

    return episode_rewards, episode_steps, successes


if __name__ == "__main__":
    agent, training_rewards = train_q_learning(
        num_episodes=2_000
    )

    evaluate_agent(agent)