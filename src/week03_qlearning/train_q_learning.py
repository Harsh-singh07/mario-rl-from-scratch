import numpy as np

from src.week02_gridworld.environment import GridWorld
from src.week03_qlearning.q_learning_agent import QLearningAgent


def train_q_learning(num_episodes=2_000):
    env = GridWorld(max_steps=50)

    agent = QLearningAgent(
        rows=env.rows,
        cols=env.cols,
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
    )

    rng = np.random.default_rng(42)

    episode_rewards = []
    successes = 0

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        total_reward = 0.0

        while True:
            # Temporary random exploration.
            # Session 4 will replace this with epsilon-greedy behavior.
            action = int(rng.integers(0, 4))

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
                truncated=truncated,
            )

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

        if terminated:
            successes += 1

        if episode % 200 == 0:
            recent_rewards = episode_rewards[-200:]
            average_recent_reward = sum(recent_rewards) / len(recent_rewards)

            print(
                f"Episode {episode:4d}"
                f" | Successes: {successes:3d}"
                f" | Recent average reward: {average_recent_reward:.3f}"
            )

    print("\n--- Training Complete ---")
    print(f"Episodes: {num_episodes}")
    print(f"Successful training episodes: {successes}")
    print(f"Non-zero Q-values: {np.count_nonzero(agent.q_table)}")

    print("\nQ-values at state (4, 3):")
    print("up, down, left, right")
    print(agent.q_table[4, 3])

    return agent, episode_rewards


if __name__ == "__main__":
    train_q_learning()