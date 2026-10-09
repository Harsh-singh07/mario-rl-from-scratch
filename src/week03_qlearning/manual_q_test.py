import numpy as np

from q_learning_agent import QLearningAgent


def test_non_terminal_update():
    agent = QLearningAgent(rows=5, cols=5)

    state = (0, 0)
    action = 3
    reward = -0.01
    next_state = (0, 1)

    td_error = agent.update(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        terminated=False,
        truncated=False,
    )

    updated_q_value = agent.q_table[0, 0, 3]

    print("Non-terminal update")
    print(f"TD error: {td_error:.4f}")
    print(f"Updated Q-value: {updated_q_value:.4f}")

    assert np.isclose(updated_q_value, -0.001)

    print("Non-terminal Q-update test passed.")


def test_terminal_update():
    agent = QLearningAgent(rows=5, cols=5)

    state = (4, 3)
    action = 3
    reward = 1.0
    next_state = (4, 4)

    agent.update(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        terminated=True,
        truncated=False,
    )

    updated_q_value = agent.q_table[4, 3, 3]

    print("\nTerminal update")
    print(f"Updated Q-value: {updated_q_value:.4f}")

    assert np.isclose(updated_q_value, 0.1)

    print("Terminal Q-update test passed.")


if __name__ == "__main__":
    test_non_terminal_update()
    test_terminal_update()
    print("\n All Q-learning update tests passed.")