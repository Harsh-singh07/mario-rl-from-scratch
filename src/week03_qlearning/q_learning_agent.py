import numpy as np


class QLearningAgent:
    def __init__(
        self,
        rows,
        cols,
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.q_table = np.zeros((rows, cols, num_actions))

    def best_action(self, state):
        row, col = state
        return int(np.argmax(self.q_table[row, col]))

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated,
    ):
        row, col = state
        next_row, next_col = next_state

        current_q_value = self.q_table[row, col, action]

        episode_finished = terminated or truncated

        if episode_finished:
            best_next_q_value = 0.0
        else:
            best_next_q_value = np.max(
                self.q_table[next_row, next_col]
            )

        target = reward + (
            self.discount_factor * best_next_q_value
        )

        td_error = target - current_q_value

        new_q_value = current_q_value + (
            self.learning_rate * td_error
        )

        self.q_table[row, col, action] = new_q_value

        return td_error