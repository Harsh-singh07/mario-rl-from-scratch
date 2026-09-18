from environment import GridWorld


ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}


def test_known_path():
    print("\n--- Test 1: Known path to goal ---")

    env = GridWorld()
    state, info = env.reset()

    print("Starting state:", state)
    env.render()

    actions = [1, 1, 1, 1, 3, 3, 3, 3]

    for action in actions:
        state, reward, terminated, truncated, info = env.step(action)

        print(
            f"\nAction: {ACTION_NAMES[action]}"
            f" | State: {state}"
            f" | Reward: {reward}"
            f" | Terminated: {terminated}"
        )

        env.render()

        if terminated or truncated:
            break

    assert terminated, "The known path should reach the goal."
    assert state == env.goal, "The final state should be the goal."
    assert reward == 1.0, "Reaching the goal should give reward 1.0."

    print("\n[PASS] Known-path test passed.")


def test_boundary():
    print("\n--- Test 2: Grid boundary ---")

    env = GridWorld()
    state, _ = env.reset()

    state, reward, terminated, truncated, info = env.step(0)

    assert state == (0, 0), "Agent should not move above the grid."
    assert reward == -0.01
    assert info["hit_wall_or_boundary"] is True

    print("[PASS] Boundary test passed.")


def test_wall():
    print("\n--- Test 3: Wall collision ---")

    env = GridWorld()
    env.reset()

    env.step(1)
    state, reward, terminated, truncated, info = env.step(3)

    assert state == (1, 0), "Agent should not move into wall at (1, 1)."
    assert reward == -0.01
    assert info["hit_wall_or_boundary"] is True

    print("[PASS] Wall test passed.")


def test_step_limit():
    print("\n--- Test 4: Episode step limit ---")

    env = GridWorld(max_steps=3)
    env.reset()

    for _ in range(3):
        state, reward, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is True
    assert info["steps"] == 3

    print("[PASS] Step-limit test passed.")


if __name__ == "__main__":
    test_known_path()
    test_boundary()
    test_wall()
    test_step_limit()

    print("\n[PASS] All GridWorld tests passed.")