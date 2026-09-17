class Gridworld:
    def __init__(self, max_steps=50):
        self.rows=5
        self.cols=5

        self.start=(0,0)
        self.goal=(4,4)

        self.walls = {
            (0, 3),
            (1, 1),
            (1, 3),
            (2, 1),
            (3, 2),
            (3, 3),
        }

        self.actions_deltas={
            0 :(-1,0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }
        self.max_steps=max_steps
        self.position=self.start
        self.steps = 0
    def reset(self):
        self.position = self.start
        self.steps = 0

        info = {
            "steps": self.steps,
            "reached_goal": False,
        }

        return self.position, info
    def step(self,action):
        if action not in self.actions_deltas:
            raise ValueError("Action must be 0, 1, 2, or 3.")

        row_change,col_change=self.actions_deltas[action]

        current_row,current_col=self.position

        next_row=current_row+row_change
        next_col=current_col+col_change

        candidate_position = (next_row, next_col)
        outside_grid = (
            next_row < 0
            or next_row >= self.rows
            or next_col < 0
            or next_col >= self.cols
        )

        hit_wall = candidate_position in self.walls
        if outside_grid or hit_wall:
            candidate_position = self.position

        self.position = candidate_position
        self.steps+=1

        terminated = self.position == self.goal
        truncated = self.steps >= self.max_steps and not terminated

        reward = 1.0 if terminated else -0.01

        info = {
            "steps": self.steps,
            "hit_wall_or_boundary": outside_grid or hit_wall,
            "reached_goal": terminated,
        }

        return self.position, reward, terminated, truncated, info

    def render(self):
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for wall_row, wall_col in self.walls:
            grid[wall_row][wall_col] = "#"
        start_row, start_col = self.start
        goal_row, goal_col = self.goal
        agent_row, agent_col = self.position

        grid[start_row][start_col] = "S"
        grid[goal_row][goal_col] = "G"
        grid[agent_row][agent_col] = "A"
        for row in grid:
            print(" ".join(row))
        
# Create the environment
env = Gridworld()

# Start/reset the environment
state, info = env.reset()

# Print information
print("State:", state)
print("Info:", info)

# Display the grid
env.render()