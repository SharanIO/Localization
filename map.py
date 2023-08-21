import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Definitions for the map, obstacles, and robot's initial position.
MAP_WIDTH, MAP_HEIGHT = 10, 10
map_2d = [[0 for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]
obstacles = [(2, 2), (2, 3), (2, 4), (5, 5), (5, 6), (7, 2), (7, 3)]
for obs in obstacles:
    map_2d[obs[1]][obs[0]] = 1

current_position = (0, 0)
goal_position = (np.random.randint(0, MAP_WIDTH), np.random.randint(0, MAP_HEIGHT))

# 2. Definitions for the motion and sensor models.
def simulate_sensor_data(true_position):
    obstacle_distances = []
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            if map_2d[y][x] == 1:  # if there's an obstacle
                distance = np.sqrt((true_position[0]-x)**2 + (true_position[1]-y)**2)
                obstacle_distances.append(distance)
    return sorted(obstacle_distances)[:3]  # taking the distances to the three closest obstacles

def motion_model(position, delta, prior_belief):
    x, y = position
    prob = 0.0
    possible_previous_positions = [(x-delta[0], y-delta[1])]
    for prev_position in possible_previous_positions:
        px, py = prev_position
        if 0 <= px < MAP_WIDTH and 0 <= py < MAP_HEIGHT:
            prob += prior_belief[py][px]
    return prob

def sensor_model_for_markov(position, sensor_reading):
    simulated = simulate_sensor_data(position)
    prob = 1.0
    for true_dist, observed_dist in zip(simulated, sensor_reading):
        error = abs(true_dist - observed_dist)
        prob *= np.exp(-error**2 / (2 * 1**2))  # 1 here represents a chosen standard deviation for the noise
    return prob

# 3. Definitions for Markov localization (belief update procedures).
belief = [[1/(MAP_WIDTH * MAP_HEIGHT) for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]

def update_belief_with_motion(belief, motion):
    new_belief = [[0 for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            new_belief[y][x] = motion_model((x, y), motion, belief)
    return new_belief

def update_belief_with_sensor(belief, sensor_reading):
    new_belief = [[0 for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            new_belief[y][x] = belief[y][x] * sensor_model_for_markov((x, y), sensor_reading)
    return new_belief

def normalize_belief(belief):
    total = sum(sum(row) for row in belief)
    if total == 0: 
        raise ValueError("Total belief is 0.")
    return [[p / total for p in row] for row in belief]

def expected_position(belief):
    expected_x = 0
    expected_y = 0
    total_weight = 0
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            expected_x += x * belief[y][x]
            expected_y += y * belief[y][x]
            total_weight += belief[y][x]
    if total_weight < 1e-9:
        print("Warning: Belief probabilities sum to zero!")
        return (0, 0)
    return (expected_x / total_weight, expected_y / total_weight)

# 4. Visualization functions.
def setup_visualization(map_2d, robot_position, belief):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1, MAP_WIDTH)
    ax.set_ylim(-1, MAP_HEIGHT)
    robot_dot, = ax.plot(robot_position[0], robot_position[1], 'ro', ms=10, label='Robot Position')
    return fig, ax, robot_dot

# 5. Animation logic.
def compute_move(current, goal):
    delta_x, delta_y = goal[0] - current[0], goal[1] - current[1]
    dx, dy = 0, 0
    if delta_x != 0:
        dx = 1 if delta_x > 0 else -1
    if delta_y != 0:
        dy = 1 if delta_y > 0 else -1
    return dx, dy

def animate_towards_goal(frame, ax, robot_dot):
    global current_position, belief
    delta = compute_move(current_position, goal_position)
    current_position = (current_position[0] + delta[0], current_position[1] + delta[1])
    sensor_reading = simulate_sensor_data(current_position)
    belief = update_belief_with_motion(belief, delta)
    belief = update_belief_with_sensor(belief, sensor_reading)
    belief = normalize_belief(belief)
    ax.clear()

    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            if map_2d[y][x] == 1:  # if there's an obstacle
                ax.plot(x, y, 'ks', ms=5)  # black square for obstacle

    ax.plot(goal_position[0], goal_position[1], 'g*', ms=15, label='Goal Position')
    ax.plot(current_position[0], current_position[1], 'ro', ms=10, label='Robot Position')

    expected_x, expected_y = expected_position(belief)
    ax.plot(expected_x, expected_y, 'bo', ms=10, label='Expected Position')  # 'bo' is blue circle

    ax.legend(loc="upper left")
    return robot_dot,

fig, ax, robot_dot = setup_visualization(map_2d, current_position, belief)
ani = FuncAnimation(fig, animate_towards_goal, frames=100, fargs=(ax, robot_dot), interval=1000, repeat=False)
plt.show()
