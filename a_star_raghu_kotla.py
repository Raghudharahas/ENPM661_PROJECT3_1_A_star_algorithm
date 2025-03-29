import numpy as np
import cv2
import time
import heapq
from collections import deque
import math


# Maze and map parameters
map_height_mm = 100
map_width_mm = 300
scale = 1

# Create blank maze in BGR
maze = np.ones((map_height_mm * scale, map_width_mm * scale, 3), dtype=np.uint8) * 255
# Add 5mm black boundary as obstacle
boundary_thickness = 5  # mm

# Top and bottom boundaries
maze[0:boundary_thickness, :] = (0, 0, 0)                     # Top edge
maze[-boundary_thickness:, :] = (0, 0, 0)                     # Bottom edge

# Left and right boundaries
maze[:, 0:boundary_thickness] = (0, 0, 0)                     # Left edge
maze[:, -boundary_thickness:] = (0, 0, 0)                     # Right edge

# Coordinate grid
yy, xx = np.meshgrid(np.arange(maze.shape[0]), np.arange(maze.shape[1]), indexing='ij')

# Flip Y for bottom-left origin
def flip_y(y, height):
    return height - y

# Rectangle mask function
# This function creates a rectangular mask for the given coordinates
def rectangle_mask(x, y, x0, x1, y0, y1):
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

# draw letter E 
def draw_letter_E(x, y, origin, width, height, thickness):
    x_shifted = x - origin[0]
    y_shifted = y - flip_y(origin[1], map_height_mm)

    spine = rectangle_mask(x_shifted, y_shifted, 0, thickness, 0, height)
    top = rectangle_mask(x_shifted, y_shifted, thickness, width, 0, thickness)
    mid = rectangle_mask(x_shifted, y_shifted, thickness, width,
        (height // 2 - thickness // 2),
        (height // 2 + thickness // 2))
    bottom = rectangle_mask(x_shifted, y_shifted, thickness, width, height - thickness, height)
    return spine | top | mid | bottom

# Letter E properties
thickness = 5
height = 25
width = 13
origin_E = (30, 65)
E_mask = draw_letter_E(xx, yy, origin_E, width, height, thickness)
maze[E_mask] = (0, 0, 0)  # Black for obstacles

# FUNCTION TO DRAW LETTER "N"
def draw_letter_N(x, y, origin, width, height, thickness, steps):
    x_shifted = x - origin[0]
    y_shifted = y - flip_y(origin[1], map_height_mm)

    left_bar = rectangle_mask(x_shifted, y_shifted, 0, thickness, 0, height)
    right_bar = rectangle_mask(x_shifted, y_shifted, width - thickness, width, 0, height)

    step_height = height // steps
    step_width = (width - 2 * thickness) // steps
    
    # Create diagonal line mask
    diagonal = np.zeros_like(left_bar)
    for i in range(steps):
        x0 = thickness + i * step_width
        x1 = x0 + thickness
        y0 = i * step_height
        y1 = y0 + thickness
        diagonal |= rectangle_mask(x_shifted, y_shifted, x0, x1, y0, y1)

    return left_bar | right_bar | diagonal 

origin_N = (70, 65)
N_mask = draw_letter_N(xx, yy, origin_N, 20, 25, 5, 10)
maze[N_mask] = (0, 0, 0)  # Black for obstacles

# ---------- FUNCTION TO DRAW LETTER "P" ----------
def semi_circle_mask(x, y, cx, cy, r, thickness):
    outer = ((x - cx)**2 + (y - cy)**2) <= r**2
    inner = ((x - cx)**2 + (y - cy)**2) >= (r - thickness)**2
    right_half = x >= cx
    return outer & inner & right_half

#Function to draw letter P
def draw_letter_P(x, y, origin, width, height, thickness, radius):
    x_shifted = x - origin[0]
    y_shifted = y - flip_y(origin[1], map_height_mm)
    
    #spine and circular loop
    spine = rectangle_mask(x_shifted, y_shifted, 0, thickness, 0, height)
    cx = thickness
    cy = radius
    circular_loop = semi_circle_mask(x_shifted, y_shifted, cx, cy, radius, thickness)
    bar = rectangle_mask(x_shifted, y_shifted, thickness, cx + radius, cy - thickness // 2, cy + thickness // 2)
    return spine | circular_loop | bar

origin_P = (110, 65)
P_mask = draw_letter_P(xx, yy, origin_P, 20, 25, 5, 6)
maze[P_mask] = (0, 0, 0)  # Black for obstacles

# ---------- FUNCTION TO DRAW LETTER "M" ----------
def draw_letter_M(x, y, origin, width, height, thickness, steps):
    x_shifted = x - origin[0]
    y_shifted = y - flip_y(origin[1], map_height_mm)

    left_bar = rectangle_mask(x_shifted, y_shifted, 0, thickness, 0, height)
    right_bar = rectangle_mask(x_shifted, y_shifted, width - thickness, width, 0, height)

    step_height = height // steps
    step_width = (width // 2 - thickness) // steps

    left_diag = np.zeros_like(left_bar)
    right_diag = np.zeros_like(right_bar)
    # Create diagonal line masks
    # Left diagonal
    for i in range(steps):
        x0 = thickness + i * step_width
        x1 = x0 + thickness
        y0 = i * step_height
        y1 = y0 + thickness
        left_diag |= rectangle_mask(x_shifted, y_shifted, x0, x1, y0, y1)
    
    # Right diagonal
    # The right diagonal is drawn in reverse order
    for i in range(steps):
        x0 = width - thickness - i * step_width - thickness
        x1 = x0 + thickness
        y0 = i * step_height
        y1 = y0 + thickness
        right_diag |= rectangle_mask(x_shifted, y_shifted, x0, x1, y0, y1)

    return left_bar | right_bar | left_diag | right_diag

origin_M = (140, 65)
M_mask = draw_letter_M(xx, yy, origin_M, 22, 25, 5, 5)
maze[M_mask] = (0, 0, 0)  # Black for obstacles

# -Function to draw digit 6
def draw_algebraic_6_first(x, y, origin):
    cx, cy = origin
    cy = flip_y(cy, map_height_mm)

    x_shifted = x - cx
    y_shifted = y - cy
    
    # Circle and ellipse parameters
    circle_mask = (x_shifted**2 + y_shifted**2) <= 9**2
    a, b = 10, 21
    outer_ellipse = ((x_shifted / a)**2 + (y_shifted / b)**2) <= 1

    angles = np.arctan2(y_shifted, x_shifted) * 180 / np.pi
    arc_range = (angles >= -180) & (angles <= -75)
    # Create the arc mask
    ellipse_arc = outer_ellipse & arc_range
    return circle_mask | ellipse_arc

#draw first digit 6
digit_6_mask_1 = draw_algebraic_6_first(xx, yy, (195, 47))
maze[digit_6_mask_1] = (0, 0, 0)  # Black for obstacles

#draw second digit 6
digit_6_mask_2 = draw_algebraic_6_first(xx, yy, (235, 47))
maze[digit_6_mask_2] = (0, 0, 0)  # Black for obstacles

# Function to draw digit 1
def draw_digit_1(x, y, origin, width=5, height=25):
    x0, y0 = origin
    y0 = flip_y(y0, map_height_mm)
    x1 = x0 + width
    y1 = y0 + height
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
origin_1 = (270, 65)
digit_1_mask = draw_digit_1(xx, yy, origin_1, 5, 25)
maze[digit_1_mask] = (0, 0, 0)  # Black for obstacles


# Convert the RGB maze to binary for path planning
def convert_to_binary(maze_rgb):
    # Convert to grayscale
    maze_gray = cv2.cvtColor(maze_rgb, cv2.COLOR_BGR2GRAY)
    # Convert to binary where obstacles are 1 and free space is 0
    _, maze_binary = cv2.threshold(maze_gray, 128, 1, cv2.THRESH_BINARY_INV)
    return maze_binary

# Clearance input
clearance = input("Enter the clearance value for the maze (default: 2): ")
if clearance == "":
    clearance = 2 # default clearance
else:
    clearance = int(clearance)
    if clearance == 0:
        clearance = 2 # default clearance

clearance = int(clearance)

robot_radius = input("Enter robot radius in mm (default 5): ") 
if robot_radius == "":
    robot_radius = 5
else:
    robot_radius = int(robot_radius)
    if robot_radius == 0:
        robot_radius = 5
robot_radius = int(robot_radius)
# Step size input
step_size = input("Enter step size (<=10 units, default 1): ")
if step_size == "": # default step size
    step_size = 1
else:
    step_size = int(step_size)
    if step_size > 10:
        step_size = 1
step_size = int(step_size) #step size in mm
total_clearance = robot_radius + clearance # total clearance in mm
print(f"Total clearance: {total_clearance} mm")
theta_values = [i for i in range(0, 360, 30)]

# Create an inflated obstacle map with different color for the inflated border
def create_inflated_map(maze_rgb, total_clearance,scale):
    # Convert to binary
    maze_binary = convert_to_binary(maze_rgb)
    inflated_map = np.ones((maze_binary.shape[0], maze_binary.shape[1], 3), dtype=np.uint8) * 255
    
    clearance_px=int(total_clearance * scale)  # Convert total_clearance from mm to pixels
    # Mark original obstacles in black
    inflated_map[maze_binary == 1] = [0, 0, 0]  # Black for original obstacles
    
    # Create inflated binary map (inflated obstacles)
    kernel = np.ones((2 * clearance_px + 1, 2 * clearance_px + 1), np.uint8)
    inflated_binary = cv2.dilate(maze_binary.astype(np.uint8), kernel, iterations=1)

    # Create a mask for just the inflated border (inflated minus original)
    border_mask = (inflated_binary == 1) & (maze_binary == 0)
    
    # Mark inflated borders in a different color (blue)
    inflated_map[border_mask] = [255, 0, 0]  # Blue for inflation border
    
    # Return both the visualization map and the binary inflated map for planning
    return inflated_map, inflated_binary

#maze inflation
inflated_maze_vis, inflated_maze_binary = create_inflated_map(maze, total_clearance,scale)
# Visited matrix setup
rows, cols = inflated_maze_binary.shape
visited = np.zeros((int(rows * 2), int(cols * 2), 12), dtype=bool)
# Display the original maze
resized_maze = cv2.resize(maze, (maze.shape[1] * 5, maze.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
cv2.imshow('Original Maze', resized_maze)
print("Press any key to continue to display the inflated maze")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the inflated maze with colored border
resized_inflated = cv2.resize(inflated_maze_vis, (inflated_maze_vis.shape[1] * 5, inflated_maze_vis.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
cv2.imshow('Inflated Maze with Colored Border', resized_inflated)
print("Press any key to continue to enter start and goal positions")
cv2.waitKey(0)
cv2.destroyAllWindows()

#action space
def wrap_angle(theta):
    return theta % 360

def angle_index(theta):
    return theta // 30

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Check if the goal is reached with threshold 1.5 for position and 30 degrees for orientation
def goal_reached(curr, goal):
    return euclidean(curr[:2], goal[:2]) <= 1.5 and abs(curr[2] - goal[2]) % 360 <= 30

# Move function to calculate new position and orientation
def move(node, angle_offset):
    x, y, theta = node
    new_theta = wrap_angle(theta + angle_offset)
    rad = math.radians(new_theta)
    new_x = x + step_size * math.cos(rad)
    new_y = y + step_size * math.sin(rad)
    return (round(new_x, 1), round(new_y, 1), new_theta)

actions = [-60, -30, 0, 30, 60] # 5 angles


# A* search algorithm
def astar_oriented(start, goal, maze):
    pq = []
    heapq.heappush(pq, (0 + euclidean(start[:2], goal[:2]), 0, start))
    parent_map = {start: None}
    visited[int(start[1]*2), int(start[0]*2), angle_index(start[2])] = True
    explored_nodes = []
    # whilwe loop to explore the nodes
    while pq:
        _, cost, current = heapq.heappop(pq)
        explored_nodes.append(current)
        
        # Check if the current node is the goal
        if goal_reached(current, goal):
            return backtrack_path(parent_map, current), parent_map,explored_nodes
        
        # Generate possible moves
        for action in actions:
            new_node = move(current, action)
            x, y, theta = new_node
            xi, yi, ti = int(x*2), int(y*2), angle_index(theta)
            # Check if the new node is within bounds and not visited
            if 0 <= xi < visited.shape[1] and 0 <= yi < visited.shape[0]:  
                # Check if the new node is not visited and not an obstacle
                if not visited[yi, xi, ti] and  inflated_maze_binary[int(y), int(x)] == 0:
                    visited[yi, xi, ti] = True
                    new_cost = cost + step_size
                    total_cost = new_cost + euclidean(new_node[:2], goal[:2])
                    heapq.heappush(pq, (total_cost, new_cost, new_node))
                    parent_map[new_node] = current
    return None, parent_map,explored_nodes

# Backtrack the path from goal to start
def backtrack_path(parent_map, goal):
    path = []
    while goal is not None:
        path.append(goal)
        goal = parent_map[goal]
    return path[::-1]

# Get valid (x, y, theta) user input with obstacle check
def get_oriented_position(prompt):
    while True:
        try:
            x, y, theta = map(int, input(prompt).split())
            if theta % 30 != 0:
                print("Theta must be a multiple of 30 degrees")
                continue
            if not (0 <= x < map_width_mm and 0 <= y < map_height_mm):
                print("Coordinates out of map bounds")
                continue
            y_flipped = map_height_mm - 1 - y
            if inflated_maze_binary[y_flipped, x] == 0:
                return (x, y, theta)
            else:
                print("Point is in obstacle or inflated region")
        except:
            print("Invalid input. Format: x y theta")


#Run A* with User Input
print("Enter Start and Goal Coordinates in format: x y θ (e.g., 10 15 30)")
start_pose = get_oriented_position("Enter START coordinates (x y θ): ")
goal_pose = get_oriented_position("Enter GOAL coordinates (x y θ): ")
print("running A* algorithm...")

# Start the timer and run A*
start_time = time.time()
path, parent_map, explored_nodes = astar_oriented(start_pose, goal_pose, inflated_maze_binary)
end_time = time.time()


# Visualization of the maze
exploration_viz = cv2.resize(inflated_maze_vis.copy(), (inflated_maze_vis.shape[1] * 5, inflated_maze_vis.shape[0] * 5), interpolation=cv2.INTER_NEAREST)

# Mark start and goal
sx, sy, _ = start_pose # start position
gx, gy, _ = goal_pose  # goal position
cv2.rectangle(exploration_viz, (sx * 5, (map_height_mm - 1 - sy) * 5), (sx * 5 + 5, (map_height_mm - 1 - sy) * 5 + 5), (0, 0, 255), -1)
cv2.rectangle(exploration_viz, (gx * 5, (map_height_mm - 1 - gy) * 5), (gx * 5 + 5, (map_height_mm - 1 - gy) * 5 + 5), (255, 0, 255), -1)

# Visualize exploration
for node in explored_nodes:
    x, y, _ = node # current node
    # Convert to pixel coordinates for visualization
    xf, yf = int(x), int(y)
    px = xf * 5
    py = (map_height_mm - 1 - yf) * 5
    # Draw the explored node
    cv2.rectangle(exploration_viz, (px, py), (px + 5, py + 5), (173, 216, 230), -1)
    cv2.imshow('Exploration', exploration_viz)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Visualize final path
if path != "None":
    for node in path:
        x, y, _ = node
        xf, yf = int(x), int(y)
        px = xf * 5
        py = (map_height_mm - 1 - yf) * 5
        cv2.rectangle(exploration_viz, (px, py), (px + 5, py + 5), (0, 255, 0), -1)
        cv2.imshow('Exploration', exploration_viz)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Press any key to exit")
print(f"A* Execution Time: {end_time - start_time:.4f} seconds")
print(f"Nodes Explored: {len(explored_nodes)}")
if path == "None":
    print("No path found")
else:
    print(f"Path found with {len(path)} steps")


print("generating video animation...")
# ---------- Create Video Animation ----------
# Increasing frame rate for faster playback
fps = 60  # Higher value = faster video

# Create video writer
output_video = "A_star_enpm661.mp4"
frame_size = (inflated_maze_binary.shape[1] * 5, inflated_maze_binary.shape[0] * 5)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

# Prepare canvas
canvas = cv2.resize(inflated_maze_vis, frame_size, interpolation=cv2.INTER_NEAREST)
canvas_copy = canvas.copy()

# Pre-render frames in batches for speed
batch_size = 20  # Process multiple nodes per frame

# Initial frame with just the maze
out.write(canvas_copy)

# Mark explored nodes (in batches)
for i in range(0, len(explored_nodes), batch_size):
    batch = explored_nodes[i:i+batch_size]
    for node in batch:
        x, y, _ = node
        px = int(x) * 5
        py = (map_height_mm - 1 - int(y)) * 5
        if (x, y, _) == start_pose or (x, y, _) == goal_pose:
            continue
        cv2.rectangle(canvas_copy, (px, py), (px + 5, py + 5), (173, 216, 230), -1)
    # Write frame after processing a batch
    out.write(canvas_copy)

# Mark the entire path at once
if path != None:
    for i in range(0, len(path), batch_size//2):  # Smaller batches for path for smoother animation
        batch = path[i:i+batch_size//2]
        for node in batch:
            x, y, _ = node
            px = int(x) * 5
            py = (map_height_mm - 1 - int(y)) * 5
            if (x, y, _) == start_pose or (x, y, _) == goal_pose:
                continue
            cv2.rectangle(canvas_copy, (px, py), (px + 5, py + 5), (0, 255, 0), -1)
        # Write frame after processing a batch
        out.write(canvas_copy)

# Mark start and goal at end
sx, sy, _ = start_pose
gx, gy, _ = goal_pose
cv2.rectangle(canvas_copy, (sx * 5, (map_height_mm - 1 - sy) * 5), (sx * 5 + 5, (map_height_mm - 1 - sy) * 5 + 5), (0, 0, 255), -1)
cv2.rectangle(canvas_copy, (gx * 5, (map_height_mm - 1 - gy) * 5), (gx * 5 + 5, (map_height_mm - 1 - gy) * 5 + 5), (255, 0, 255), -1)
out.write(canvas_copy)

# Add final frame with start and goal for a few seconds at the end
for _ in range(fps * 2):  # Hold for 2 seconds
    out.write(canvas_copy)

# Release video writer
out.release()
print(f"A_Star Animation Saved as: {output_video}")
print(f"A_star Execution Time: {end_time - start_time:.4f} seconds")
print(f"Nodes Explored: {len(explored_nodes)}")
print("Video animation complete.")
cv2.destroyAllWindows()
