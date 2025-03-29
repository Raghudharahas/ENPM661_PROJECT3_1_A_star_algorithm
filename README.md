ENPM661_Project03: A* Pathfinding Algorithm with Orientation

Overview:
This project implements the A* pathfinding algorithm for a differential drive robot with orientation, navigating a maze containing obstacles. The maze is generated using semi-algebraic equations that form the letters "ENPM661". The goal is to find an optimal path from a start position to a goal position while avoiding obstacles and ensuring sufficient clearance.

Dependencies:
Ensure you have the following Python libraries installed before running the code:
1. numpy
2. opencv-python (cv2)
3. time
4. heapq (used for priority queue management)
5. math
6. collections

To install the dependencies, run:
pip install numpy opencv-python

How to Run the Code:
1. Run the script in your preferred Python environment:
   - Download the file and run it in your Python IDE.
   - Or clone the repository to your local machine and run the script.

2. Enter start and goal coordinates when prompted:
   - Input format: x y θ (e.g., 10 15 30)
   - The program ensures that the points are valid (not inside obstacles or inflated regions).

3. Watch the visualization:
   - The program visualizes the maze, the explored nodes, and the final path.
   - A video file (A_star_enpm661.mp4) will be generated, showing the exploration process and the optimal path found by the A* algorithm.

Input:
- Start and goal coordinates in the format x y θ:
   - x: x-coordinate (column index)
   - y: y-coordinate (row index)
   - θ: robot orientation in degrees (must be a multiple of 30°)
- The grid is represented as a 100x300 matrix where:
   - 0: Free space
   - 1: Obstacles
   - Blue border: Inflated obstacle regions (clearance included)

Output:
- Visualization of the explored nodes and the optimal path.
- Execution time and the number of explored nodes are displayed.
- A recorded animation of the process (.mp4 file) is generated to visualize the exploration and pathfinding.

Algorithms Used:
- A* Pathfinding Algorithm: 
   - A* is a search algorithm that finds the shortest path from the start position to the goal position while considering both position and orientation.
   - It uses a priority queue to explore the lowest-cost path first.
   - The algorithm incorporates obstacle avoidance, robot clearance, and orientation constraints.

   The core steps of the A* algorithm are:
   - Goal check: Determines if the robot has reached the goal within a set threshold of position and orientation.
   - Node expansion: At each step, the algorithm explores neighboring nodes (potential robot poses) by applying specific movements and checking the cost (distance + heuristic to goal).
   - Path reconstruction: Once the goal is reached, the algorithm traces back the path from the goal to the start.

Notes:
- The maze is drawn using semi-algebraic equations to form the letters "E", "N", "P", "M", "6", and "1", which are placed within a grid. The grid is 100 mm by 300 mm, with 5 mm boundary obstacles.
- The robot is modeled as a differential drive system, which requires consideration of orientation (theta) during path planning.
- A visual representation of the maze is created, showing obstacles, inflated regions, explored nodes, and the path taken by the robot.
- The program generates a video file (A_star_enpm661.mp4) that shows the robot's exploration and final path.

Visualization:
- The visualization includes:
   - Original Maze: Shows the maze layout with obstacles and boundary regions.
   - Inflated Maze: Displays the obstacle clearance regions (marked in blue) along with obstacles.
   - Explored Nodes: These are nodes that have been visited during the pathfinding process.
   - Optimal Path: The final path found by the A* algorithm, shown in green.

How the Maze is Created:
The maze is constructed using semi-algebraic equations to draw letters and digits:
- E: Formed using a vertical spine and horizontal bars.
- N: Formed using vertical bars and a diagonal line.
- P: Formed using a vertical spine and a semi-circular loop.
- M: Formed using vertical bars and two diagonal lines.
- 6: Formed using a circle and ellipse arc.
- 1: A simple vertical line.

Robot Configuration:
- Clearance: The space around the robot that should be kept clear of obstacles. Default clearance is 2 mm.
- Robot Radius: The radius of the robot (default: 5 mm).
- Step Size: The distance the robot moves in each step of the A* search. Default: 1 mm.

The program prompts the user for these parameters and calculates the required clearance and robot movement steps based on input.

Key Commands:
- Press any key to proceed through different visualization stages.
- Press ESC to exit early during animations.
- Video Animation: The process is recorded as a video (A_star_enpm661.mp4) that can be viewed after the program finishes.

Example Run:
1. The user is prompted to enter the start and goal coordinates (e.g., 10 15 30 for the start and 50 75 90 for the goal).
2. The maze is displayed, followed by the inflated maze with obstacle clearance regions.
3. The A* algorithm runs, showing the exploration of nodes and the construction of the final path.
4. The video animation is saved and can be viewed later.

Conclusion:
This project demonstrates the application of the A* algorithm for pathfinding in a maze with obstacles and orientation constraints. The generated video file allows for an easy-to-follow visualization of the robot's movement and pathfinding process.

Regards,  
Raghu Dharahas
