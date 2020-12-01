# Dynamic Path Planning for Autonomous Wheelchairs
### Building a Local and Global Planner in Python

For this project, I designed and built a program that plans and adjusts paths in unstructured environments. In order to accomplish this, a local planner is used to plan an initial path, and a global planner adjusts the path as a simulated wheelchair navigates the initial path.

The local planner was designed by coding and then modifying the A* algorithm, whereas the global planner was designed by building dynamic windows throughout the environment and determining potential safe trajectories within these windows. 

Through the use of differential equations, the motion of the wheelchair was simulated.

This project demonstrated proficiency in the following areas:
* Object-oriented programming
* Algorithms design and analysis
* Data structures (heaps and tree-based structures)
* Calculus

In the future, the project will be imported to ROS and simulated in a 3D environment.
