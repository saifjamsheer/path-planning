#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:23:41 2020

@author: Saif
"""

import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import global_planner
import local_planner

def length(path):
    """
    Calculates the total path length of an inputted path.

    path: A list consisting of the waypoints of the path
    """
    total_length = 0
    
    for i in range(len(path) - 1):
        
        dx = abs(path[i+1][0] - path[i][0])
        dy = abs(path[i+1][1] - path[i][1])
        
        length = math.sqrt(dx**2 + dy**2)
        
        total_length += length
    
    return total_length

def min_clearance(path, obstacles):
    """
    Calculates the minimum clearance between an inputted path
    and any obstacles within an environment.

    path: A list consisting of the waypoints of the path
    obstacles: A list consisting of the positions of the obstacles
    """
    
    distances = []
    
    for i in path:
        for j in obstacles:
            dist = global_planner.heuristic(i, j, 3)
            distances.append(dist)
    
    return min(distances)

def convert_to_array(file_path):
    """
    Converts an excel file to a numpy array to be used as
    the test envionrment.

    file_path: Path of the file
    """
    
    grid_data = pd.read_excel(file_path, header=None)
    grid = grid_data.values
    
    return grid

def obstacles(grid):
    """
    Returns a list of the positions of all obstacles within
    the inputted grid.

    grid: A numpy array of 1's and 0's where 1's represent
    the positions of obstacles.
    """
    
    obs_array = np.where(grid == 1)
    obs_list = tuple(zip(*obs_array))
    obs = [list(i) for i in obs_list]
    obs = [[i[1], i[0]] for i in obs_list]
    obs = np.array(obs)
    
    return obs

def main():
    
    # Known and partially known maps of the environment
    data_known = 'Maps/Plot/M12.xlsx'
    data_partial = 'Maps/Dynamic/M12.xlsx'
    map_known = convert_to_array(data_known)
    map_partial = convert_to_array(data_partial)
    
    #####################
    # A* IMPLEMENTATION #
    #####################
    
    # s = (2, 2)
    # g = (25, 42)
    s = (45, 3)
    g = (3, 46)
    
    t = time.time()
    
    path = global_planner.AStar(s, g, map_known)
    
    if path:
        path = [g] + path + [s]
        path = path[::-1]
    
    smooth_path = global_planner.post_smooth_path(path, map_known)

    # Calculating the computational time required to find a solution
    elapsed = time.time() - t
    
    # Calculating the path length and minimum clearance of the initial path
    path_length = length(path)
    obs = obstacles(map_known)
    clearance = min_clearance(smooth_path, obs)
    
    print("\n")
    print("Run Time = " + str(elapsed))
    print("Total Path Length = " + str(path_length))
    print("Minimum Clerance = " + str(clearance))
    print("\n")
    
    ###############
    # A* PLOTTING #
    ###############
    
    x_wayps = []
    y_wayps = []
    print(smooth_path)

    for i in (range(0,len(smooth_path))):
    
        x_1 = smooth_path[i][0]
        y_1 = smooth_path[i][1]
        x_wayps.append(x_1)
        y_wayps.append(y_1)
        
    f, ax = plt.subplots(figsize=(5,5))
    
    ax.imshow(map_partial, cmap=plt.cm.binary)
    for i in (range(0,len(smooth_path))):
        ax.scatter(smooth_path[i][1], smooth_path[i][0],  marker = ".", color = "#FFBF00", 
               s = 50)
    
    ax.scatter(s[1],s[0], marker = ".", color = "#0433FF", 
               s = 150)
    ax.scatter(g[1],g[0], marker = ".", color = "#00F900", 
               s = 150)
    
    
               
    ax.set_xlabel('X [meters]')
    ax.set_ylabel('Y [meters]')
    
    # ax.plot(y_wayps,x_wayps, color = "#D35A37")
    plt.gca().invert_yaxis()
    
    plt.show()
    
    ######################
    # DWA IMPLEMENTATION #
    ######################
    
    # Initial state of the Assist 
    # [x (m), y (m), theta (rad), v (m/s), dtheta (rad/s)]
    # x = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
    x = np.array([3.0, 45.0, -math.pi/2.0, 0.0, 0.0])

    # Array with coordinates of obstacles 
    obs = obstacles(map_partial)
    
    # Array with coordinates of goal waypoints
    goal = local_planner.waypoints(smooth_path)
    
    # Input configuration
    assist = local_planner.Assist()
    trajectory = np.array(x)
    n = 0
    f, ax = plt.subplots(figsize=(6.2,6.2))
    
    # Repeatedly implementing the DWA until the goal is reached
    while n < len(goal):
        u, predic_traj = local_planner.dwa(x, assist, goal[n], obs)
        x = local_planner.diff_drive_model(x, u, assist.dt)
        trajectory = np.vstack((trajectory, x))

        plt.cla()
        plt.grid(True)
        plt.axis("equal")
        # Plotting the positions of the obstacles
        plt.plot(obs[:, 0], obs[:, 1], "sk")
        # Plotting the position of the goal waypoint
        if n != (len(goal) - 1):
            plt.plot(goal[n][0], goal[n][1], "o", color="#FFBF00", markersize=4)
        else:
            plt.plot(goal[n][0], goal[n][1], "o", color="#00F900", markersize=5)
        # Plotting the Assist
        local_planner.plot_assist(x[0], x[1], x[2], assist)
        # Plotting the predicted trajectory of the Assist
        plt.plot(predic_traj[:, 0], predic_traj[:, 1], "-", color="#D35A37")
        plt.pause(0.001)
        
        # Determining if the goal waypoint is reached
        target_heading = math.hypot(x[0] - goal[n][0], x[1] - goal[n][1])
        if n != (len(goal) - 1):
            if target_heading <= assist.radius*5:
                n += 1
        else:
            if target_heading <= assist.radius:
                n += 1
    
    plt.close()

    # Getting the waypoints of the final trajectory
    trj = trajectory[:, 0:2]
    
    # Calculating the path length and minimum clearance of the final path
    local_length = length(trj)
    local_clearance = min_clearance(trj, obs)
    
    print("\n")
    print("Total Path Length = " + str(local_length))
    print("Minimum Clerance = " + str(local_clearance))
    print("\n")
    
    ################
    # DWA PLOTTING #
    ################
    
    f, ax = plt.subplots(figsize=(5,5))
    
    ax.imshow(map_partial, cmap=plt.cm.binary)
    ax.scatter(s[1],s[0], marker = ".", color = "#0433FF", 
               s = 150)
    ax.scatter(g[1],g[0], marker = ".", color = "#00F900", 
               s = 150)
    ax.set_xlabel('X [meters]')
    ax.set_ylabel('Y [meters]')
    
    plt.gca().invert_yaxis()
    plt.plot(trajectory[:, 0], trajectory[:, 1], "-", color="#D35A37")
    
    plt.show()
    
if __name__ == '__main__':
    main()