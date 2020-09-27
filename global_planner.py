#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:27:27 2020

@author: Saif
"""

import math
import heapq
import numpy as np
            
def line_of_sight(s, n, grid):
    """
    Determines whether there is line of sight between two
    points on a grid.

    s: Coordinates of initial point.
    n: Coordinates of test point.
    grid: A numpy array of 1's and 0's where 1's represent the
    positions of obstacles and 0's represent free spaces.
    """
    
    x0 = s[0]
    y0 = s[1]
    x1 = n[0]
    y1 = n[1]
    
    n = 4  # Steps per unit distance
    dxy = (np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)) * n
    i = np.rint(np.linspace(x0, x1, dxy)).astype(int)
    j = np.rint(np.linspace(y0, y1, dxy)).astype(int)
    has_collision = np.any(grid[i, j])
    los = not has_collision
    
    return los

def post_smooth_path(path, grid):
    """
    Eliminates any unnecessary waypoints on the inputted path to
    reduce the path length.

    path: List consisting of waypoints of the generated path.
    grid: A numpy array of 1's and 0's where 1's represent the
    positions of obstacles and 0's represent free spaces.
    """
    
    s = [i for i in path]
    t = s
    j = 0
    t[j] = s[0]
    
    length = len(s)
    
    for i in range(1, length-1):
        if not line_of_sight(t[j], s[i+1], grid):
            j = j+1
            t[j] = s[i]
    
    j = j+1
    t[j] = s[-1]
    
    t = t[:j+1]
    
    return t

def heuristic(start, goal, func):
    """
    Calculates the distance between the start and goal positions
    based on the selected heuristic.

    start: Coordinates of start position.
    goal: Coordinates of goal position.
    func: Integer to determine which heuristic function is used.
    """

    # Manhattan distance
    if func == 0:
        D = 1
        
        dx = abs(start[0] - goal[0])
        dy = abs(start[1] - goal[1])
        
        return D * (dx + dy)
    
    # Chebyshev (diagonal) distance
    elif func == 1:
        D = 1
        D2 = 1
        
        dx = abs(start[0] - goal[0])
        dy = abs(start[1] - goal[1])
        
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
    
    # Octile (diagonal) distance
    elif func == 2:
        D = 1
        D2 = math.sqrt(2)
        
        dx = abs(start[0] - goal[0])
        dy = abs(start[1] - goal[1])
        
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
    
    # Euclidean Distance
    else:
        D = 1
        
        dx = abs(start[0] - goal[0])
        dy = abs(start[1] - goal[1])
        
        return D * math.sqrt(dx * dx + dy * dy)

def reconstruct_path(cameFrom, current):
    """
    Creates a list of coordinates representing the waypoints
    of the path found using the A* algorithm.

    cameFrom: List of the available coordinates to select from.
    current: Coordinates of the goal position.
    """
    
    totalPath = []
    cellsProcessed = 0
    
    while current in cameFrom:
        current = cameFrom[current]
        totalPath.append(current)
        cellsProcessed += 1
    
    print("\n")
    print("Total Cells Processed = " + str(cellsProcessed))
    print("\n")
    
    return totalPath

def AStar(start, goal, grid):
    """
    A* algorithm used to find the shortest path from the start
    position to the goal positon on a grid-based environment.

    start: Coordinates of the start position (x,y).
    goal: Coordinates of the goal position (x, y).
    grid: A numpy array of 1's and 0's where 1's represent the
    positions of obstacles and 0's represent free spaces.
    """
    
    h = 2 # Octile distance heuristic
    
    # Creating the three main lists
    openSet = []
    closedSet = set()
    cameFrom = {}
    
    # Allowing eight possible directions of movement
    neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    G = {} # G score
    F = {} # F score
    
    # Initializing the G and F scores
    G[start] = 0
    F[start] = G[start] + heuristic(start, goal, h)
    
    heapq.heappush(openSet, (F[start], start))
    
    # Repeating the A* process whilst the open list is not empty
    while openSet:
        
        current = heapq.heappop(openSet)[1]
        
        if current == goal:
            # Feasible path is found
            return reconstruct_path(cameFrom, current)
        
        closedSet.add(current)
        
        for new in neighbors:
            
            neighbor = current[0] + new[0], current[1] + new[1]
            tentativeG = G[current] + heuristic(current, neighbor, h)
            
            # Making sure neighbor is within the range (inside the grid)
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:                
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # Neighbor does not fit within the y bounds of the grid
                    continue
            else:
                # Neighbor does not fit within the x bounds of the grid
                continue
            
            # Checking if neighbor has already been traversed or if the tentative
            # cost is greater than the stored cost for that position
            if neighbor in closedSet and tentativeG > G.get(neighbor, 0):
                continue
            
            # Checking if the tenative cost is less than the stored cost for that
            # position or if the position has not been visited yet
            if (tentativeG < G.get(neighbor, 0) or 
               neighbor not in [i[1] for i in openSet]):
                cameFrom[neighbor] = current
                G[neighbor] = tentativeG
                F[neighbor] = G[neighbor] + heuristic(neighbor, goal, h)
                heapq.heappush(openSet, (F[neighbor], neighbor))
       
    # Open set is empty but goal was never reached
    return False