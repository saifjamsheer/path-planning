#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:35:20 2020

@author: Saif
"""

import math
import numpy as np
import matplotlib.pyplot as plt

class Assist: 
    """
    Assist geometry and dynamics specifications.
    """

    def __init__(self):
        # Assist specifications
        self.max_vel = 2.8  # [m/s]
        self.min_vel = -2.8  # [m/s]
        self.max_ang_vel = 110.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 1.0  # [m/s^2]
        self.max_ang_accel = 110.0 * math.pi / 180.0  # [rad/s^2]
        self.vel_reso = 0.03  # [m/s]
        self.ang_vel_reso = 0.3 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.time_to_predict = 1.0  # [s]
        self.alpha = 0.3
        self.gamma = 1.0
        self.beta = 1.0

        self.radius = 0.6  # [m] Max radius of Assist for collision avoidance 

def waypoints(path):
    """
    Returns a list of waypoints based on the global path.

    path: The initial path generated by the global planner.
    """
    
    wp = [[i[1], i[0]] for i in path]
    wp = wp[1:]
    waypoints = np.array(wp)
    
    return waypoints

def diff_drive_model(x, u, dt):
    """
    Differential drive motion model of the Assist.

    x: Array consisting of the initial state parameters for the Assist model.
    u: Array consisting of the initial velocities of the Assist model.
    dt: Time interval in seconds.
    """
    
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[2] += u[1] * dt
    x[3] = u[0]
    x[4] = u[1]

    return x

def dynamic_window(x, assist):
    """
    Calculating the dynamic window based on the current state
    of the Assist, x
    
    x: Array consisting of the initial state parameters for the Assist model
    assist: Assist model.
    """

    # Dynamic window based on Assist specs
    Va = [assist.min_vel, assist.max_vel,
          -assist.max_ang_vel, assist.max_ang_vel]

    # Dynamic window based on differential drive model
    Vd = [x[3] - assist.max_accel * assist.dt,
          x[3] + assist.max_accel * assist.dt,
          x[4] - assist.max_ang_accel * assist.dt,
          x[4] + assist.max_ang_accel * assist.dt]

    #  Final dynamic window
    dw = [max(Va[0], Vd[0]), min(Va[1], Vd[1]),
          max(Va[2], Vd[2]), min(Va[3], Vd[3])]

    return dw

def trajectories(init_x, v, w, assist):
    """
    Predicting the trajectory of the Assist based on the velocities 
    and initial state.

    init_x: Array consisting of the initial state parameters for the 
    Assist model.
    v: Linear velocity in m/s.
    w: Angular velocity in rad/s.
    assist: Assist model.
    """

    x = np.array(init_x)
    trajectory = np.array(x)
    time = 0
    while time <= assist.time_to_predict:
        x = diff_drive_model(x, [v, w], assist.dt)
        trajectory = np.vstack((trajectory, x))
        time += assist.dt

    return trajectory

def comp_trajectory_and_velocity(x, dw, assist, goal, obs):
    """
    Computing the ideal velocity and trajectory based on the 
    objective function.

    x: Array consisting of the state parameters for the 
    Assist model.
    dw: Dynamic window.
    assist: Assist model.
    goal: Coordinates of the goal waypoint.
    obs: List of the positions of any obstacles.
    """

    init_x = x[:]
    cost_min = float("inf")
    ideal_u = [0.0, 0.0]
    ideal_trajectory = np.array([x])

    # Evaluating all trajectories with all potential inputs within the dynamic 
    # window
    for v in np.arange(dw[0], dw[1], assist.vel_reso):
        for w in np.arange(dw[2], dw[3], assist.ang_vel_reso):

            trajectory = trajectories(init_x, v, w, assist)

            # Calculating the costs for the objective function
            heading_cost = assist.alpha * comp_heading_cost(trajectory, goal)
            ob_cost = assist.beta * comp_obstacle_cost(trajectory, obs, assist)
            velocity_cost = assist.gamma * (assist.max_vel - trajectory[-1, 3])

            cost_total = heading_cost + velocity_cost + ob_cost

            # search minimum trajectory
            if cost_min >= cost_total:
                cost_min = cost_total
                ideal_u = [v, w]
                ideal_trajectory = trajectory

    return ideal_u, ideal_trajectory

def dwa(x, assist, goal, obs):
    """
    Implementing the repeatable dynamic window approach.

    x: Array consisting of the initial state parameters for the Assist model.
    assist: Assist model.
    goal: Coordinates of the goal waypoint.
    obs: List consisting of the positions of all obstacles.
    """

    dw = dynamic_window(x, assist)

    u, trajectory = comp_trajectory_and_velocity(x, dw, assist, goal, obs)

    return u, trajectory

def comp_obstacle_cost(trajectory, obs, assist):
    """
    Computing the obstacle clearance cost for the objective function.

    trajectory: Array consisting of the potential trajectories of the Assist.
    obs: List of the positions of any obstacles.
    assist: Assist model.
    """
    obs_x = obs[:, 0]
    obs_y = obs[:, 1]
    dx = trajectory[:, 0] - obs_x[:, None]
    dy = trajectory[:, 1] - obs_y[:, None]
    dist = np.hypot(dx, dy)

    if (dist <= assist.radius).any():
        return float("Inf")

    min_dist = np.min(dist)
    return 1.0 / min_dist 

def comp_heading_cost(trajectory, goal):
    """
    Computing the target heading cost for the objective function.

    trajectory: Array consisting of the potential trajectories of the Assist.
    goal: Coordinates of the goal waypoint.
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost

def plot_assist(x, y, theta, assist):  
    """
    Plotting a circle representing the Assist.

    x: x-coordinate of the Assist in meters.
    y: y-coordinate of the Assist in meters.
    theta: direction of the Assist in radians.
    assist: Assist model.
    """

    circle = plt.Circle((x, y), assist.radius, color="#0433FF")
    plt.gcf().gca().add_artist(circle)
    outer_x, outer_y = (np.array([x, y]) +
                       np.array([np.cos(theta), np.sin(theta)]) * 
                       assist.radius)
    plt.plot([x, outer_x], [y, outer_y], "-b")