# 3D Drone Racing Simulator

A physics-based 3D drone racing simulator built in Python to explore motion, control systems, and visual simulation of autonomous flight.

## Overview

This project simulates a drone as a point mass moving through 3D space under the influence of gravity, thrust, and aerodynamic drag. A PD controller is used to guide the drone through a sequence of gates while minimizing tracking error.

The system is visualized using Plotly with an interactive 3D animation.

## Key Features

- 3D physics simulation (Newtonian motion)
- PD controller for waypoint navigation
- Gravity, thrust, and drag force modeling
- Motor lag via first-order thrust response
- Gate-based racing system with lap tracking
- Performance metrics (energy, tracking error, completion, score)

## Visualization

The project uses Plotly to render an interactive 3D environment:

- Drone trajectory in 3D space
- Color-coded speed along the flight path
- Real-time force vectors (gravity, thrust, drag, net force)
- Interactive play/pause animation
- Frame slider for scrubbing through time
- Adjustable camera view

## Purpose

The goal of this project is to progressively build toward a full rigid-body drone model, extending from simple point-mass dynamics to full rotational physics (roll, pitch, yaw).

## Future Improvements

- Full rigid-body dynamics (orientation affects motion)
- Quaternion-based rotation model
- More realistic motor and propeller physics
- Improved controller (LQR or MPC)
- Real-time simulation instead of precomputed frames

## How to Run

```bash
python drone_sim.py
