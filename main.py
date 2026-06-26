import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# Constants
g = 9.81  # Gravity (m/s²)
rho = 1.225  # Air density (kg/m³)

Cd_xy = 1.0  # Drag coefficient xy
Cd_z = 0.1  # Drag coefficient z
A_xy = 0.02  # Cross-sectional area xy (m²) 
A_z = 0.03  # Cross-sectional area z (m²) 

dt = 0.05  # Time step (s)
T = 10  # Total simulation time (s)
steps= int(T / dt)

# Physics
def compute_forces(state, thrustVector, mass):
    # Compute forces acting on the drone
    velocity = state[3:6]

    fGravity = np.array([0, 0, -mass * g])
    fThrust = np.array(thrustVector)

    fDrag = np.array([
        -0.5 * rho * Cd_xy * A_xy * velocity[0] * abs(velocity[0]),
        -0.5 * rho * Cd_xy * A_xy * velocity[1] * abs(velocity[1]),
        -0.5 * rho * Cd_z * A_z * velocity[2] * abs(velocity[2])
    ])

    return (
        fGravity + fThrust + fDrag,
        fGravity,
        fThrust,
        fDrag
    )

def simulate_drone(gates, mass = 1.0, k_p = 300.0, k_d = 50.0):
    # Original state: [x, y, z, vx, vy, vz]
    state = np.zeros(6)

    gateNum = 0
    positions = []
    orientations = []
    gravity_history = []
    thrust_history = []
    drag_history = []
    net_force_history = []

    # Angular state
    rolls = [] # Tilt left/right
    pitches = [] # Tilt forward/back
    yaws = [] # Heading direction in XY plane

    maxThrust = 20.0
    motorTau = 0.08
    currentThrust = np.zeros(3)
    total_energy = 0.0
    total_tracking_error = 0.0
    vel_change = 0.0
    prev_velocity = np.zeros(3)

    gates_passed = set()

    current_time = 0.0
    lap_time = T

    for step in range(steps):
        current_time += dt
        pos = state[0:3]  # (x, y, z)
        velocity = state[3:6]

        targetPos = gates[gateNum] # Next gate position
        errorPos = targetPos - pos
        total_tracking_error += np.linalg.norm(errorPos) * dt
        errorVel = -velocity  # Want velocity near zero at the gate

        # Gate switching
        # Determine if magnitude of error is less than 1
        if np.linalg.norm(errorPos) < 1.0:
            gates_passed.add(gateNum)

            gateNum += 1
    
            # Check lap completion
            if gateNum >= len(gates):
                lap_time = current_time
                gateNum = 0
        
        # PD controller
        desiredAcceleration = (
            (k_p * errorPos + k_d * errorVel) / mass
            + np.array([0, 0, g])
        )

        # Force command
        desiredThrust = mass * desiredAcceleration

        # First order motor model
        currentThrust += (
            desiredThrust - currentThrust
        ) * dt / motorTau

        thrustVector = currentThrust

        thrustMag = np.linalg.norm(thrustVector)

        # Normalize vector
        if thrustMag > maxThrust:
            currentThrust = (currentThrust / thrustMag) * maxThrust

            thrustVector = currentThrust
            thrustMag = maxThrust

        total_energy += thrustMag ** 2 * dt

        # Physics update
        (
            netForce,
            fGravity,
            fThrust,
            fDrag
        ) = compute_forces(state, thrustVector, mass)
        acceleration = netForce / mass

        gravity_history.append(fGravity.copy())
        thrust_history.append(fThrust.copy())
        drag_history.append(fDrag.copy())
        net_force_history.append(netForce.copy())

        state[3:6] += acceleration * dt
        velocity = state[3:6]

        if thrustMag > 1e-6: 
            forward = currentThrust / thrustMag
        
        else: 
            forward = np.array([0.0, 0.0, 1.0])
        
        orientations.append(forward.copy())

        eps = 1e-6

        roll = np.arctan2(thrustVector[1], thrustVector[2] + eps)
        pitch = np.arctan2(-thrustVector[0], thrustVector[2] + eps)
        yaw = np.arctan2(velocity[1], velocity[0] + eps)

        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)

        vel_change += np.linalg.norm(velocity - prev_velocity)

        prev_velocity = velocity.copy()

        state[0:3] += state[3:6] * dt

        positions.append(state[0:3].copy())

    # Metrics
    completion_ratio = len(gates_passed) / len(gates)

    # Update scoring
    score = (
        2.0 * lap_time +
        0.5 * total_energy +
        3.0 * total_tracking_error +
        5.0 * (1 - completion_ratio) +
        1.0 * vel_change
    )   

    return (
        np.array(positions),
        np.array(orientations),
                
        np.array(gravity_history),
        np.array(thrust_history),
        np.array(drag_history),
        np.array(net_force_history),    
        np.array(rolls),
        np.array(pitches),
        np.array(yaws),

        {
            "lap_time": lap_time,
            "energy": total_energy,
            "tracking_error": total_tracking_error,
            "velocity_change": vel_change,
            "completion": completion_ratio,
            "score": score
        }
    )

def create_force_trace(pos, color, name):
    return go.Scatter3d(
        x = [pos[0], pos[0]],
        y = [pos[1], pos[1]],
        z = [pos[2], pos[2]],
        mode = "lines",
        line = dict(width = 8, color = color),
        name = name
    )

def create_force_vector(pos, force, scale, color):
    return go.Scatter3d(
        x = [pos[0], pos[0] + force[0] * scale],
        y = [pos[1], pos[1] + force[1] * scale],
        z = [pos[2], pos[2] + force[2] * scale],
        mode = "lines",
        line = dict(width = 8, color = color),
    )

if __name__ == "__main__":
    gates = [
        np.array([5, 0, 5]),
        np.array([10, 10, 8]),
        np.array([15, 5, 12]),
        np.array([20, 15, 6]),
    ]

    (
        positions,
        orientations,

        gravity_history,
        thrust_history,
        drag_history,
        net_force_history,

        rolls,
        pitches,
        yaws,

        metrics
    ) = simulate_drone(
        gates,
        mass = 1.0,
        k_p = 35.0, 
        k_d = 12.0
    )

    print(metrics)

    # Plotting
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    fig = go.Figure()

    speed = np.linalg.norm(np.diff(positions, axis = 0), axis = 1)
    speed /= speed.max()
    
    colors = sample_colorscale("Turbo", speed)

    for i in range(len(x) - 1):
        fig.add_trace(go.Scatter3d(
            x = [x[i], x[i + 1]],
            y = [y[i], y[i + 1]],
            z = [z[i], z[i + 1]],
            mode = "lines",
            line = dict(width = 6, color = colors[i]),
            showlegend = False
        ))

    # Gates
    gate_x = [g[0] for g in gates]
    gate_y = [g[1] for g in gates]
    gate_z = [g[2] for g in gates]
    fig.add_trace(go.Scatter3d(
        x = gate_x,
        y = gate_y,
        z = gate_z,
        mode = "markers",
        marker = dict(size = 6, color = "red"),
        name = "Gates"
    ))
    
    fig.add_trace(create_force_trace(positions[0], "blue", "Gravity"))
    fig.add_trace(create_force_trace(positions[0], "green", "Thrust"))
    fig.add_trace(create_force_trace(positions[0], "orange", "Drag"))
    fig.add_trace(create_force_trace(positions[0], "purple", "Net Force"))

    num_leading_traces = (len(x) - 1) + 1

    animated_traces_indices = list(range(num_leading_traces, num_leading_traces + 6))
    
    maxForce = max(
        np.max(np.linalg.norm(gravity_history, axis = 1)),
        np.max(np.linalg.norm(thrust_history, axis = 1)),
        np.max(np.linalg.norm(drag_history, axis = 1)),
        np.max(np.linalg.norm(net_force_history, axis = 1)),
    )

    forceScale = 3.0 / maxForce

    # Animation frames
    frames = []

    for i in range(len(positions)):
        pos = positions[i]

        gravity = gravity_history[i]
        thrust = thrust_history[i]
        drag = drag_history[i]
        net_force = net_force_history[i]
        forward = orientations[i]

        frames.append(
            go.Frame(
                name = str(i),
                data = [
                    # Drone
                    go.Scatter3d(
                        x = [pos[0]],
                        y = [pos[1]],
                        z = [pos[2]],
                        mode = "markers",
                        marker = dict(size = 8, symbol = "circle", color = "black")
                    ),

                    create_force_vector(pos, gravity, forceScale, "blue"),
                    create_force_vector(pos, thrust, forceScale, "green"),
                    create_force_vector(pos, drag, forceScale, "orange"),
                    create_force_vector(pos, net_force, forceScale, "purple"),

                ],
                traces = [1, 3, 4, 5, 6]
            )
        )
        
    fig.frames = frames
              

    # Controls
    fig.update_layout(
        title = "Drone Sim",
        scene = dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',

            aspectmode = 'data',

            camera = dict(
                eye = dict(
                    x = -1.5,
                    y = 1.5,
                    z = 1.2
                )
            )
        ),
        sliders=[{
            "steps": [
                {
                    "method": "animate",
                    "label": str(i),
                    "args": [
                        [str(i)],
                        {
                            "mode": "next",
                            "frame": {"duration": 5,"redraw": True},
                            "transition": {"duration": 0}
                        }
                    ]
                }
                for i in range(len(frames))
            ]
        }],
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.05,
            "y": 0.95,
            "xanchor": "left",
            "yanchor": "top",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 20, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }
                    ]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                }
            ]
        }]

    )   

    fig.write_html("index.html", auto_open = True)
