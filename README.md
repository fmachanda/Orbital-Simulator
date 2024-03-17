# Orbital-Simulator

This is a multi-body orbit simulator with a **pygame** interface. It allows for the modeling of more realistic spacecraft trajectories, calculating forces frame-by-frame rather than using two-body ellipses and spheres of influences.

## Requirements

* Python 3.12.0+
    * **pip** modules listed in `requirements.txt`
* MacOS 13.0+

## Installation

1. Clone this repo into the desired directory (change `<directory>` in the commands below as desired).

    ```bash
    git clone https://github.com/fmachanda/Orbital-Simulator.git <directory>
    ```
    ```bash
    cd <directory>/Orbital-Simulator
    ```

2. Ensure you have all required modules with pip.

    ```bash
    python -m pip install -r requirements.txt
    ```

## Satellite Setup

To setup the initial states of the satellites, open `setup.py` with a text editor. You should see some lines of code that create a list of satellites, such as this:

```python
satellites = [
    Orbiter(earth, 1e7, 1e7),
    Orbiter(moon, 1e5, 5e6, f=pi),
    Orbiter(earth, 1e7, 1e8, direction=-1, w=pi/2),
    Orbiter(sun, 1e11, 1e12, f=pi/2, w=3*pi/4),
    ...
]
```

Adding or deleting the `Orbiter` lines within the `satellites` list declaration will create or remove independent, small mass satellites from the simulation.

The `Orbiter` constructors allow the user to input some inital, elliptical orbital parameters for each satellite. This tells the simulator where to put each satellite, how much speed to give it, and which direction it's moving in. If no other celestial bodies were present in the simulator, the satellite would remain in this elliptical orbit forever. However, interactions with the other bodies present in the simulator will immediately start to change this orbit.

The first argument of the `Orbiter` constructor specifies which celestial body the orbit should be constructed in reference to. Again, there is no guarantee that the satellite will continue to orbit this body.

The second argument of the constructor specifies the initial *periapsis* of the orbit in meters.

The third argument of the constructor specifies the initial *apoapsis* of the orbit in meters. These first three arguments are required.

`w` optionally specifies the *argument of periapsis* in radians. For this 2D simulation, this the the counterclockwise angle from the x-axis (an imaginary horizontal ray from the center of the specified body extending to the right) to the point of periapsis. Use `pi` if needed.

`direction` optionally specifies whether the orbit should appear to go counterclockwise (`direction = 1`, which is the default) or clockwise (`direction = -1`).

`f` optionally specifies the *true anomaly* of the inital satellite position in radians.  This does not affect the orbit itself, just where in the orbit the satellite will start. Use `pi` if needed.

## Running the Simulation

Run `run.py` to open the simulator.

```bash
python run.py
```

A black sqaure window should open with the simulation paused. You should see the Earth (centered) and Moon, as well as any satellites you created (if they are in frame).

### Controls

Action|Control(s)
---|---
Zoom in|**=**
||*Scroll up*
Zoom out|**-**
||*Scroll down*
Select Earth|**e**
Select Moon|**m**
Select Sun|**s**
Select Satellite 1|**1**
Select Satellite 2|**2**
||...
Show orbit info|**p**
Map current field|**d**
Pause/Unpause|**Spacebar**
Step one calculation|**Up Arrow**
Step one frame|**Right Arrow**
Increase **DT**|**u**
Decrease **DT**|**j**
Increase calculations/frame|**i**
Decrease calculations/frame|**i**
Reset calculations/frame|**,**
Increase slowdown|**o**
Decrease slowdown|**l**
Remove slowdown|**.**

### Time management and DT
In the bottom right corner, you should see two numbers. The first is **DT**, which tells you how much time passes in between each calculation performed by the simulator. The second tells you how many of these calculations are performed in between each frame refresh. A higher **DT** will let the simulation run faster, but with a lower precision (and vice versa). If you have a very fast satellite (e.g., with a low periapsis), **DT** should be low. Setting **DT** too high will cause innacuracies (the Moon may even escape Earth orbit).

---
*README.md modified 11 March 2024*