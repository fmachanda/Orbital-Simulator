from orbit import Orbiter, Point
from orbit import earth, sun, moon
from math import pi

"""
Add satellites here using the following arguemnts of the Orbiter class

    Orbiter(
        reference body,
        periapsis (m),
        apoapsis (m),
        direction (1 for ccw, -1 for cw)
        true anomaly (rads),
        argument of periapsis (rads)
    )

"""
satellites = [
    Orbiter(earth, 1e7, 1e7),
    Orbiter(earth, 1e7, 5e7),
    Orbiter(earth, 1e7, 1e8),
]
