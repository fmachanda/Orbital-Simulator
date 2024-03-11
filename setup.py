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
    Orbiter(moon, 6.4e7, 6.4e7),
]
