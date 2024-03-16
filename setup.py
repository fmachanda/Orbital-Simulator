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
    Orbiter(earth, 3e7, 3.9e8),
    Orbiter(earth, 3e7, 4e8),
    Orbiter(earth, 3e7, 4.1e8),

    Orbiter(earth, 3e7, 3.9e8, w=pi/6),
    Orbiter(earth, 3e7, 4e8, w=pi/6),
    Orbiter(earth, 3e7, 4.1e8, w=pi/6),

    Orbiter(earth, 3e7, 3.9e8, w=-pi/6),
    Orbiter(earth, 3e7, 4e8, w=-pi/6),
    Orbiter(earth, 3e7, 4.1e8, w=-pi/6),

    # Orbiter(earth, 3e7, 1e10),

]
