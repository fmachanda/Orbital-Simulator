import numpy as np

#region Constants
# Physical constants
G = 6.6743e-11

M_EARTH = 5.972e24
R_EARTH = 6.37814e6
V_EARTH = 2.978e4
SOI_EARTH_SQ = 9.29e8**2

M_SUN = 1.989e30
R_SUN = 6.96340e8

M_MOON = 7.384e22
R_MOON = 1.7374e6
V_MOON = 1.022e3
SOI_MOON_SQ = 6.43e7**2

R_ES = 1.4845e11
R_EM = 3.844e8
#endregion

# Calculation functions
def direction(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    a = np.arctan2(y2-y1, x2-x1)
    return np.cos(a), np.sin(a)

def r_squared(x1: float, y1: float, x2: float, y2: float) -> float:
    return (x1-x2)**2+(y1-y2)**2

def force(m1: float, m2: float, r_squared: float) -> float:
    return G * m1 * m2 / r_squared

# Main classes
class Point:
    def __init__(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, ref: 'Body | None' = None) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ref = ref

    @property
    def mu(self) -> float:
        return 0 if self.ref is None else self.ref.m * G
    
    @property
    def ref_x(self) -> float:
        return 0 if self.ref is None else self.x-self.ref.x
    
    @property
    def ref_y(self) -> float:
        return 0 if self.ref is None else self.y-self.ref.y
    
    @property
    def ref_vx(self) -> float:
        return 0 if self.ref is None else self.vx-self.ref.vx
    
    @property
    def ref_vy(self) -> float:
        return 0 if self.ref is None else self.vy-self.ref.vy

    @property
    def r(self) -> np.ndarray:
        return 0 if self.ref is None else np.array([self.ref_x, self.ref_y, 0])

    @property
    def mag_r(self) -> float:
        return 0 if self.ref is None else np.linalg.norm(self.r)

    @property
    def v(self) -> np.ndarray:
        return 0 if self.ref is None else np.array([self.ref_vx, self.ref_vy, 0])

    @property
    def mag_v(self) -> float:
        return 0 if self.ref is None else np.linalg.norm(self.v)

    @property
    def h(self) -> float:
        return 0 if self.ref is None else self.ref_x*self.ref_vy - self.ref_y*self.ref_vx
    
    @property
    def vec_e(self) -> np.ndarray:
        return 0 if self.ref is None else np.cross(self.v, (np.array([0, 0, self.h])))/self.mu - self.r/self.mag_r

    @property
    def e(self) -> float:
        return 0 if self.ref is None else np.linalg.norm(self.vec_e)

    @property
    def a(self) -> float:
        return 0 if self.ref is None else self.h**2 / (self.mu * (1-self.e**2))
    
    @property
    def apo(self) -> float:
        return 0 if self.ref is None else self.a * (1+self.e)
    
    @property
    def peri(self) -> float:
        return 0 if self.ref is None else self.a * (1-self.e)
    
    @property
    def apo_surf(self) -> float:
        return 0 if self.ref is None else self.a * (1+self.e) - self.ref.radius
    
    @property
    def peri_surf(self) -> float:
        return 0 if self.ref is None else self.a * (1-self.e) - self.ref.radius

    @property
    def w(self) -> float:
        return 0 if self.ref is None else np.arctan2(self.vec_e[1], self.vec_e[0])

class Body(Point):
    count = 0

    def __init__(self, x: float, y: float, m: float, r: float, vx: float = 0, vy: float = 0, name: str | None = None, soi_sq: float = 0.0, ref: 'Body | None' = None) -> None:
        super().__init__(x, y, vx, vy, ref)
        Body.count += 1
        self.m: float = m
        self.radius: float = r
        self.name = name if name is not None else f"Body {Body.count}"
        self.soi_sq = soi_sq
    
    def acceleration(self, body: Point) -> np.ndarray:
        """Acceleration from self on point mass."""
        if (rs:=r_squared(self.x, self.y, body.x, body.y)) < self.radius**2:
            return np.array([0.0, 0.0])
        a = force(self.m, 1, rs)
        i, j = direction(body.x, body.y, self.x, self.y)
        return np.array([a*i, a*j])
    
    def acceleration_map(self, xcoords: np.ndarray, ycoords: np.ndarray) -> np.ndarray:
        MIN_R_SQUARED = self.radius**2
        rs: np.ndarray = r_squared(xcoords, ycoords, self.x, self.y)
        rs = rs.clip(MIN_R_SQUARED, float('inf'))
        a = force(self.m, 1, rs)
        i, j = a*direction(self.x, self.y, xcoords, ycoords)
        i[rs==MIN_R_SQUARED] = 0.0
        j[rs==MIN_R_SQUARED] = 0.0
        return np.array([i, j])
    
class Orbiter(Point):
    def __init__(self, ref: Body, periapsis: float, apoapsis: float, direction: int = 1, f: float = 0.0, w: float = 0.0) -> None:
        self.ref = ref
        assert direction in [1, -1] # ccw=1, cw=-1

        ap = max(apoapsis + ref.radius, periapsis + ref.radius)
        pe = min(periapsis + ref.radius, apoapsis + ref.radius)

        a = (ap + pe)/2
        e = (ap-pe)/(2*a)
        mu = ref.m * G
        mag_r = (a * (1-e**2))/(1+e*np.cos(f))
        mag_v = np.sqrt(mu * (2/mag_r - 1/a))
        h = np.sqrt(mu * a * (1 - e**2))
        fpa = np.arccos(max(-1, min(1, h / (mag_r * mag_v))))
        rx = np.cos(f)
        ry = np.sin(f)
        sx = np.cos(f+np.pi/2)
        sy = np.sin(f+np.pi/2)

        x = rx*mag_r
        y = ry*mag_r
        vx = mag_v * (sx * np.cos(fpa) + rx * np.sin(fpa))
        vy = mag_v * (sy * np.cos(fpa) + ry * np.sin(fpa))

        super().__init__(
            (x*np.cos(w)-y*np.sin(w)) + ref.x,
            (x*np.sin(w)+y*np.cos(w)) + ref.y,
            direction*(vx*np.cos(w)-vy*np.sin(w)) + ref.vx,
            direction*(vx*np.sin(w)+vy*np.cos(w)) + ref.vy,
            ref
        )

# Body creation
sun = Body(0, 0, M_SUN, R_SUN, name="Sun")
earth = Body(0, R_ES, M_EARTH, R_EARTH, vx=-V_EARTH, name="Earth", soi_sq=SOI_EARTH_SQ, ref=sun)
moon = Body(-R_EM+earth.x, 0+earth.y, M_MOON, R_MOON, vy=-V_MOON+earth.vy, vx=earth.vx, name="Moon", soi_sq=SOI_MOON_SQ, ref=earth)