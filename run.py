import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pygame
from pygame.locals import *
import time

from orbit import sun, earth, moon
from orbit import Orbiter, Body
from orbit import Point, G

#region Constants
VERSION = 1.0

# Calculation settings
EARTH = True
MOON = True
SUN = True

# Graphics Settings
WIDTH_P = HEIGHT_P = 600
WIDTH_M = HEIGHT_M = 1e9
MIN_M = 2e7
MAX_M = 5e11
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
SPRITE_SCALE = 1
FONT_SIZE = 12
LINE_SPACE = 6
FONT_COLOR = (255, 255, 255)
TEXT_X = 20
TEXT_Y = 20
ORBIT_COLOR = (255, 255, 255)

DT = 32 # Seconds per calculation
T_RES = 64 # Calculations per frame
DELAY = 0 # Added time buffer

# Plotting settings
X_RES = 500
Y_RES = 500
#endregion

# Satellite creation
from setup import satellites

INFO = False

# Graphics functions
def zoom_out() -> None:
    global WIDTH_M, HEIGHT_M
    WIDTH_M = min(WIDTH_M*1.2, MAX_M)
    HEIGHT_M = min(HEIGHT_M*1.2, MAX_M)

def zoom_in() -> None:
    global WIDTH_M, HEIGHT_M
    WIDTH_M = max(WIDTH_M/1.2, MIN_M)
    HEIGHT_M = max(HEIGHT_M/1.2, MIN_M)

# Plotting functions
def display_amap(xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    xspace = np.linspace(xmin, xmax, X_RES)
    yspace = np.linspace(ymin, ymax, Y_RES)
    xcoords, ycoords = np.meshgrid(xspace, yspace)

    amap = np.linalg.norm(EARTH*earth.acceleration_map(xcoords, ycoords) + MOON*moon.acceleration_map(xcoords, ycoords) + SUN*sun.acceleration_map(xcoords, ycoords), axis=0)
    
    plt.pcolormesh(xcoords, ycoords, amap, norm=LogNorm())
    plt.colorbar()
    plt.axis('square')
    plt.show()

# Simulation functions

in_sun = []
in_moon = []
in_earth = []

x: np.ndarray = Point.state[:, 0:2]
d: np.ndarray = x[:Body.count, None, :] - np.broadcast_to(x, (Body.count,) + x.shape)
r: np.ndarray = np.linalg.norm(d, axis=2)

for i in np.where(r[1, Body.count:] > earth.soi)[0]:
    in_sun.append(satellites[i])

for i in np.where(r[2, Body.count:] < moon.soi)[0]:
    in_moon.append(satellites[i])

for sat in satellites:
    if sat not in in_sun and sat not in in_moon:
        in_earth.append(sat)

def calc(info: bool = True) -> None:
    x: np.ndarray = Point.state[:, 0:2]
    d: np.ndarray = x[:Body.count, None, :] - np.broadcast_to(x, (Body.count,) + x.shape)
    r: np.ndarray = np.linalg.norm(d, axis=2)
    a: np.ndarray = G * Body.masses[:, None] * np.divide(1, r**2, out=np.zeros_like(r), where=r!=0)
    n: np.ndarray = np.divide(d.astype(np.float64), np.dstack([r]*2), out=np.zeros_like(d.astype(np.float64)), where=np.dstack([r]*2)!=0)
    Point.state[:, 2:] += DT * np.sum(n * a[:, :, None], axis=0)
    Point.state[:, 0:2] += DT * Point.state[:, 2:]

    if info:
        if in_earth:
            if (c:=(r[1, [sat.index for sat in in_earth]] > earth.soi)).any():
                for i in np.where(c)[0]:
                    in_earth[i].ref = sun
                    # print(f"{in_earth[i].name.upper()} entering Sun orbit!")
                    in_sun.append(in_earth[i])
                    in_earth.pop(i)

            if (c:=(r[2, [sat.index for sat in in_earth]] < moon.soi)).any():
                for i in np.where(c)[0]:
                    in_earth[i].ref = moon
                    if in_earth[i].e < 1 and in_earth[i].peri < moon.soi:
                        # print(f"{in_earth[i].name.upper()} entering Moon orbit!")
                        in_moon.append(in_earth[i])
                        in_earth.pop(i)
                    else:
                        in_earth[i].ref = earth

        if in_moon and (c:=(r[2, [sat.index for sat in in_moon]] > moon.soi)).any():
            for i in np.where(c)[0]:
                in_moon[i].ref = earth
                if in_moon[i].e < 1 and in_moon[i].peri < earth.soi:
                    # print(f"{in_moon[i].name.upper()} entering Earth orbit!")
                    in_earth.append(in_moon[i])
                    in_moon.pop(i)
                else:
                    in_moon[i].ref = moon

        if in_sun and (c:=(r[1, [sat.index for sat in in_sun]] < earth.soi)).any():
            for i in np.where(c)[0]:
                in_sun[i].ref = earth
                if in_sun[i].e < 1 and in_sun[i].peri < earth.soi:
                    # print(f"{in_sun[i].name.upper()} entering Earth orbit!")
                    in_earth.append(in_sun[i])
                    in_sun.pop(i)
                else:
                    in_earth[i].ref = sun

    if (c:=(r[:, Body.count:] - Body.radii[:, None] < 0)).any():
        for i in np.unique(np.where(c)[1]):
            # print(f"{satellites[i].name.upper()} crashed!")
            if satellites[i] in in_earth:
                in_earth.remove(satellites[i])
            if satellites[i] in in_moon:
                in_moon.remove(satellites[i])
            if satellites[i] in in_sun:
                in_sun.remove(satellites[i])
            satellites[i].delete()
            satellites.pop(i)

def draw_orbit(focus_x, focus_y, a, e, angle, M_TO_P) -> None:

    angle += np.pi

    b = a * np.sqrt(1-(min(1, e))**2)
    d = a * e
    cx = focus_x + d*np.cos(angle)
    cy = focus_y - d*np.sin(angle)

    cx = int((cx)*M_TO_P)+WIDTH_P//2
    cy = int((cy)*M_TO_P)+HEIGHT_P//2
    a = int(a*M_TO_P)
    b = int(b*M_TO_P)

    if abs(cx)>WIDTH_P*3 or abs(cy)>HEIGHT_P*3:
        return

    target_rect = pygame.Rect((0, 0, 2*a, 2*b))
    target_rect.center = (cx, cy)
    size = (max(1, target_rect.size[0]), max(1, target_rect.size[1]))
    shape_surf = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, ORBIT_COLOR, (0, 0, *size), width=1)
    angle = np.degrees(angle)
    rotated_surf = pygame.transform.rotate(shape_surf, angle)
    screen.blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

def update_screen() -> None:
    M_TO_P = WIDTH_P / WIDTH_M

    screen.fill(BLACK)

    for i, satellite in enumerate(satellites):
        pygame.draw.circle(screen, (max(0, 255-10*i), 100, min(255, 100+10*i)), (int((satellite.x-ref.x) * M_TO_P)+WIDTH_P//2, int((ref.y-satellite.y) * M_TO_P)+HEIGHT_P//2), int(2 * SPRITE_SCALE))
    
    if MOON:    
        pygame.draw.circle(screen, GRAY, (int((moon.x-ref.x) * M_TO_P)+WIDTH_P//2, int((ref.y-moon.y) * M_TO_P)+HEIGHT_P//2), max(2 * SPRITE_SCALE, int(moon.radius * SPRITE_SCALE * M_TO_P)))
    if EARTH:
        pygame.draw.circle(screen, BLUE, (int((earth.x-ref.x) * M_TO_P)+WIDTH_P//2, int((ref.y-earth.y) * M_TO_P)+HEIGHT_P//2), max(2 * SPRITE_SCALE, int(earth.radius * SPRITE_SCALE * M_TO_P)))
    if SUN:
        pygame.draw.circle(screen, YELLOW, (int((sun.x-ref.x) * M_TO_P)+WIDTH_P//2, int((ref.y-sun.y) * M_TO_P)+HEIGHT_P//2), max(2 * SPRITE_SCALE, int(sun.radius * SPRITE_SCALE * M_TO_P)))

    if INFO:
        text = [ref.name.upper()] if ref.ref is None else [
            ref.name.upper(),
            f"Reference: {(ref.ref.name.upper())}",
            f"Altitude: {(ref.mag_r-ref.ref.radius)/1e3:.2f} km",
            f"Velocity: {ref.mag_v:.2f} m/s",
            f"Apoapsis: {ref.apo_surf/1e3:.2f} km AGL ({ref.apo/1e3:.2f} km)",
            f"Periapsis: {ref.peri_surf/1e3:.2f} km AGL ({ref.peri/1e3:.2f} km)",
            f"a: {ref.a/1e3:.2f} km",
            f"e: {ref.e:.2f}",
            f"w: {np.degrees(ref.w)%360:.2f}\u00b0",
        ]
        label = []
        for line in text: 
            label.append(font.render(line, True, FONT_COLOR))

        for i, text in enumerate(label):
            screen.blit(text, (TEXT_X, TEXT_Y+i*(FONT_SIZE+LINE_SPACE)))
        if ref.ref is not None:
            draw_orbit(ref.ref.x-ref.x, ref.y-ref.ref.y, ref.a, ref.e, ref.w, M_TO_P)

    dt_text = font.render(f"{float(DT)}s x {T_RES}", True, (100, 100, 100))
    dt_rect = dt_text.get_rect()
    dt_rect.bottomright = (WIDTH_P-TEXT_X, HEIGHT_P-TEXT_Y)
    screen.blit(dt_text, dt_rect)

    pygame.display.flip()

if __name__=="__main__":
    # Graphics loop
    pygame.init()
    screen = pygame.display.set_mode((WIDTH_P, HEIGHT_P))
    pygame.display.set_caption(f'Orbit Simulator v{VERSION}')
    font = pygame.font.Font('freesansbold.ttf', FONT_SIZE)
    running = True
    paused = True
    ref = earth

    while running:
        if not paused:
            time.sleep(DELAY/1e3)
            for n in range(T_RES):
                calc(not n)

        update_screen()
        
        Point.state -= Point.state[0]

        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    running = False
                    pygame.quit()
                case pygame.MOUSEWHEEL:
                    if event.y==1:
                        zoom_in()
                    elif event.y==-1:
                        zoom_out()
                case pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_e:
                            ref = earth
                        case pygame.K_m:
                            ref = moon
                        case pygame.K_s:
                            ref = sun
                        case pygame.K_d:
                            paused = True
                            display_amap(-WIDTH_M//2+ref.x, WIDTH_M//2+ref.x, -HEIGHT_M//2+ref.y, HEIGHT_M//2+ref.y)
                        case pygame.K_EQUALS:
                            zoom_in()
                        case pygame.K_MINUS:
                            zoom_out()
                        case pygame.K_SPACE:
                            paused = not paused
                        case pygame.K_RIGHT:
                            if paused:
                                for _ in range(T_RES):
                                    calc()
                        case pygame.K_UP:
                            if paused:
                                calc()
                        case pygame.K_u:
                            DT *= 2
                        case pygame.K_j:
                            DT /= 2
                        case pygame.K_i:
                            T_RES = int(min(2*T_RES, 1024))
                        case pygame.K_k:
                            T_RES = int(max(T_RES/2, 1))
                        case pygame.K_COMMA:
                            T_RES = 64
                        case pygame.K_o:
                            DELAY += 50
                        case pygame.K_l:
                            DELAY = (max(DELAY-50, 0))
                        case pygame.K_PERIOD:
                            DELAY = 0
                        case pygame.K_p:
                            INFO = not INFO
                        case _:
                            try:
                                i = int(pygame.key.name(event.key))-1
                            except ValueError:
                                continue
                            i = 9 if i==-1 else i
                            try:
                                ref = satellites[i]
                            except IndexError:
                                pass
