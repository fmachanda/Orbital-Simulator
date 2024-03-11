import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pygame
from pygame.locals import *
import time

from orbit import sun, earth, moon
from orbit import Orbiter, Body

#region Constants
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
def calc() -> None:
    for i, satellite in enumerate(satellites):
        apx, apy = EARTH*earth.acceleration(satellite) + MOON*moon.acceleration(satellite) + SUN*sun.acceleration(satellite)
        satellite.vx += apx*DT
        satellite.vy += apy*DT
        satellite.x += satellite.vx*DT
        satellite.y += satellite.vy*DT

        satellite.ref = sun

        for body in [sun, earth, moon]:
            if (body.x - body.radius < satellite.x < body.x + body.radius) and (body.y - body.radius < satellite.y < body.y + body.radius):
                if (body.x - np.sqrt(body.radius**2 - (satellite.y-body.y)**2) < satellite.x < body.x + np.sqrt(body.radius**2 - (satellite.y-body.y)**2)) and (body.y - np.sqrt(body.radius**2 - (satellite.x-body.x)**2) < satellite.y < body.y + np.sqrt(body.radius**2 - (satellite.x-body.x)**2)):
                    print(f"COLLISION: Satellite #{i} with {body.name}")
                    satellites.pop(i)

            if ((body.x-satellite.x)**2 + (body.y-satellite.y)**2)<body.soi_sq and body is not sun:
                satellite.ref = body

    if MOON:
        amx, amy = (EARTH*earth.acceleration(moon)+SUN*sun.acceleration(moon))
        moon.vx += amx*DT
        moon.vy += amy*DT
        moon.x += moon.vx*DT
        moon.y += moon.vy*DT

    if EARTH:
        aex, aey = (MOON*moon.acceleration(earth)+SUN*sun.acceleration(earth))
        earth.vx += aex*DT
        earth.vy += aey*DT
        earth.x += earth.vx*DT
        earth.y += earth.vy*DT

    if SUN:
        asx, asy = (EARTH*earth.acceleration(sun)+MOON*moon.acceleration(sun))
        sun.vx += asx*DT
        sun.vy += asy*DT
        sun.x += sun.vx*DT
        sun.y += sun.vy*DT

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
        if isinstance(ref, Orbiter):
            name = f"Satellite #{satellites.index(ref)+1}"
        elif isinstance(ref, Body):
            name = ref.name
        text = [
            name,
            f"Reference: {(ref.ref.name)}",
            f"Altitude: {(ref.mag_r-ref.ref.radius)/1e3:.2f} km",
            f"Velocity: {ref.mag_v:.2f} m/s",
            f"Apoapsis: {ref.apo_surf/1e3:.2f} km AGL ({ref.apo/1e3:.2f} km)",
            f"Periapsis: {ref.peri_surf/1e3:.2f} km AGL ({ref.peri/1e3:.2f} km)",
            f"a: {ref.a/1e3:.2f} km",
            f"e: {ref.e:.2f}",
            f"w: {np.degrees(ref.w)%360:.2f}\u00b0",
        ] if ref.ref is not None else [name]
        label = []
        for line in text: 
            label.append(font.render(line, True, FONT_COLOR))

        for i, text in enumerate(label):
            screen.blit(text, (TEXT_X, TEXT_Y+i*(FONT_SIZE+LINE_SPACE)))

    dt_text = font.render(f"{float(DT)}s x {T_RES}", True, (100, 100, 100))
    dt_rect = dt_text.get_rect()
    dt_rect.bottomright = (WIDTH_P-TEXT_X, HEIGHT_P-TEXT_Y)
    screen.blit(dt_text, dt_rect)

    pygame.display.flip()

if __name__=="__main__":
    # Graphics loop
    pygame.init()
    screen = pygame.display.set_mode((WIDTH_P, HEIGHT_P))
    pygame.display.set_caption('Orbit Simulator v0.01')
    font = pygame.font.Font('freesansbold.ttf', FONT_SIZE)
    running = True
    paused = True
    ref = earth

    while running:
        if not paused:
            time.sleep(DELAY/1e3)
            for _ in range(T_RES):
                calc()

        update_screen()

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
