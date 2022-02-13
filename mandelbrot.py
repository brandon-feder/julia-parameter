import pygame
import numpy as np
import time
from numba import jit, vectorize, guvectorize, complex64, int32

RUNNING = True # Whether program is running
MAX_ITERS = 35 # The max number of iterations
C = complex(-1.1, -0.2) # The chosen number
CENTER = None # Center of viewport
VIEW_WIDTH = None # The viewport widths
VIEW_HEIGHT = None # The viewport height
VIEW_RATIO = None # The ratio of the viewports width and height
WIDTH = None # Width of the pygame window
HEIGHT = None # Height of the pygame window
START = None # Complex coordinate coresponding to top left corner of the window
END = None # Complex coordinate coresponding to bottom right corner of the window
LEFT_MOUSE_DOWN = False # Flag whether the mouse is being pressed
WINDOW = None # The main window
MAIN_SURFACE = None # Main surface the will hold image of M-set in order not to render it every frame
FONT = None # The font for drawing text
CLOCK = None # A clock for limiting GPS
Z0 = None # The initial values of all the sample points
ITERATIONS = None # Number of iterations before escaping of each sample point
VIEW_CONSTANT = 1.2 # A constant factor by which the viewport is scaled
MOVE_C_RANDOMLY = True

# Calculate the number of iterations before z0 escapes
@jit(int32(complex64))
def iteratePoint(z0):
    z = z0
    for i in range(MAX_ITERS):
        if abs(z) > 2:
            return i

        z = z*z + z0

    return -1

# Calculate the iteration of all samples of the mandelbrot set
# This is very quick because of the optimizations done by numba's guvectorize
@guvectorize([(complex64[:], int32[:])], '(n)->(n)', target='parallel')
def calcPoints(z0, it):
    for k in range(z0.shape[0]):
        it[k] = iteratePoint(z0[k])

# Calculates the color of all the pixels based off of the number of iterations
# before they escaped
@guvectorize([(int32[:], int32[:])], '(n)->(n)', target='parallel')
def setPixels(iterations, pixels):
    r1, g1, b1 = 0, 0, 0
    r2, g2, b2 = 255, 0, 0

    for k in range(iterations.shape[0]):
        if iterations[k] == -1:
            r = g = b = 255
        else:
            m = iterations[k]/MAX_ITERS
            r = int(r1 + (r2-r1)*m)
            g = int(g1 + (g2-g1)*m)
            b = int(b1 + (b2-b1)*m)

        pixels[k] = int((r<<16) + (g<<8) + b)

# Updates the value of C if the left mouse button is being pressed
def updateC():
    global C
    if LEFT_MOUSE_DOWN:
        C = coordToComplex(pygame.mouse.get_pos())

# Converts a coordinate to it's corresponding complex number
def coordToComplex(c):
    x, y = c
    return complex(
        CENTER[0] - VIEW_WIDTH + x*2*VIEW_WIDTH/WIDTH,
        CENTER[1] - VIEW_HEIGHT + y*2*VIEW_HEIGHT/HEIGHT)

# Converts a complex number to it's corresponding coordinate
def complexToCoord(p):
    x, y = p.real, p.imag
    return (
        (x - CENTER[0] + VIEW_WIDTH) * ( WIDTH / (2*VIEW_WIDTH) ),
        (y - CENTER[1] + VIEW_HEIGHT) * ( HEIGHT / (2*VIEW_HEIGHT) ))

# Function to setup viewport, surfaces, mandelbrot renderings, etc.
def setupViewportAndSamples():
    # Setup viewport
    global VIEW_RATIO, CENTER, VIEW_WIDTH, VIEW_HEIGHT, WIDTH, HEIGHT, MAIN_SURFACE, Z0, ITERATIONS
    
    VIEW_RATIO = WIDTH/HEIGHT
    CENTER = (-0.5, 0)
    
    if HEIGHT < WIDTH:
        VIEW_WIDTH = VIEW_CONSTANT*VIEW_RATIO
    else:
        VIEW_WIDTH = VIEW_CONSTANT

    VIEW_HEIGHT = VIEW_WIDTH/VIEW_RATIO

    # Init pygame
    MAIN_SURFACE = pygame.Surface((WIDTH, HEIGHT))

    # Calculate all the point samples
    r1 = np.linspace(CENTER[0]-VIEW_WIDTH, CENTER[0]+VIEW_WIDTH, WIDTH, dtype=np.float32)
    r2 = np.linspace(CENTER[1]-VIEW_HEIGHT, CENTER[1]+VIEW_HEIGHT, HEIGHT, dtype=np.float32)
    Z0 = (r1 + r2[:,None]*1j).flatten()

    # Calculate the number of iterations at each point
    ITERATIONS = calcPoints(Z0)

    # Get the pixels colors from the iteration counts
    pixels = setPixels(ITERATIONS).reshape((WIDTH, HEIGHT), order="F")

    # Blit those pixels to the mainSurface
    pygame.surfarray.blit_array(MAIN_SURFACE, pixels)


# Function to handle keyboard and mouse events
def handleEvents():
    global LEFT_MOUSE_DOWN, RUNNING, WIDTH, HEIGHT
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            RUNNING = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # Left click
                LEFT_MOUSE_DOWN = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: # Left click
                LEFT_MOUSE_DOWN = False
        if event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.w, event.h
            setupViewportAndSamples()

# Main loop that the process will call
def run(width, height, pipe):
    # Initial setup
    global VIEW_RATIO, WIDTH, HEIGHT, WINDOW, MAIN_SURFACE, FONT, CLOCK, PIXELS, RUNNING

    WIDTH = width
    HEIGHT = height

    pygame.display.init()
    pygame.display.set_caption('Choose a Point on the Mandelbrot Set')
    pygame.font.init()
    WINDOW = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    CLOCK = pygame.time.Clock()
    FONT = pygame.font.SysFont('Comic Sans MS', 15)
    setupViewportAndSamples()

    # Start main loop
    while RUNNING:
        start = time.time() # In order to calculate FPS

        mainMsg = pipe.recv()
        RUNNING = mainMsg["running"]

        if RUNNING == False:
            print("Info: Mandelbrot process quiting after next frame")

        CLOCK.tick(60) # Limit FPS to 60
        handleEvents() # Handle user events
        updateC() # Update C if appropriate

        # Update main window
        WINDOW.blit(MAIN_SURFACE, (0, 0))
        pygame.draw.circle(WINDOW, (4, 255, 0), complexToCoord(C), 3)
        WINDOW.blit(FONT.render('C = {} + {}j'.format(
            round(C.real, 4), round(C.imag, 4)
        ), False, (0, 0, 0)), (10, 10))
        pygame.display.flip()

        # Stop the frame timer
        end = time.time()

        # Send status through pope
        pipe.send({
            "C":C,
            "running":RUNNING
        })

    # Quit pygame
    pygame.quit()