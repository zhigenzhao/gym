import pygame 
from pygame.locals import *
from pygame.color import * 
import pyglet 
from pyglet.gl import *
from pyglet.window import key, mouse 

import pymunk
from pymunk import Vec2d
import pymunk.pyglet_util
from scipy.spatial import ConvexHull
import time 

import numpy as np 
import random 
import cv2
import PIL 

"""Main Simulation class for carrots.

The attributes and methods are quite straightforward, see carrot_sim.py
for basic usage.
"""
class SingleIntegratorSim(pyglet.window.Window):
    def __init__(self):
        pyglet.window.Window.__init__(self, vsync=False)

        # Sim window parameters. These also define the resolution of the image
        self.width = 500
        self.height = 500
        self.set_caption("SingleIntegratorSim")

        # Simulation parameters. 
        self.global_time = 0.0
        self.radius = 20
        self.velocity_scale = 20

        self.space = pymunk.Space()

        self.image = None
        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.draw_options.shape_outline_color = (255, 255, 255, 255)
        self.graphics_batch = pyglet.graphics.Batch()

        self.create_world()

        # User flags.
        # If this flag is enabled, then the rendering will be done at every
        # simulation timestep. This makes the sim much slower, but is better
        # for visualizing the pusher continuously.
        self.RENDER_EVERY_TIMESTEP = False


    """
    1. Methods for Generating and Removing Sim Elements
    """

    """
    1.1 Methods for Initializing World
    """
    def create_world(self):
        self.space.gravity = Vec2d(0,0) # planar setting 
        self.space.damping = 0.0001 # quasi-static. low value is higher damping.
        self.space.iterations = 1 # TODO(terry-suh): re-check. what does this do? 
        self.space.color = pygame.color.THECOLORS["white"]

        # Choose spawn position at random to fit within window. 
        self.add_circle()
        self.render()

    """
    1.2 Methods for Generating Circle Body
    """
    def generate_random_position(self, width, height, radius):
        # TODO(terry-suh): Make sure these are not flipped?
        position_x = random.randint(radius, width - radius)
        position_y = random.randint(radius, height - radius)
        return np.array([position_x, position_y])

    def create_circle(self):
        """
        Create a single bar by defining its shape, mass, etc.
        """
        body = pymunk.Body(1e7, pymunk.inf)
        draw_position = self.generate_random_position(self.width, self.height, self.radius)
        body.position = Vec2d(draw_position[0], draw_position[1])
        shape = pymunk.Circle(body, self.radius)
        shape.elasticity = 0.1
        shape.friction = 0.6
        shape.color = (255, 255, 255, 255)
        return body, shape

    def add_circle(self):
        """
        add the circle from simulation.
        """
        self.body, self.shape = self.create_circle()
        self.space.add(self.body, self.shape)        

    def remove_circle(self):
        """
        remove the circle from simulation.
        """
        self.space.remove(self.body, self.shape)

    """
    2. Methods for Updating and Rendering Sim
    """

    """
    2.1 Methods Related to Updating
    """

    # Update the sim by applying action, progressing sim, then stopping. 
    def update(self, u):
        """
        Once given a control action, run the simulation forward and return.
        """
        # Parse into integer coordinates
        ux = int(self.velocity_scale * u[0])
        uy = int(self.velocity_scale * u[1])

        # This updates the position and prevents the ball from going out of screen.
        self.body.position = np.clip(self.body.position + np.array([ux, uy]),
                                     [self.radius, self.radius],
                                     [self.width - self.radius, self.height - self.radius])
        self.space.step(0.01)

        # Wait 1 second in sim to slow down moving pieces, and render.
        #self.wait(1.0)
        self.render()

        return None

    """
    2.2 Methods related to rendering
    """
    def on_draw(self):
        self.render()

    def render(self):
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        self.space.debug_draw(self.draw_options)
        self.dispatch_events() # necessary to refresh somehow....
        self.flip()
        self.update_image()

    """
    2.3 Methods related to image publishing
    """
    def update_image(self):
        pitch = -(self.width * len('RGB'))
        img_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().\
            get_data('RGB', pitch=pitch)
        pil_im = PIL.Image.frombytes('RGB', (self.width, self.height), img_data)
        cv_image = np.array(pil_im)[:,:,::-1].copy()
        self.image = cv_image

    def get_current_image(self):
        return self.image

    """
    3. Methods for External Commands
    """

    def refresh(self):
        self.remove_circle()
        self.add_circle()
        self.render()
