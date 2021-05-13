import pygame 
from pygame.locals import *
from pygame.color import * 
import pyglet 
from pyglet.gl import *
from pyglet.window import key, mouse 
from pyglet import shapes

import pymunk
from pymunk import Vec2d
import pymunk.pyglet_util
from shapely.geometry import Polygon
import time 

import numpy as np 
import random 
import cv2
import PIL 

import matplotlib.pyplot as plt

"""Main Simulation class for Bubbles.

The attributes and methods are quite straightforward, see carrot_sim.py
for basic usage.
"""
class BubbleSim(pyglet.window.Window):
    def __init__(self):
        pyglet.window.Window.__init__(self, vsync=False)

        # Sim window parameters. These also define the resolution of the image
        self.width = 500
        self.height = 500
        self.set_caption("BubbleSim")

        # Simulation parameters. 
        self.global_time = 0.0
        self.radius = 20

        self.PV = 9e5 # product of P * V according to ideal gas law.

        
        self.time_now = 0.0

        self.count = 0        

        self.image = None
        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.draw_options.flags = self.draw_options.DRAW_SHAPES
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
        self.space = pymunk.Space()        
        self.space.gravity = Vec2d(0,0) # planar setting 
        self.space.damping = 0.6 # quasi-static. low value is higher damping.
        self.space.iterations = 100 # TODO(terry-suh): re-check. what does this do? 
        self.space.color = pygame.color.THECOLORS["black"]

        self.add_box()
        self.create_bubble()
        self.render()

        def draw_collision(arbiter, space, data):
            self.graphics_batch = pyglet.graphics.Batch()
            count = 0
            for c in arbiter.contact_point_set.points:
                p = list(map(int, c.point_a))
                self.collision_pairs.append(p)

        ch = self.space.add_default_collision_handler()
        ch.post_solve = draw_collision


    """
    1.2 Methods for Generating and Removing Onion Pieces
    """

    def generate_random_position(self, width, height, radius):
        pos_x = random.randint(radius, width - radius)
        pos_y = random.randint(radius, height - radius)
        
        pos_x = 150
        pos_y = 150
        return np.array([pos_x, pos_y])

    def create_box(self):
        body = pymunk.Body()
        body.position = Vec2d(250, 250)
        box_size = 30

        mass = 20.0
        points = [(-box_size, -box_size), (-box_size, box_size), (box_size, box_size), (box_size, -box_size)]
        moment = pymunk.moment_for_poly(mass, points, (0, 0))
        shape = pymunk.Poly(body, points)
        shape.mass = mass
        shape.elasticity = 1.0
        shape.friction = 1.0
        shape.color = (255, 255, 255, 255)

        def big_damping(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, (0, 0), 0.0001, dt)

        body.velocity_func = big_damping

        return body, shape

    def add_box(self):
        self.body, self.shape = self.create_box()
        self.space.add(self.body, self.shape)

    """
    2. Methods for adding rope. 
    """

    def create_bubble(self):
        self.bubble_body_lst = []

        # 1. Create center element.
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = (150, 150)
        self.body.angle = 0.0
        shape = pymunk.Segment(self.body, (-40, 0), (40, 0), 1)
        shape.mass = 100.0
        shape.elasticity = 1.0
        shape.color = (255, 0, 0, 0)
        shape.friction = 1.0

        self.space.add(self.body, shape)

        # 2. Create boundary elements. 
        radius = 40
        self.num_elements = 60
        self.boundary_element_lst = []
        for elem in range(self.num_elements):
            body = pymunk.Body(0.1)
            body.position = (self.body.position[0] + radius * np.cos(elem * 2.0 * np.pi / self.num_elements),
                             self.body.position[1] + radius * np.sin(elem * 2.0 * np.pi / self.num_elements))
            shape = pymunk.Circle(body, 2)
            shape.mass = 10.0
            shape.friction = 1.0
            if elem % 30 == 0:
                shape.color = (255, 0, 0, 0)

            self.space.add(body, shape)
            self.boundary_element_lst.append(body)

            if elem % 30 == 0:
                #joint = pymunk.DampedSpring(body, center_body, (0, 0), (0, 0),
                #    radius, 2000, 800)
                joint = pymunk.PivotJoint(body, self.body, body.position)
                joint.collide_bodies = False 
                self.space.add(joint)

        # 3. Create boundary joints 
        for elem in range(self.num_elements):
            body_a = self.boundary_element_lst[elem % self.num_elements]
            body_b = self.boundary_element_lst[(elem + 1) % self.num_elements]

            rl = np.linalg.norm(np.array(body_a.position) - np.array(body_b.position))

            joint = pymunk.DampedSpring(body_a, body_b, (0, 0), (0, 0),
                0, 400, 100)
            joint.collide_bodies = False
            self.space.add(joint)
                            
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
        self.collision_pairs = []        

        # This updates the position and prevents the ball from going out of screen.
        self.body.position += (u[0], u[1])
        self.body.angle += u[2]

        done = False
        if (self.body.position < np.array([self.radius, self.radius])).any():
            done = True 
        if (self.body.position > np.array([self.width - self.radius, self.height - self.radius])).any():
            done = True
        
        for x in range(60):
            self.body.position = self.body.position
            self.body.angle = self.body.angle
            #self.boundary_element_lst[0].position = self.boundary_element_lst[0].position
            #self.boundary_element_lst[40].position = self.boundary_element_lst[40].position
            self.space.step(1.0 / 60.0)

            # 1. Compute area and Laplacian.

            body_position_lst = []
            body_laplacian_lst = []
            # Compute Laplacian and add pressure terms.
            for body in range(len(self.boundary_element_lst)):
                # Compute Laplacian.
                body_now = self.boundary_element_lst[body % self.num_elements]
                body_next = self.boundary_element_lst[(body + 1) % self.num_elements]
                body_prev = self.boundary_element_lst[(body - 1) % self.num_elements]

                vec_next = (body_next.position - body_now.position) / np.linalg.norm(
                    body_next.position - body_now.position)
                vec_prev = (body_prev.position - body_now.position) / np.linalg.norm(
                    body_prev.position - body_now.position)
                sign = -np.sign(np.cross(vec_next, vec_prev))

                normal = float(sign) * (vec_next + vec_prev) / np.linalg.norm(vec_next + vec_prev)

                body_position_lst.append(body_now.position)
                body_laplacian_lst.append(normal)

            # Comptue area.
            boundary_shape = Polygon(body_position_lst)
            volume = boundary_shape.area
            pressure = self.PV / volume

            for body in range(len(self.boundary_element_lst)):
                self.boundary_element_lst[body].force = pressure * body_laplacian_lst[body]

        self.render()
        self.time_now = time.time()
        self.count += 1

        return body_position_lst, False

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
        #self.graphics_batch.draw()
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
        cv2.imwrite("record/{:05d}.png".format(self.count), cv_image)

    def get_current_image(self):
        return self.image

    """
    3. Methods for External Commands
    """

    def refresh(self):
        self.create_world()
        self.render()
