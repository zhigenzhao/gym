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
from scipy.spatial import ConvexHull
import time 

import numpy as np 
import random 
import cv2
import PIL 

import matplotlib.pyplot as plt

"""Main Simulation class for Ropes."""
class RopeSim(pyglet.window.Window):
    def __init__(self):
        pyglet.window.Window.__init__(self, vsync=False)

        # Sim window parameters. These also define the resolution of the image
        self.width = 500
        self.height = 500
        self.set_caption("RopeSim")

        # Simulation parameters. 
        self.global_time = 0.0
        self.radius = 20
        self.velocity_scale = 1.0

        self.count = 0        
        self.time_now = 0.0

        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.draw_options.flags = self.draw_options.DRAW_SHAPES
        self.graphics_batch = pyglet.graphics.Batch()

        self.create_world()

    """
    1. Methods for Generating and Removing Sim Elements
    """

    """
    1.1 Methods for Initializing World
    """
    def create_world(self):
        self.space = pymunk.Space()
        self.count = 0
        self.has_been_touched = False
        self.collision_pairs = []

        self.space.gravity = Vec2d(0,-1000) # Gravity acts downwards.
        self.space.damping = 0.0001 # quasi-static. low value is higher damping.
        self.space.iterations = 5 # TODO(terry-suh): re-check. what does this do? 
        self.space.color = pygame.color.THECOLORS["black"]       

        self.add_circle()
        self.create_rope()
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
        
        pos_x = 200
        pos_y = 200
        return np.array([pos_x, pos_y])

    def create_circle(self):
        body = pymunk.Body()
        draw_position = self.generate_random_position(self.width, self.height, self.radius)
        body.position = Vec2d(draw_position[0], draw_position[1])
        shape = pymunk.Circle(body, self.radius)
        shape.mass = 1000.0
        shape.elasticity = 1.0
        shape.friction = 1.0
        shape.color = (255, 255, 255, 255)
        return body, shape

    def add_circle(self):
        self.body, self.shape = self.create_circle()
        self.space.add(self.body, self.shape)

    def remove_circle(self):
        self.space.remove(self.body, self.shape)

    """
    2. Methods for adding rope. 
    """

    def create_rope(self):
        self.rope_body_lst = []
        self.rope_shape_lst = []
        self.joint_lst = []

        static_body = pymunk.Body(body_type = pymunk.Body.STATIC)
        static_body.position = (250, 300)
        self.rope_body_lst.append(static_body)

        for rope_index in range(60):
            body = pymunk.Body(0.1)
            top_position = 300 - 5 * rope_index
            bottom_position = 300 - 5 * (rope_index + 1)
            shape = pymunk.Segment(body, (250, top_position),
                (250, bottom_position), 2)
            shape.mass = 0.1
            shape.friction = 0.1
            self.space.add(body, shape)
            self.rope_body_lst.append(body)

            joint = pymunk.PivotJoint(
                self.rope_body_lst[rope_index],
                self.rope_body_lst[rope_index + 1],
                (250, top_position)
            )

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
        ux = int(self.velocity_scale * u[0])
        uy = int(self.velocity_scale * u[1])

        self.collision_pairs = []

        self.body.position += (ux, uy)

        done = False
        
        for x in range(10):
            self.body.position = self.body.position
            self.body.force = (0, self.body.mass * 1000) # offset gravity
            self.space.step(1.0 / 100.0)
        self.render()            

        self.collision_pairs = np.array(self.collision_pairs, dtype=float)

        if (len(self.collision_pairs) > 0) and (not self.has_been_touched):
            print("Touch detected!")
            self.has_been_touched = True

        if (self.has_been_touched and len(self.collision_pairs) == 0):
            print("Touch ended")
            done = True
        
        """
        plt.figure()

        plt.gca().add_patch(plt.Circle(self.body.position, 20, fill=False))
        if len(self.collision_pairs) > 0:
            plt.plot(self.collision_pairs[:,0], self.collision_pairs[:,1], 'ro')
        plt.plot(250, 300, 'b+')
        plt.axis("equal")
        plt.gca().set_xlim([0, 600])
        plt.gca().set_ylim([0, 600])

        plt.savefig("record/{:05d}.png".format(self.count))
        plt.close()
        """

        self.time_now = time.time()
        self.count += 1

        return done    

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
        #self.update_image()

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
        cv2.imwrite("image/{:05d}.png".format(self.count), cv_image)

    def get_current_image(self):
        return self.image

    """
    3. Methods for External Commands
    """

    def refresh(self):
        self.create_world()
        self.render()
