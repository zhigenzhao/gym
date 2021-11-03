import numpy as np
from matplotlib import pyplot as plt

class HopperObservationLogger():
    def __init__(self, env):
        self._env = env
        self._length = 0
        self._time = []
        self._position = []
        self._velocity = []
        self._control = []
        self._contact_forces = {}
        self._contact_penetration = {}

    ##################### properties ########################
    @property
    def time(self):
        return self._time

    @property
    def position(self):
        return self._position
        
    @property
    def velocity(self):
        return self._velocity

    @property
    def control(self):
        return self._control

    @property
    def contact_forces(self):
        return self._contact_forces

    @property
    def contact_penetration(self):
        return self._contact_penetration
    
    ###################### update ############################
    def add_observation(self, obs):
        self._time.append(obs["time"])
        self._position.append(obs["position"])
        self._velocity.append(obs["velocity"])
        self._control.append(obs["control"])
        
        for key in self._contact_forces:
            if key not in obs["contact_forces"]:
                self._contact_forces[key].append(np.zeros(3))
                self._contact_penetration[key].append(0)

        for key in obs["contact_forces"]:
            if key in self._contact_forces:
                self._contact_forces[key].append(obs["contact_forces"][key])
                self._contact_penetration[key].append(obs["contact_penetration"][key])
            else:
                self._contact_forces[key] = []
                self._contact_penetration[key] = []
                for _ in range(self._length):
                    self._contact_forces[key].append(np.zeros(3))
                    self._contact_penetration[key].append(0)

        self._length += 1


    ################### plotting #############################
    def plot_contact(self):
        for key in self._contact_forces:
            forces = np.array(self._contact_forces[key])
            plt.figure()
            plt.plot(self._time[1:], forces[:, 0])
            plt.xlabel("Time (s)")
            plt.ylabel("Force X (N)")
            plt.title("Contact pair " + str(key))

            plt.figure()
            plt.plot(self._time[1:], forces[:, 2])
            plt.xlabel("Time (s)")
            plt.ylabel("Force Z (N)")
            plt.title("Contact pair " + str(key))

            plt.figure()
            plt.plot(self._time[1:], self._contact_penetration[key])
            plt.xlabel("Time (s)")
            plt.ylabel("Penetration Depth (m)")
            plt.title("Contact pair " + str(key))
        
        plt.show()
