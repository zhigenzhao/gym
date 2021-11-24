import numpy as np
from matplotlib import pyplot as plt

HOPPER_CONTACT_PAIR = {
    (20, 44): "Toe",
    (20, 41): "Heel"
}

############################## hopper functions ############################
def hopper_running_cost(x, u):
    R = 0.01*np.ones(3)
    Q = np.array([1, 10, 10, 100, 100, 1, 1, 1, 1, 1])
    R = R/2
    Q = Q/2
    xf = np.array([4, 1.5, 0.72273432, -1.44546857, 2.29353058, 0, 0, 0, 0, 0])
    # state reg cost
    state_cost = np.dot((x - xf)**2, Q)
    # control reg cost
    control_weight = np.dot(u**2, R)

    return state_cost + control_weight

######################### clases ###########################################

class HopperObservationLogger():
    def __init__(self, env):
        self._env = env
        self._length = 0
        self._time = []
        self._position = []
        self._velocity = []
        self._control = []
        self._running_cost = []
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
    
    @property
    def running_cost(self):
        return self._running_cost
    
    ###################### update ############################
    def add_observation(self, obs):
        self._time.append(obs["time"])
        self._position.append(obs["position"])
        self._velocity.append(obs["velocity"])
        self._control.append(obs["control"])
        if "running_cost" in obs:
            self._running_cost.append(obs["running_cost"])
        else:
            self._running_cost.append(0)
        
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
        contact_pair_names = []
        for key in self._contact_forces:
            if key in HOPPER_CONTACT_PAIR:
                contact_pair_names.append(HOPPER_CONTACT_PAIR[key])
            else:
                contact_pair_names.append("Contact pair " + str(key))

            forces = np.array(self._contact_forces[key])
            plt.figure(1)
            plt.plot(self._time[1:], forces[:, 0])

            plt.figure(2)
            plt.plot(self._time[1:], forces[:, 2])
            
            plt.figure(3)
            plt.plot(self._time[1:], self._contact_penetration[key])
        
        plt.figure(1)
        plt.legend(contact_pair_names)
        plt.xlabel("Time (s)")
        plt.ylabel("Force X (N)")
        plt.title("X Friction")

        plt.figure(2)
        plt.legend(contact_pair_names)
        plt.xlabel("Time (s)")
        plt.ylabel("Force Z (N)")
        plt.title("Z Normal Force")

        plt.figure(3)
        plt.legend(contact_pair_names)
        plt.xlabel("Time (s)")
        plt.ylabel("Depth (m)")
        plt.title("Penetration Depth")
        
        plt.show()

    def plot_cum_cost(self):
        plt.figure(4)
        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative Cost")
        plt.plot(self._time[:], np.cumsum(self._running_cost))
        plt.show()

    def plot_torque(self):
        plt.figure(5)
        plt.xlabel("Time (s)")
        plt.ylabel("Control Torque (N$\cdot$m)")
        controls = np.array(self._control)
        plt.plot(self._time[:], controls[:, 0])
        plt.plot(self._time[:], controls[:, 1])
        plt.plot(self._time[:], controls[:, 2])

        plt.legend(["Hip", "Knee", "Ankle"])
        plt.show()

    def plot_joints(self):
        plt.figure(6)
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Angles (rad)")
        states = np.array(self._position)
        plt.plot(self._time[:], states[:, 2])
        plt.plot(self._time[:], states[:, 3])
        plt.plot(self._time[:], states[:, 4])

        plt.legend(["Hip", "Knee", "Ankle"])
        plt.show()