import numpy as np
import robodk
import time
import queue
from scipy.spatial.transform import Rotation, Slerp
from PIL import Image
from robolink import *
from matplotlib import pyplot as plt
import multiprocessing
from constants import BELT_VELOCITY

BOX_RANDOM_ANGLE = np.pi / 8.0
BOX_X_RANDOM = 50.0
GRAVITY = -9.81

class SimulationLoop:
    CONVEYOR_BELT_END = 100.0
    def __init__(self, queue, lock):
        self.sleep_for = 1.0 / 60.0
        self.link = Robolink()
        self.box_velocity = np.array([0.0, -BELT_VELOCITY, 0.0])
        self.paused = False
        self.done = False
        self.previous_sim_time = None
        self.queue = queue
        self.box = self.link.Item('Box')
        self.write_lock = lock

    def run(self):
        self.link.setSimulationSpeed(1.0)
        self.previous_sim_time = self.link.SimulationTime()
        while not self.done:
            self._read_queue()
            if self.paused:
                time.sleep(0.05)
                continue
            self._step_simulation()
            time.sleep(self.sleep_for)

    def _read_queue(self):
        try:
            msg = self.queue.get(False)
            try:
                self.write_lock.acquire()
                getattr(self, msg[0])(*msg[1:])
            finally:
                self.write_lock.release()

        except queue.Empty:
            pass

    def _step_simulation(self):
        current_time = self.link.SimulationTime()
        diff = current_time - self.previous_sim_time
        try:
            self.write_lock.acquire()
            self.previous_sim_time = current_time
            if self.box.Parent().Name() != 'picking_setup':
                # Box is in the robot's hand. Don't do anything.
                return

            current_pose = np.array(self.box.Pose().Rows())
            if current_pose[1, 3] < self.CONVEYOR_BELT_END:
                self.reset_box()
                return

            if self.box.Parent().Name() == "picking_setup":
                # On conveyor belt. Let's move it.
                current_pose[:3, 3] += diff * self.box_velocity * 1000.0 # Pose is in millimeters.
                if current_pose[2, 3] > 5.0:
                    z = current_pose[2, 3]
                    current_pose[2, 3] = max(0.0, z + diff * GRAVITY * 1000.0)

            self.box.setPose(robodk.Mat(current_pose.tolist()))
        finally:
            self.write_lock.release()

    def reset_box(self):
        gripper = self.link.Item('Gripper')
        gripper.DetachAll()
        try:
            box = self.link.Item('Box')
            if box.Name() == "Box":
                box.Delete()
        except Exception as e:
            print(e)

        box_template = self.link.Item('BoxTemplate')
        box_template.Copy()
        self.box = self.link.Paste(self.link.Item('picking_setup'))
        self.box.setName("Box")

        self.box.setParent(self.link.Item('picking_setup'))

        box_pose = np.array(self.box.Pose().Rows())
        box_pose[:3, :3] = Rotation.from_rotvec([0.0, 0.0,
            -np.pi / 2.0 + np.random.uniform(-BOX_RANDOM_ANGLE, BOX_RANDOM_ANGLE)
        ]).as_matrix()
        box_pose[0, 3] = 200.0 + np.random.uniform(-BOX_X_RANDOM, BOX_X_RANDOM)
        box_pose[1, 3] = 1800.0
        box_pose[2, 3] = 0.0
        self.box.setPose(robodk.Mat(box_pose.tolist()))
        self.box.Scale(np.random.uniform(np.array([0.7, 0.7, 0.1]), np.ones(3)).tolist())

    def pause(self, value):
        self.paused = value
        if not self.paused:
            self.previous_sim_time = self.link.SimulationTime()

    def close(self):
        self.done = True

def simulation_loop(queue, lock):
    loop = SimulationLoop(queue, lock).run()

class Simulation:
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.write_lock = multiprocessing.Lock()
        self.background_thread = multiprocessing.Process(target=simulation_loop, args=(self.queue, self.write_lock), daemon=True)
        self.background_thread.start()

    def reset_box(self):
        self.queue.put(('reset_box',))

    def pause(self, value):
        self.queue.put(('pause', value))

    def close(self):
        self.queue.put(('close',))

