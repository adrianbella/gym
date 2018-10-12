import imageio
import glob
import numpy as np
import math
import cv2

import gym
from gym import spaces
from gym.utils import seeding

from gym.envs.virtual_drone.display import Display


class Environment(gym.Env):
    CLASS_TAG = 'Environment: '
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.seed()

        self.iterations = 0
        self.episode_steps = 0
        self.iteration_limit = 100
        self.done = False
        self.info = {
            'iterations': self.iterations
        }
        self.optimal_position = np.array([3, 0, 10])

        # set of the name of the figures were used during the training process
        self.traning_names = np.array(
            ['andromeda', 'assistant', 'dreyar', 'eve', 'jasper', 'kachujin', 'liam', 'lola', 'malcolm', 'mark',
             'medea',
             'peasant'])

        # set of the name of the figures were used during the validation process
        self.validation_names = np.array(['regina', 'remy', 'stefani'], dtype='U10')  #

        self.img_array = np.zeros((7, 45, 21, 200, 200)).astype(np.uint8)

        self.current_name = ''
        self.current_state = np.zeros(3)
        self.previous_state = np.zeros(3)

        self.actions = np.array(['forward', 'backward', 'up', 'down', 'left', 'right'])
        self.action_space = spaces.Discrete(self.actions.size)

        self.observation_space = spaces.Box(low=0, high=255, shape=(200, 200), dtype=np.uint8)

        try:
            self.read_files()
        except IOError as e:
            print(self.CLASS_TAG + "Error: can\'t find the files or read data")
        else:
            print(self.CLASS_TAG + "Reading screenshots was successful!")

    def read_files(self):

        self.current_name = self.get_random_name()
        current_figure_screenShots_path = "./Screenshots/" + self.current_name + "/*.png"

        for im_path in glob.glob(current_figure_screenShots_path):
            x, y, z = self.get_descartes_coordinates(im_path)
            r, fi, theta = self.get_polar_coordinates(x, y, z)

            r_index = int((r - 0.5) / 0.15)  # r_index [0:6]
            fi_index = int(fi / 8)  # fi_index [0:44]
            theta_index = int((theta - 10) / 8)  # theta_index [0:20]

            img = imageio.imread(im_path)
            self.img_array[r_index, fi_index, theta_index] = self.gray_scale(img)

    def get_random_name(self):
        figures_count = len(self.traning_names)
        figure_index = self.np_random.randint(0,
                                              figures_count)  # choose a random figure_index from the possible ones
        name = self.traning_names[figure_index]  # get the name of the figure
        return name

    def get_descartes_coordinates(self, img_path):

        # cut the required x, y and z string values from the filename
        from_str = img_path.split('(')[1]
        coordinates_with_coma = from_str.split(')')[0]
        str_x, str_y, str_z = coordinates_with_coma.split(',')
        # -----------------------------------------------------------

        return float(str_x), float(str_y), float(str_z)

    def get_polar_coordinates(self, x, y, z):

        shifted_y = y - 1.6

        r = math.sqrt(math.pow(x, 2) + math.pow(shifted_y, 2) + math.pow(z, 2))  # calculate the |r| value
        theta = math.acos(shifted_y / r)
        fi = math.asin(x / (math.sin(theta) * r))

        # convert theta an fi from rad to degree
        theta = math.degrees(theta)
        fi = math.degrees(fi)

        theta = int(round(theta))
        fi = int(round(fi))
        r = round(r, 2)

        # convert fi to [0:360] interval
        if fi < 0:
            fi += 360

        if z > 0:
            if x > 0:
                fi = 180 - fi
            elif x < 0:
                fi = 180 + (360 - fi)
        # -------------------------------

        return r, fi, theta

    def gray_scale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.uint8)

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def reset(self):

        self.episode_steps = 0
        self.done = False
        self.previous_state = np.zeros(3)  # set previous state to null

        r_index = self.np_random.randint(0, 7)
        fi_index = self.np_random.randint(0, 45)
        theta_index = self.np_random.randint(0, 21)

        self.current_state = np.array([r_index, fi_index, theta_index])

        initial_observation = self.img_array[r_index, fi_index, theta_index]

        return initial_observation

    def render(self, mode='human'):
        r_index = self.current_state[0]
        fi_index = self.current_state[1]
        theta_index = self.current_state[2]

        if self.episode_steps != 0:
            Display.close()

        if mode == 'rgb_array':
            return self.img_array[r_index, fi_index, theta_index]  # return RGB frame suitable for video
        elif mode is 'human':
            Display.show_img(self.img_array[r_index, fi_index, theta_index])
            return self.img_array[r_index, fi_index, theta_index]
        else:
            super(Environment, self).render(mode=mode)  # just raise an exception

    def close(self):
        Display.close()
        self.img_array = None

    def step(self, action_index):

        # validate action index
        if not (0 <= action_index <= 5):
            print(self.CLASS_TAG + 'Invalid action index: ' + action_index + ', It must be between [0:5]')
        # --------------------------------------------------------------------------------------------------

        action = self.actions[action_index]

        self.previous_state = self.current_state  # save the previous polar coordinates of the state

        r_index = self.current_state[0]
        fi_index = self.current_state[1]
        theta_index = self.current_state[2]

        if action == 'forward':
            if r_index != 0:  # check if we can move forward
                r_index -= 1
        elif action == 'backward':
            if r_index != 6:  # check if we can move backward
                r_index += 1
        elif action == 'up':
            if theta_index != 0:  # check if we can move up
                theta_index -= 1
        elif action == 'down':
            if theta_index != 20:  # check if we can move down
                theta_index += 1
        elif action == 'left':
            if fi_index == 0:  # check if we are across from the figure
                fi_index = 44
            else:  # all other cases
                fi_index -= 1
        elif action == 'right':
            if fi_index == 44:  # check if we reached the maximal fi_index value
                fi_index = 0
            else:  # all other cases
                fi_index += 1

        # validate and set indexes
        if not (0 <= r_index <= 6):
            print(self.CLASS_TAG + 'Invalid r index: ' + r_index)
        elif not (0 <= fi_index <= 44):
            print(self.CLASS_TAG + 'Invalid fi index: ' + fi_index)
        elif not (0 <= theta_index <= 20):
            print(self.CLASS_TAG + 'Invalid theta index: ' + theta_index)
        else:
            self.current_state = np.array(
                [r_index, fi_index, theta_index])  # save the new polar coordinates of the current state
            observation = self.img_array[r_index, fi_index, theta_index]  # find the new camera input
        # -------------------------------------------------------------------------------------------------

        reward = self.get_reward()  # calcuate the reward

        self.iterations += 1
        self.episode_steps += 1
        self.info.update({'iterations': self.iterations})

        # if we reach the maximal given iteration count, or the optimal position, set done true
        if self.episode_steps == self.iteration_limit or np.array_equal(self.optimal_position, self.current_state):
            self.done = True

        return observation, reward, self.done, self.info

    def get_reward(self):

        prev_r_distance = abs(self.previous_state[0] - self.optimal_position[0])
        if self.previous_state[1] > 22:
            prev_fi_distance = abs(self.previous_state[1] - 45)
        else:
            prev_fi_distance = abs(self.previous_state[1] - self.optimal_position[1])
        prev_theta_distance = abs(self.previous_state[2] - self.optimal_position[2])

        previous_distance = prev_r_distance + prev_fi_distance + prev_theta_distance

        current_r_distance = abs(self.current_state[0] - self.optimal_position[0])
        if self.current_state[1] > 22:
            current_fi_distance = abs(self.current_state[1] - 45)
        else:
            current_fi_distance = abs(self.current_state[1] - self.optimal_position[1])
        current_theta_distance = abs(self.current_state[2] - self.optimal_position[2])

        current_distance = current_r_distance + current_fi_distance + current_theta_distance

        reward_dist = 0

        if (current_distance - previous_distance) >= 0:
            reward_dist -= 1
        else:
            reward_dist += 1

        reward_bonus = 0

        if reward_dist > 0 and current_distance < 5:
            reward_bonus = math.exp(5 / (pow(current_distance * 2, 2) + 2))

        reward_sum = reward_dist + reward_bonus

        return reward_sum
