import logging
import datetime


class Logger:
    @staticmethod
    def create():
        try:
            filename = './log/' + str(datetime.datetime.now()) + '.log'
            logging.basicConfig(filename=filename, level=logging.DEBUG)
        except IOError:
            print('No such file or directory: {}'.format(filename))

    @staticmethod
    def log_step(action, previous_state, current_state, reward, iterations, done):
        logging.info(
            "Action: {}, state diff.: r={} to {}, fi={} to {}, theta={} to {}, reward: {}, iterations:{}, finished:{}".format(
                action, previous_state[0], current_state[0], previous_state[1], current_state[1], previous_state[2],
                current_state[2], reward, iterations, done))

    @staticmethod
    def log_error(class_tag, error):
        logging.error(class_tag + error)

    @staticmethod
    def log_msg(msg):
        logging.info(msg)
