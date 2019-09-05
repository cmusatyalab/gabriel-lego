import random

from lego_engine import config
from cv import bitmap as bm
from lego_engine.tasks.Task import Task


class ExTask(Task):
    def __init__(self, bitmaps):
        Task.__init__(self, bitmaps)
        self.target_state_idx = -1

    def get_first_guidance(self):
        result = {'status': 'success'}

        self.target_state_idx = 0
        target_state = self.get_state(self.target_state_idx)

        result['speech'] = \
            "Welcome to the Lego task. As a first step, please " \
            "find a piece of 1x%d %s brick and put it on the " \
            "board." % (target_state.shape[1],
                        config.COLOR_ORDER[target_state[0, 0]])
        result['animation'] = bm.bitmap2guidance_animation(target_state,
                                                           config.ACTION_TARGET)
        result['time_estimate'] = self.time_estimates[0]
        img_guidance = bm.bitmap2guidance_img(target_state, None,
                                              config.ACTION_TARGET)
        return result, img_guidance

    def search_next(self, current_state, bm_diffs, search_type='more'):
        pass  # todo

    def is_final_state(self):
        pass

    def get_guidance(self):
        result = {'status': 'success'}
        target_state = self.get_state(self.target_state_idx)

        ## Check if we at least reached the previously desired state
        if bm.bitmap_same(self.current_state, target_state):
            ## Task is done
            # if self.is_final_state():
            if self.target_state_idx == len(self.states) - 1:
                result['speech'] = "You have completed the task. " \
                                   "Congratulations!"
                result['animation'] = bm.bitmap2guidance_animation(
                    self.current_state, config.ACTION_TARGET)
                img_guidance = bm.bitmap2guidance_img(self.current_state, None,
                                                      config.ACTION_TARGET)
                return result, img_guidance

            ## Not done
            ## Next state is simply the next one in line
            self.target_state_idx += 1
            target_state = self.get_state(self.target_state_idx)

            ## Determine the type of change needed for the next step

            diff = bm.bitmap_diff(self.current_state, target_state)
            assert diff  # states can't be the same
            assert diff['n_diff_pieces'] == 1  # for now only change one
            # piece at the time

            self.prev_good_state = self.current_state
            self.good_word_idx = (self.good_word_idx +
                                  random.randint(1, 3)) % 4
            if self.target_state_idx != -1:
                result['time_estimate'] = self.time_estimates[
                    self.target_state_idx]

            if diff['larger'] == 2:  # target state has one more piece
                result['speech'] = bm.generate_message(
                    self.current_state,
                    target_state,
                    config.ACTION_ADD,
                    diff['first_piece'],
                    step_time=self.current_time - self.prev_time,
                    good_word_idx=self.good_word_idx)

                result['animation'] = bm.bitmap2guidance_animation(
                    target_state,
                    config.ACTION_ADD,
                    diff_piece=diff['first_piece'])

                img_guidance = bm.bitmap2guidance_img(target_state,
                                                      diff['first_piece'],
                                                      config.ACTION_ADD)
                return result, img_guidance
            else:  # target state has one less piece
                result['speech'] = bm.generate_message(
                    self.current_state,
                    target_state,
                    config.ACTION_REMOVE,
                    diff['first_piece'],
                    step_time=self.current_time - self.prev_time,
                    good_word_idx=self.good_word_idx)

                result['animation'] = bm.bitmap2guidance_animation(
                    self.current_state,
                    config.ACTION_REMOVE,
                    diff_piece=diff['first_piece'])

                img_guidance = bm.bitmap2guidance_img(self.current_state,
                                                      diff['first_piece'],
                                                      config.ACTION_REMOVE)

                return result, img_guidance


        else:
            self.target_state_idx = self.state2idx(self.prev_good_state)
            result['speech'] = "This is incorrect, please undo the last " \
                               "step and revert to the model shown on " \
                               "the screen."
            result['animation'] = bm.bitmap2guidance_animation(
                self.prev_good_state, config.ACTION_TARGET)
            img_guidance = bm.bitmap2guidance_img(self.prev_good_state,
                                                  None,
                                                  config.ACTION_TARGET)
            return result, img_guidance
