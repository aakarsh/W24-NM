import math
import numpy as np
import pandas as pd


def compute_angles_partner(gradual_attempts, perturbation_angle, mouse_angle):
        # gradual counterclockwise perturbation of perturbation_angle in 10 steps, with perturbation_angle/10, each step lasts 3 attempts
        gradual_step =  perturbation_angle / 10
        step_attempts = 3
        current_step = gradual_attempts // step_attempts

        if current_step <= 10:
            perturbed_mouse_angle = mouse_angle - perturbation_angle * current_step
        else:
            perturbed_mouse_angle = mouse_angle
        return perturbed_mouse_angle

def compute_angles_min(gradual_attempts, perturbation_angle, mouse_angle):
    """
    Function to calculate new position of the circle cursor 
    """
    perturbed_mouse_angle = mouse_angle - gradual_step * (perturbation_angle / 10)
    # Each step lasts 3 attempts, Thus we increase the gradual step only after 3 attempts
    if gradual_attempts % 3 == 0: # Attempts then increase it.  
        gradual_step += 1

    # After we reach 10 steps, we reset the gradual step.
    if gradual_step >= 10: # Reset the gradual_step after 10 steps
        gradual_step = 0

def test_counter_clock_wise():
    start_position = (598, 376)
    mouse_position = (598, 366)

    dx, dy  = mouse_position[0] - start_position[0], mouse_position[1] - start_position[1]
    distance = math.hypot(dx, dy)
    # Assert distance traveled is 10 for vertical movement
     