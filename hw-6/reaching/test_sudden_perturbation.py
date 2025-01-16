import math
import numpy as np
import pandas as pd

def compute_perturbed_position(pos,distance, angle):
    """
    Function to calculate new position of the circle cursor 
    """
    new_x = pos[0] + distance * math.cos(angle)
    new_y = pos[1] + distance * math.sin(angle)
    return new_x, new_y


def test_clockwise_rotation_vertical_up():
    start_position = (598, 376)
    mouse_position = (598, 366)
    dx, dy  = mouse_position[0] - start_position[0], mouse_position[1] - start_position[1]
    distance = math.hypot(dx, dy)
    # Assert distance travelled is 10 for vertical movement
    assert np.isclose(distance, 10)
    
    # Assert the angle is -\pi/2 degrees
    mouse_angle = math.atan2(dy, dx)
    assert np.isclose(mouse_angle, -np.pi/2)
    # Assert that we are going to perturm the mouse angle by 30 degrees.
    perturbation_angle = math.radians(30)
    new_angle = mouse_angle + perturbation_angle
    # Assert that the new angle is -\pi/2 + 30 degrees
    assert np.isclose(new_angle, -np.pi/2 + perturbation_angle)
    # compute the new position.
    new_dx, new_dy = distance * math.cos(new_angle), distance * math.sin(new_angle)
    new_x, new_y = start_position[0] + new_dx, start_position[1] + new_dy
    # Assert the new position is (608, 376)
    assert np.isclose(new_x, 603)
    assert np.isclose(new_y, 367.33974596215563)
    assert mouse_position[0] < new_x, "new x is larger than the mouse position"
    assert mouse_position[1] < new_y, "new y is larger than the mouse position"
    assert np.isclose(new_x, compute_perturbed_position(start_position, distance, new_angle)[0])
    assert np.isclose(new_y, compute_perturbed_position(start_position, distance, new_angle)[1])

   
    
def test_clockwise_rotation_vertical_down():
    start_position = (598, 376)
    mouse_position = (598, 386)
    dx, dy  = mouse_position[0] - start_position[0], mouse_position[1] - start_position[1]
    distance = math.hypot(dx, dy)
    # Assert distance travelled is 10 for vertical movement
    assert np.isclose(distance, 10)
    
    # Assert the angle is \pi/2 degrees
    mouse_angle = math.atan2(dy, dx)
    assert np.isclose(mouse_angle, np.pi/2)
    # Assert that we are going to perturm the mouse angle by 30 degrees.
    perturbation_angle = math.radians(30)
    new_angle = mouse_angle + perturbation_angle
    # Assert that the new angle is \pi/2 + 30 degrees
    assert np.isclose(new_angle, np.pi/2 + perturbation_angle)
    # compute the new position.
    new_dx, new_dy = distance * math.cos(new_angle), distance * math.sin(new_angle)
    new_x, new_y = start_position[0] + new_dx, start_position[1] + new_dy
    assert mouse_position[0] > new_x, "new x is smaller than the mouse position"
    assert mouse_position[1] > new_y, "new y is smaller than the mouse position"
    
def test_clockwise_rotation_horizontal_left():
    start_position = (598, 376)
    mouse_position = (588, 376)
    dx, dy  = mouse_position[0] - start_position[0], mouse_position[1] - start_position[1]
    distance = math.hypot(dx, dy)
    # Assert distance travelled is 10 for vertical movement
    assert np.isclose(distance, 10)
    
    # Assert the angle is 0 degrees
    mouse_angle = math.atan2(dy, dx)
    # Assert that we are going to perturm the mouse angle by 30 degrees.
    perturbation_angle = math.radians(30)
    new_angle = mouse_angle + perturbation_angle
    # compute the new position.
    new_dx, new_dy = distance * math.cos(new_angle), distance * math.sin(new_angle)
    new_x, new_y = start_position[0] + new_dx, start_position[1] + new_dy
    # Assert the new position is (608, 376)
    assert mouse_position[0] < new_x, "new x is larger than the mouse position"
    assert mouse_position[1] > new_y, "new y is smaller than the mouse position"
    
def test_clockwise_rotation_horizontal_right():
    start_position = (598, 376)
    mouse_position = (608, 376)
    dx, dy  = mouse_position[0] - start_position[0], mouse_position[1] - start_position[1]
    distance = math.hypot(dx, dy)
    assert np.isclose(distance, 10)
    
    # Assert the angle is 0 degrees
    mouse_angle = math.atan2(dy, dx)
    # Assert that we are going to perturm the mouse angle by 30 degrees.
    perturbation_angle = math.radians(30)
    new_angle = mouse_angle + perturbation_angle
    # compute the new position.
    new_dx, new_dy = distance * math.cos(new_angle), distance * math.sin(new_angle)
    new_x, new_y = start_position[0] + new_dx, start_position[1] + new_dy
    # Assert the new position is (608, 376)
    assert mouse_position[0] > new_x, "new x is smaller than the mouse position"
    assert mouse_position[1] < new_y, "new y is larger than the mouse position"