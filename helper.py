import numpy as np
import math as m

# Crucial that'll determine the angle for the postures (i.e.)
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calc_dist(a,b):
    return m.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)