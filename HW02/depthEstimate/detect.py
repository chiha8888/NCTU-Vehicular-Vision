import numpy as np

def depth_estimation(img, bb):
    focalLength = 300
    knownWidth = 1.65
    perWidth = abs(bb[0][1] - bb[1][1])
    return (knownWidth * focalLength) / perWidth
