###library and set directory
import os
import sys
import open3d
import pandas as pd
import numpy as np
from math import *

mesh = open3d.io.read_triangle_mesh("data/LabeledDB_new/Human/1.off")

###Which point the triangle is made of
triangle = np.asarray(mesh.triangles)

###The coordination of the points
vertices = np.asarray(mesh.vertices)
#print(vertices[0])
#print(triangle)
#print("length vertices:", len(vertices))

counter = 0

###Total sum
fi = [0, 0, 0]

###Get the center of each triangle
for i in triangle:
    #print("triangle point: ", i)
    j = i
    coordinates = []

    ###Get coordinates of the points
    for j in i:
        #print("coordinates of the points: ", vertices[j])
        ###List of the coordinates of the three points
        coordinates.append(vertices[j])
    ###sum the coordinates to get the center point
    x_coord = sum(k[0] for k in coordinates)/3
    y_coord = sum(k[1] for k in coordinates)/3
    z_coord = sum(k[2] for k in coordinates)/3
    #print("coordinate: ", x_coord, y_coord, z_coord)

    ###Store coordinates in numpy
    center_coord = [x_coord, y_coord, z_coord]

    fi += np.sign(center_coord) * np.square(center_coord)

###Transformation matrix
print(fi[0], fi[1], fi[2])
F = np.matrix([[np.sign(fi[0]), 0, 0], [0, np.sign(fi[1]), 0], [0, 0, np.sign(fi[2])]])
print('F: ', F)


###Flipping code
###But since the flipping is a matrix of 4x4, we need to know how to flip it with 3x3
#flipping = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
#mesh_test = mesh_test.transform(flipping)
