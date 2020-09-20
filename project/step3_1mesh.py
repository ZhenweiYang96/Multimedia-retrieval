###library and set directory
import os
import sys
import open3d
import pandas as pd
import numpy as np
from math import *

mesh = open3d.io.read_triangle_mesh("data/LabeledDB_new/Human/1.off")

#####################################
###Setting the mesh to the right axis
###By detecting which part is the longest side,
###That part will be the x-axis, then the y-axis
###This is done by using PCA and eigenvectors
#####################################
mesh.get_center()  # this is the current center (it is not 0,0,0)
mesh.translate(-np.asarray(mesh.get_center()))  # move it to 0,0,0
# open3d.visualization.draw_geometries([mesh])
mesh.get_center()  # now it is almost 0,0,0
# get eigenvectors
pcd = open3d.geometry.PointCloud()
pcd.points = mesh.vertices
pcd.colors = mesh.vertex_colors
pcd.normals = mesh.vertex_normals
# print(pcd)
mean_covariance = pcd.compute_mean_and_covariance()
covariance = mean_covariance[1]
eigen_values, eigen_vectors = np.linalg.eig(covariance)
# print("Eigenvector: \n", eigen_vectors, "\n")
# print("Eigenvalues: \n", eigen_values, "\n")

###Position of the array, reverse it after sorting
position = np.argsort(eigen_values)[::-1]
print("Position: \n", position, "\n")

rotation_mat = np.column_stack((eigen_vectors[:, position[0]], eigen_vectors[:, position[1]],
                                eigen_vectors[:, position[2]]))  # first column longest

print("rotation matrix: ", rotation_mat)
###Rotation angle
theta1 = -np.arcsin(rotation_mat[2, 0])  # the is the rotation angle for y axis
theta2 = np.pi - theta1

psi1 = atan2(eigen_vectors[2, 1] / cos(theta1),
             eigen_vectors[2, 2] / cos(theta1))  # this is the rotation angle for x axis
psi2 = atan2(eigen_vectors[2, 1] / cos(theta2), eigen_vectors[2, 2] / cos(theta2))

fai1 = atan2(eigen_vectors[1, 0] / cos(theta1),
             eigen_vectors[0, 0] / cos(theta1))  # this is the rotation angle for z axis
fai2 = atan2(eigen_vectors[1, 0] / cos(theta2), eigen_vectors[0, 0] / cos(theta2))
###note: here both 1&2 are valid, so only use psi1, theta1 and fai1
R = mesh.get_rotation_matrix_from_xyz((-psi1, -theta1, -fai1))  # angle
###the angle will be multiplied with -1 because we want to align the eigenvector to original coordinate-frame
mesh.rotate(R, center=(0, 0, 0))  # rotate

################################################
###Flipping
###If the mesh is not flipped, then the mesh
###needs to be flipped. If not, then no flipping
###is needed
################################################

###Which point the triangle is made of
triangle = np.asarray(mesh.triangles)

###The coordination of the points
vertices = np.asarray(mesh.vertices)
#print(vertices[0])
#print(triangle)
#print("length vertices:", len(vertices))


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
F = np.matrix([[np.sign(fi[0]), 0, 0, 0],
               [0, np.sign(fi[1]), 0, 0],
               [0, 0, np.sign(fi[2]), 0],
               [0, 0, 0, 1]])
print('F: ', F)

mesh = mesh.transform(F)
open3d.visualization.draw_geometries([mesh])


#################################
###Step 3.2: Feature extraction
###surface area
###compactness (with respect to a sphere)
###axis-aligned bounding-box volume
###diameter
###eccentricity (ratio of largest to smallest eigenvalues of covariance matrix)
#################################

#surface_area = mesh.get_surface_area()
