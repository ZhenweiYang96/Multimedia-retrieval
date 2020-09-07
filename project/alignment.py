###library and set directory
import os
import sys
import open3d
import pandas as pd
import numpy as np
from math import *

######STEP3:
mesh_test = open3d.io.read_triangle_mesh("data/LabeledDB_new/Airplane/63.off")
# center the shape
mesh_test.get_center() # this is the current center (it is not 0,0,0)
mesh_test.translate(-np.asarray(mesh_test.get_center())) # move it to 0,0,0
open3d.visualization.draw_geometries([mesh_test])
mesh_test.get_center() # now it is almost 0,0,0
# get eigenvectors
pcd = open3d.geometry.PointCloud()
pcd.points = mesh_test.vertices
pcd.colors = mesh_test.vertex_colors
pcd.normals = mesh_test.vertex_normals
print(pcd)
mean_covariance = pcd.compute_mean_and_covariance()
covariance = mean_covariance[1]
eigen_values, eigen_vectors = np.linalg.eig(covariance)
print("Eigenvector: \n", eigen_vectors, "\n")
print("Eigenvalues: \n", eigen_values, "\n")

###Position of the array, reverse it after sorting
position = np.argsort(eigen_values)[::-1]
print("Position: \n", position, "\n")

rotation_mat = np.column_stack((eigen_vectors[:, position[0]], eigen_vectors[:, position[1]],
                                eigen_vectors[:, position[2]])) # first column longest
rotation_mat
print("value 1: \n", eigen_values[position[0]])
print("value 2: \n", eigen_values[position[1]])
print("value 3: \n", eigen_values[position[2]])
###Rotation angle
theta1 = -np.arcsin(rotation_mat[2, 0]) # the is the rotation angle for y axis
theta2 = np.pi - theta1

psi1 = atan2(eigen_vectors[2, 1]/cos(theta1), eigen_vectors[2, 2]/cos(theta1)) # this is the rotation angle for x axis
psi2 = atan2(eigen_vectors[2, 1]/cos(theta2), eigen_vectors[2, 2]/cos(theta2))

fai1 = atan2(eigen_vectors[1, 0]/cos(theta1), eigen_vectors[0, 0]/cos(theta1)) # this is the rotation angle for z axis
fai2 = atan2(eigen_vectors[1, 0]/cos(theta2), eigen_vectors[0, 0]/cos(theta2))
###note: here both 1&2 are valid, so only use psi1, theta1 and fai1
R = mesh_test.get_rotation_matrix_from_xyz((-psi1, -theta1, -fai1))  # angle
###the angle will be multiplied with -1 because we want to align the eigenvector to original coordinate-frame
mesh_test.rotate(R, center=(0, 0, 0)) # rotate


###Flip the mesh upside down if need
#Todo

open3d.visualization.draw_geometries([mesh_test])


