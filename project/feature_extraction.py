#!/bin/python3
"""
Feature extraction module
"""

###library and set directory
import os
import sys
import open3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import *


mesh_df = pd.read_excel("excel_file/results.xlsx")

###Directory
directory = "data/LabeledDB_new"
dirs = os.fsencode(directory)
file_list = os.listdir(dirs)

for mesh_id, class_shape in zip(mesh_df["id"], mesh_df["class_shape"]):
    mesh = open3d.io.read_triangle_mesh(
        directory + "/" + class_shape + "/" + str(mesh_id) + ".off"
    )

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
    # print("Position: \n", position, "\n")

    rotation_mat = np.column_stack(
        (
            eigen_vectors[:, position[0]],
            eigen_vectors[:, position[1]],
            eigen_vectors[:, position[2]],
        )
    )  # first column longest

    ###Rotation angle
    theta1 = -np.arcsin(rotation_mat[2, 0])  # the is the rotation angle for y axis
    theta2 = np.pi - theta1

    psi1 = atan2(
        eigen_vectors[2, 1] / cos(theta1), eigen_vectors[2, 2] / cos(theta1)
    )  # this is the rotation angle for x axis
    psi2 = atan2(eigen_vectors[2, 1] / cos(theta2), eigen_vectors[2, 2] / cos(theta2))

    fai1 = atan2(
        eigen_vectors[1, 0] / cos(theta1), eigen_vectors[0, 0] / cos(theta1)
    )  # this is the rotation angle for z axis
    fai2 = atan2(eigen_vectors[1, 0] / cos(theta2), eigen_vectors[0, 0] / cos(theta2))
    ###note: here both 1&2 are valid, so only use psi1, theta1 and fai1
    R = mesh.get_rotation_matrix_from_xyz((-psi1, -theta1, -fai1))  # angle
    ###the angle will be multiplied with -1 because we want to align the eigenvector to original coordinate-frame
    mesh.rotate(R, center=(0, 0, 0))  # rotate

    # Flip the mesh upside down if need
    eccentricity = abs(eigen_values[position[0]]) / abs(eigen_values[position[1]])
    print("eccentricity: \n", eccentricity)
    print(mesh_id, class_shape)
    flipping = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    mesh = mesh.transform(flipping)

    open3d.visualization.draw_geometries([mesh])
