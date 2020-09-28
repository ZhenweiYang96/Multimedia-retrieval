###library and set directory
import os
import sys
import open3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import *


mesh_df = pd.read_excel('excel_file/standard_result.xlsx')
# assign the target face count
face_count = 4000

###Which meshes are above the average
mesh_upper_outlier = mesh_df[mesh_df['num_faces'] > face_count * 1.15]

###Which meshes are below the average
mesh_lower_outlier = mesh_df[mesh_df['num_faces'] < face_count * 0.85]

###Directory
directory = 'data/backup/LabeledDB_new'
dirs = os.fsencode(directory)
file_list = os.listdir(dirs)

for mesh_id, class_shape in zip(mesh_df['id'], mesh_df['class_shape']):
    mesh = open3d.io.read_triangle_mesh(directory + '/' + class_shape + '/' + str(mesh_id) + '.off')
    #open3d.visualization.draw_geometries([mesh])

    mesh_mv = mesh.translate(-np.asarray(mesh.get_center()))

    ###Normalization, this happens for all meshes
    bounding_box = open3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh_mv)
    ###get the x y z
    extent_length = np.asarray(open3d.geometry.AxisAlignedBoundingBox.get_extent(bounding_box))
    ###choose the longest side (x/y/z)
    scale_value = max(extent_length[0], extent_length[1], extent_length[2])
    ###use the longest one to scale
    mesh_mv = mesh_mv.scale(1 / scale_value, center=mesh_mv.get_center())
    # print(np.asarray(mesh.vertices))
    ###get the bounding box for the scaled mesh
    # bounding_box_aftnorm = open3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh)
    # open3d.visualization.draw_geometries([mesh_smp])
    # mesh_smp.compute_vertex_normals()
    # open3d.visualization.draw_geometries([mesh_smp])
    if 1 in list(mesh_upper_outlier['id']):
        mesh_mv = mesh_mv.simplify_quadric_decimation(
            target_number_of_triangles=int(face_count))
    elif 1 in list(mesh_lower_outlier['id']):
        mesh_mv = mesh_mv.subdivide_loop(number_of_iterations=1)
        mesh_mv = mesh_mv.simplify_quadric_decimation(
            target_number_of_triangles=int(face_count))
    # make a 3 * n matrix for the coordinates
    A = np.asmatrix(np.transpose(np.asarray(mesh_mv.vertices)))
    #print(A)
    # covariance matrix
    A_cov = np.cov(A)  # 3x3 matrix
    # eigenvectors and eigenvalues
    eigen_values, eigen_vectors = np.linalg.eig(A_cov)

    position = np.argsort(eigen_values)[::-1] # align longest eigenvector with x axis
    x1 = eigen_vectors[:, position[0]]
    y1 = eigen_vectors[:, position[1]]
    z1 = eigen_vectors[:, position[2]]
    rotation_mat = np.linalg.inv(np.column_stack((eigen_vectors[:, position[0]],
                                                  eigen_vectors[:, position[1]],
                                                  eigen_vectors[:, position[2]])))

    mesh_mv = mesh_mv.rotate(rotation_mat, center=mesh_mv.get_center())  # rotate
    #coordinate = open3d.geometry.TriangleMesh.create_coordinate_frame()
    #open3d.visualization.draw_geometries([mesh, coordinate], mesh_show_wireframe=True)
    #open3d.visualization.draw_geometries([mesh_mv, coordinate], mesh_show_wireframe=True)
    triangle = np.asarray(mesh_mv.triangles)
    vertice = np.asarray(mesh_mv.vertices)
    ###Get the center of each triangle
    coordinate = open3d.geometry.TriangleMesh.create_coordinate_frame()
    open3d.visualization.draw_geometries([mesh_mv, coordinate], mesh_show_wireframe=True)
    ###Total sum
    fi = [0, 0, 0]
    for i in triangle:
        j = i
        coordinates = []
        ###Get coordinates of the points
        for j in i:
            # print("coordinates of the points: ", vertices[j])
            ###List of the coordinates of the three points
            coordinates.append(vertice[j])
        ###sum the coordinates to get the center point
        x_coord = sum(k[0] for k in coordinates) / 3
        y_coord = sum(k[1] for k in coordinates) / 3
        z_coord = sum(k[2] for k in coordinates) / 3
        ###Store coordinates in numpy
        center_coord = [x_coord, y_coord, z_coord]
        fi += np.sign(center_coord) * np.square(center_coord)
    ###Transformation matrix
    # print(fi[0], fi[1], fi[2])
    F = np.matrix([[np.sign(fi[0]), 0, 0, 0],
                   [0, np.sign(fi[1]), 0, 0],
                   [0, 0, np.sign(fi[2]), 0],
                   [0, 0, 0, 1]])
    # print('F: ', F)
    mesh_mv = mesh_mv.transform(F)

    open3d.visualization.draw_geometries([mesh_mv, coordinate], mesh_show_wireframe=True)