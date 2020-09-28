###library and set directory
import os
import sys
import open3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from read_data import mesh_to_data
from math import *

mesh_df = pd.read_excel('excel_file/standard_result.xlsx')
############################################
###Statistics
###Using histogram to check the distribution
###Undersample and oversample the outliers
############################################

###Histogram for the meshes
mesh_size = mesh_df['num_faces']
plt.hist(mesh_size, bins=20)
plt.xlabel('The number of faces')
plt.ylabel('Frequency')
# plt.show()

# distance to the origin
distance = round(mesh_df['distance from barycenter to the origin'], 1)
plt.hist(distance, bins=20)
plt.xlabel('The distance from barycenter to the origin')
plt.ylabel('Frequency')
# plt.show()

# volume
vol_diff = mesh_df['max_length'] ** 3 - 1
plt.hist(vol_diff)
plt.xlabel('difference between current cube volume to unit cube volume')
plt.ylabel('Frequency')
# plt.show()

###Directory
directory = 'data/backup/LabeledDB_new'
dirs = os.fsencode(directory)
file_dir = os.listdir(dirs)

###Print average shape
# print(mesh_df[np.logical_and(mesh_df['size'] > 1.408, mesh_df['size'] < 1.42)])
# mesh = open3d.io.read_triangle_mesh(directory + '/' + 'Airplane' + '/' + '77.off')
# open3d.visualization.draw_geometries([mesh])

# assign the target face count
face_count = 4000

###Which meshes are above the average
mesh_upper_outlier = mesh_df[mesh_df['num_faces'] > face_count * 1.15]

###Which meshes are below the average
mesh_lower_outlier = mesh_df[mesh_df['num_faces'] < face_count * 0.85]

###Preprocess the meshes if 50k faces, else normalize it directly
for mesh_id, class_shape in zip(mesh_df['id'], mesh_df['class_shape']):
    mesh = open3d.io.read_triangle_mesh(directory + '/' + class_shape + '/' + str(mesh_id) + '.off')

    # translation to the barycenter
    mesh = mesh.translate(-np.asarray(mesh.get_center()))
    ###If the mesh is an outlier, process it into less faces
    # if mesh_id in mesh_upper_outlier['id']:

    ###Reduce vertices
    # voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 50
    # mesh = mesh.simplify_vertex_clustering(
    # voxel_size=voxel_size,
    # contraction=open3d.geometry.SimplificationContraction.Average)

    ###Step 2: showing the new values for the refined meshes.
    # print(mesh_smp)
    ###Normalization, this happens for all meshes
    bounding_box = open3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh)

    ###get the x y z
    extent_length = np.asarray(open3d.geometry.AxisAlignedBoundingBox.get_extent(bounding_box))

    ###choose the longest side (x/y/z)
    scale_value = max(extent_length[0], extent_length[1], extent_length[2])

    ###use the longest one to scale
    mesh = mesh.scale(1 / scale_value, center=mesh.get_center())
    # print(np.asarray(mesh.vertices))
    ###get the bounding box for the scaled mesh
    # bounding_box_aftnorm = open3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh)
    # open3d.visualization.draw_geometries([mesh_smp])
    # mesh_smp.compute_vertex_normals()
    # open3d.visualization.draw_geometries([mesh_smp])

    if mesh_id in list(mesh_upper_outlier['id']):
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(face_count))
    elif mesh_id in list(mesh_lower_outlier['id']):
        mesh = mesh.subdivide_loop(number_of_iterations=1)
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(face_count))

    # make a 3 * n matrix for the coordinates
    A = np.asmatrix(np.transpose(np.asarray(mesh.vertices)))

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

    mesh = mesh.rotate(rotation_mat, center=mesh.get_center())  # rotate
    coordinate = open3d.geometry.TriangleMesh.create_coordinate_frame()
    #open3d.visualization.draw_geometries([mesh, coordinate], mesh_show_wireframe=True)
    #open3d.visualization.draw_geometries([mesh, coordinate], mesh_show_wireframe=True)
    triangle = np.asarray(mesh.triangles)
    vertice = np.asarray(mesh.vertices)
    ###Get the center of each triangle
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
    mesh = mesh.transform(F)
    coordinate = open3d.geometry.TriangleMesh.create_coordinate_frame()
    # save meshes
    open3d.io.write_triangle_mesh(
        'C:\\Users\\User\\Documents\\GitHub\\Multimedia-retrieval\\project\\data\\LabeledDB_new' + '\\' + class_shape + '\\' + str(
            mesh_id) + '.off', mesh)

directory = 'C:\\Users\\User\\Documents\\GitHub\\Multimedia-retrieval\\project\\data\\LabeledDB_new'
mesh_to_data(directory, 'normalized_result')

############################################
###Statistics
###Using histogram to check the distribution
###Undersample and oversample the outliers
############################################

mesh_df_aftnorm = pd.read_excel('excel_file\\normalized_result.xlsx')
mesh_df_aftnorm = pd.DataFrame(mesh_df_aftnorm)
mesh_size_aftnorm = mesh_df_aftnorm['num_faces']
plt.hist(mesh_size_aftnorm, bins=5)
plt.xlabel('The number of faces')
plt.ylabel('Frequency')
plt.xlim(0, 10000)
plt.show()

###distance to the origin
distance_aftnorm = round(mesh_df_aftnorm['distance from barycenter to the origin'], 1)
plt.hist(distance_aftnorm, bins=20)
plt.xlabel('The distance from barycenter to the origin')
plt.ylabel('Frequency')
plt.xlim(0, 0.8)
plt.show()

###volume
vol_diff_aftnorm = round(mesh_df_aftnorm['max_length'] ** 3 - 1, 1)
plt.hist(vol_diff_aftnorm)
plt.xlabel('difference between current volume to unit cube volume')
plt.ylabel('Frequency')
plt.xlim(-1, 7)
plt.show()


# cos(angle)
cos_majorangle_aftnorm = mesh_df_aftnorm['major angle']
plt.hist(cos_majorangle_aftnorm, bins=20)
plt.xlabel('cosine of the angle between major vector and x axis')
plt.ylabel('Frequency')
plt.show()

cos_secondangle_aftnorm = mesh_df_aftnorm['second angle']
plt.hist(cos_secondangle_aftnorm, bins=20)
plt.xlabel('cosine of the angle between second vector and x axis')
plt.ylabel('Frequency')
plt.show()
