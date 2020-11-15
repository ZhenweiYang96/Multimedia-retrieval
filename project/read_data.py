# library and set directory
import os
import sys
import open3d
import pandas as pd
import numpy as np
import copy
from math import *
import trimesh
from preprocess_function import calc_eigen
# output a table with points （点型）


#visual_data('data\\LabeledDB_new\\Human\\1.off',2,0,1)


#mesh = open3d.io.read_triangle_mesh('data\\backup\\LabeledDB_new\\Human\\1.off')
#mesh.compute_vertex_normals()
#open3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)


def mesh_to_data(directory, name):

    mesh_df = pd.DataFrame(
        columns=['id', 'class_shape', 'num_faces', 'num_vertices', 'type_shape', 'barycenter',
                 'distance from barycenter to the origin', 'extent_length',
                 'extent_x', 'extent_y', 'extent_z', 'max_length', 'major angle',
                 'second angle'])
    dirs = os.fsencode(directory)
    #print(dirs)
    file_list = os.listdir(dirs)
    for file in file_list:
        file_name = os.fsdecode(file)
        ###the class of the shape
        class_shape = file_name
        ###going through the meshes of the class
        file_path = os.fsencode(directory + '/' + class_shape)
        for off_file in os.listdir(file_path):
            meshes_file = os.fsdecode(off_file)
            ###Catch all the OFF files
            if meshes_file.endswith(".off"):
                ###Get id of the mesh
                sep = '.'
                mesh_id = meshes_file.split(sep, 1)[0]
                # print(directory + '\\' + class_shape + '\\' + meshes_file)
                mesh = trimesh.load(directory + '/' + class_shape + '/' + meshes_file)
                #mesh = open3d.io.read_triangle_mesh(directory + '/' + class_shape + '/' + meshes_file)
                ###the number of faces and vertices of the shape
                mesh_triangles = len(mesh.faces)
                mesh_vertices = len(mesh.vertices)
                ###the type of faces
                faces_type = "triangles"
                ###barycenter
                mesh_barycenter = mesh.centroid
                mesh_distance_to_origin = np.linalg.norm(mesh_barycenter)
                ###the axis-aligned 3D bounding box of the shapess
                extent_length = mesh.bounding_box.extents
                extent_x = extent_length[0]
                extent_y = extent_length[1]
                extent_z = extent_length[2]
                max_extent = max(extent_x, extent_y, extent_z)
                #bounding_box_volume = bounding_box.volume()

                A = np.asmatrix(np.transpose(mesh.vertices))
                # covariance matrix
                A_cov = np.cov(A)  # 3x3 matrix
                # eigenvectors and eigenvalues
                eigen_values, eigen_vectors = np.linalg.eig(A_cov)
                position = np.argsort(eigen_values)[::-1]

                major_eigen = eigen_vectors[:, position[0]]
                cos_major_angle = (major_eigen[0] * 1) / (
                    sqrt(major_eigen[0] ** 2 + major_eigen[1] ** 2 + major_eigen[2] ** 2))

                second_eigen = eigen_vectors[:, position[1]]
                cos_second_angle = (second_eigen[1] * 1) / (
                    sqrt(second_eigen[0] ** 2 + second_eigen[1] ** 2 + major_eigen[2] ** 2))
                mesh_df = mesh_df.append(pd.DataFrame([[mesh_id, class_shape, mesh_triangles, mesh_vertices, faces_type,
                                                        mesh_barycenter, mesh_distance_to_origin,
                                                        extent_length, extent_x, extent_y,
                                                        extent_z, max_extent, cos_major_angle,
                                                        cos_second_angle]],
                                                      columns=mesh_df.columns))
    saved_directory = 'excel_file\\'
    print(saved_directory + name + '.xlsx')
    return mesh_df.to_excel(saved_directory + name + '.xlsx', index=False)  # save an excel


mesh_to_data('data\\LabeledDB_new', 'standard_result')



