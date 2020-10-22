import os
import pandas as pd
from step3_function import *
import open3d
import time

seed_matrix = np.arange(0, 1900).reshape(380, 5)


def feature_extraction(directory, name):
    mesh_df = pd.DataFrame(
        columns=['mesh_id', 'class', 'surface_area', 'sphericity', 'bounding_box_volume', 'diameter', 'eccentricity',
                 'A3', 'D1', 'D2', 'D3', 'D4'])
    dirs = os.fsencode(directory)
    # print(dirs)
    file_list = os.listdir(dirs)
    counter = 0
    for file in file_list:
        file_name = os.fsdecode(file)
        ###the class of the shape
        class_shape = file_name
        ###going through the meshes of the class
        file_path = os.fsencode(directory + '\\' + class_shape)
        for off_file in os.listdir(file_path):
            meshes_file = os.fsdecode(off_file)
            ###Catch all the OFF files
            if meshes_file.endswith(".off"):
                print(counter)
                mesh_path = directory + '\\' + class_shape + '\\' + meshes_file
                ###Get id of the mesh
                sep = '.'
                mesh_id = meshes_file.split(sep, 1)[0]
                # print(directory + '\\' + class_shape + '\\' + meshes_file)
                mesh = open3d.io.read_triangle_mesh(mesh_path)
                convex_surface = calc_convex_surface(mesh_path)
                sphericity = calc_sphericity(mesh_path)
                bounding_box_volume = calc_bounding_box_volume(mesh)
                diameter = calc_diameter(mesh)
                eccentricity = calc_eccentricity(mesh)
                a3 = A3(mesh, seed=seed_matrix[counter, 0])
                d1 = D1(mesh, seed=seed_matrix[counter, 1])
                d2 = D2(mesh, seed=seed_matrix[counter, 2])
                d3 = D3(mesh, seed=seed_matrix[counter, 3])
                d4 = D4(mesh, seed=seed_matrix[counter, 4])
                counter += 1
                mesh_df = mesh_df.append(
                    pd.DataFrame([[mesh_id, class_shape, convex_surface, sphericity, bounding_box_volume, diameter, eccentricity,
                                   a3, d1, d2, d3, d4]],
                                 columns=mesh_df.columns))

    saved_directory = 'excel_file\\'
    print(saved_directory + name + '.csv')
    return mesh_df.to_csv(saved_directory + name + '.csv', index=False)  # save an csv

t0 = time.time()
feature_extraction('data\\Normalized', 'features')
t1 = time.time()
print(t1 - t0)