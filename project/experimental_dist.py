###library and set directory
import os
import pandas as pd
import numpy as np
import random
import open3d
##################################
###Shape property distributions###
##################################
seed_matrix = np.arange(0, 1900).reshape(380, 5)

def vertex_output(mesh):
    vertex = np.asarray(mesh.vertices)
    num_vertex = vertex.shape[0]
    return vertex, num_vertex


def D3(mesh, number):
    vertex, num_vertex = vertex_output(mesh)
    area = []
    for i in range(0, number):
        sample = random.sample(range(0, num_vertex), 3)
        v1 = vertex[sample[1]] - vertex[sample[0]]
        v2 = vertex[sample[2]] - vertex[sample[0]]
        area.append(0.5 * np.linalg.norm(np.cross(v1, v2)))
    return area
# sns.kdeplot(area)


# D4
def D4(mesh, number):
    vertex, num_vertex = vertex_output(mesh)
    volume_D4 = []
    for i in range(0, number):
        sample = random.sample(range(0, num_vertex), 4)
        v1 = vertex[sample[1]] - vertex[sample[0]]
        v2 = vertex[sample[2]] - vertex[sample[0]]
        v3 = vertex[sample[3]] - vertex[sample[0]]
        volume_D4.append(1/6 * abs(np.dot(np.cross(v1, v2), v3)))
    return volume_D4


mesh_df = pd.DataFrame(
    columns=['D4_500k', 'D4_1mil', 'D4_2mil', 'D4_5mil'])



def mesh_dist():
    global mesh_df
    directory = 'data\\LabeledDB_new'
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
                D3_50k = D3(mesh, 500000)
                d3_100k = D3(mesh, 1000000)
                d3_250k = D3(mesh, 2000000)
                d3_500k = D3(mesh, 5000000)
                counter += 1
                mesh_df = mesh_df.append(
                    pd.DataFrame(
                        [[D3_50k, d3_100k, d3_250k, d3_500k]],
                        columns=mesh_df.columns))
                if counter == 11:
                    saved_directory = 'excel_file\\'
                    return mesh_df.to_csv(saved_directory + 'experimental' + '.csv', index=False)


mesh_dist()
