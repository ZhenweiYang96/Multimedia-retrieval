import os
import pandas as pd
from step3_function import *

directory = 'data\\LabeledDB_new'

dirs = os.fsencode(directory)
# print(dirs)
file_list = os.listdir(dirs)
counter = 0
a=[]
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
            mesh = open3d.io.read_triangle_mesh(directory + '/' + class_shape + '/' + meshes_file)
            v = np.asarray(mesh.vertices)
            unq, count = np.unique(v, axis=0, return_counts=True)
            if len(unq) < len(v):
                print(mesh_id)
            else:
                a.append(mesh_id)


print(len(a))
