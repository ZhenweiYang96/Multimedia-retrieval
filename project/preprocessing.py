###library and set directory
from preprocess_function import *
import matplotlib.pyplot as plt
import trimesh
from collections import Counter
from read_data import mesh_to_data
import time
import pandas as pd
import open3d
import os
mesh_df = pd.read_excel('excel_file\\standard_result.xlsx')
############################################
###Statistics
###Using histogram to check the distribution
###Undersample and oversample the outliers
############################################
print('1')
###Histogram for the meshes
mesh_size = mesh_df['num_faces']
plt.hist(mesh_size, bins=20, rwidth = 0.7)
plt.xlabel('The number of faces')
plt.ylabel('Frequency')
plt.show()

# distance to the origin
distance = round(mesh_df['distance from barycenter to the origin'], 1)
plt.hist(distance, bins=20)
plt.xlabel('The distance from barycenter to the origin')
plt.ylabel('Frequency')
plt.show()

# volume
vol_diff = mesh_df['max_length'] ** 3 - 1
plt.hist(vol_diff,bins = 20, rwidth = 0.7)
plt.xlabel('difference between current cube volume to unit cube volume')
plt.ylabel('Frequency')
plt.show()

# cos(major eigenvector
cos_majorangle_aftnorm = mesh_df['major angle']
plt.hist(cos_majorangle_aftnorm, bins=20,rwidth = 0.7)
plt.xlabel('cosine of the angle between major vector and x axis')
plt.ylabel('Frequency')
plt.show()

cos_secondangle_aftnorm = mesh_df['second angle']
plt.hist(cos_secondangle_aftnorm, bins=20,rwidth = 0.7)
plt.xlabel('cosine of the angle between medium vector and x axis')
plt.ylabel('Frequency')
plt.show()

###Directory
directory = 'data\\backup\\LabeledDB_new'
dirs = os.fsencode(directory)
file_dir = os.listdir(dirs)

is_flipped = []
counter = 0
###Preprocess the meshes

for mesh_id, class_shape in zip(mesh_df['id'], mesh_df['class_shape']):
    print(counter)
    mesh = open3d.io.read_triangle_mesh(directory + '/' + class_shape + '/' + str(mesh_id) + '.off')
    mesh = mesh_sampling(mesh)
    open3d.io.write_triangle_mesh('data\\Normalized' + '\\' + class_shape + '\\' + str(mesh_id) + '.off', mesh)
    mesh = trimesh.load_mesh('data\\Normalized' + '\\' + class_shape + '\\' + str(mesh_id) + '.off')
    mesh = basic_normalization(mesh)
    mesh = align_axis(mesh)
    mesh = flipping(mesh, 'flip')
    mesh = scaling(mesh)
    #coordinate = open3d.geometry.TriangleMesh.create_coordinate_frame()
    #open3d.visualization.draw_geometries([mesh, coordinate], mesh_show_wireframe=True)

    is_flipped.append(flipping(mesh, 'hist'))
    mesh.export('data\\Normalized' + '\\' + class_shape + '\\' + str(mesh_id) + '.off')
    counter += 1


# coordinate = open3d.geometry.TriangleMesh.create_coordinate_frame()
# open3d.visualization.draw_geometries([mesh, coordinate], mesh_show_wireframe=True)
# open3d.visualization.draw_geometries([mesh, coordinate], mesh_show_wireframe=True)


directory = 'data\\Normalized'

t0 = time.time()
mesh_to_data(directory, 'normalized_result')
t1 = time.time()

print(t1-t0)
############################################
###Statistics
###Using histogram to check the distribution
###Undersample and oversample the outliers
############################################

mesh_df_aftnorm = pd.read_excel('excel_file\\normalized_result.xlsx')
mesh_df_aftnorm = pd.DataFrame(mesh_df_aftnorm)
mesh_size_aftnorm = mesh_df_aftnorm['num_faces'].tolist()
#mesh_size_aftnorm = [int(x) for x in mesh_size_aftnorm]
plt.hist(mesh_size_aftnorm, bins=5)
plt.xlabel('The number of faces')
plt.ylabel('Frequency')
#plt.xlim(0,10000)
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
plt.xlabel('cosine of the angle between medium vector and x axis')
plt.ylabel('Frequency')
plt.show()

###Flipping
is_flipped = np.array(is_flipped)
print(is_flipped)
plt.hist(is_flipped, bins=20)
plt.xlabel('Flipped or not?')
plt.ylabel('Frequency')
plt.show()

