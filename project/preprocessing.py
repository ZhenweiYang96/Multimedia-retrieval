###library and set directory
import os
import sys
import open3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from read_data import mesh_to_data

mesh_df = pd.read_excel('excel_file/standard_result.xlsx')

############################################
###Statistics
###Using histogram to check the distribution
###Undersample and oversample the outliers
############################################

average_size = sum(mesh_df['num_faces'])/len(mesh_df)
print(average_size)

###Histogram for the meshes
mesh_size = mesh_df['num_faces']
plt.hist(mesh_size, bins=20)
plt.show()


###Lower bounds are negative
#print(mesh_df['num_faces'].mean() - 2 * mesh_df[['num_faces']].std()) # lower bound
#print(mesh_df['num_vertices'].mean() - 2 * mesh_df[['num_faces']].std()) # lower bound

###Create df to check the size of the meshes
mesh_df['size'] = mesh_df['extent_x'] * mesh_df['extent_y'] * mesh_df['extent_z']

###Upper bound
upper_bound = mesh_df['size'].mean() + 2 * mesh_df['size'].std()

###Get all the outliers, aka size higher than upper bound
outlier = mesh_df[mesh_df['size'] > upper_bound]

###Mean of the size
mean_size = mesh_df['size'].mean()

###Directory
directory = 'data/backup/LabeledDB_new'
dirs = os.fsencode(directory)
file_dir = os.listdir(dirs)

###Print all outliers of the shape
counter = 0
for file in file_dir:
    file_name = os.fsdecode(file)
    ###the class of the shape
    class_shape = file_name
    ###going through the meshes of the class
    file_path = os.fsencode(directory + '/' + class_shape)
    for off_file in os.listdir(file_path):
        meshes_file = os.fsdecode(off_file)

        ###Catch the specific off file
        outlier_id = outlier['id'].iloc[counter]
        if meshes_file.endswith(str(outlier_id) + ".off"):
            counter += 1
            #mesh = open3d.io.read_triangle_mesh(directory + '/' + class_shape + '/' + meshes_file)
            #open3d.visualization.draw_geometries([mesh])
            if counter == len(outlier):
                break



###Print average shape
#print(mesh_df[np.logical_and(mesh_df['size'] > 1.408, mesh_df['size'] < 1.42)])
#mesh = open3d.io.read_triangle_mesh(directory + '/' + 'Airplane' + '/' + '77.off')
#open3d.visualization.draw_geometries([mesh])

###Which meshes are above the average
mesh_upper_outlier = mesh_df[mesh_df['num_faces'] > average_size * 1.2]

###Which meshes are below the average
mesh_lower_outlier = mesh_df[mesh_df['num_faces'] < average_size * 0.8]

###Preprocess the meshes if 50k faces, else normalize it directly
for mesh_id, class_shape in zip(mesh_df['id'], mesh_df['class_shape']):
    mesh = open3d.io.read_triangle_mesh(directory + '/' + class_shape + '/' + str(mesh_id) + '.off')

    ###Correspond with code 78
    #mesh.compute_vertex_normals()
    #open3d.visualization.draw_geometries([mesh])
    #print(mesh_id, class_shape)
    ###If the mesh is an outlier, process it into less faces
    #if mesh_id in mesh_upper_outlier['id']:

        ###Reduce vertices
        #voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 50
        #mesh = mesh.simplify_vertex_clustering(
            #voxel_size=voxel_size,
            #contraction=open3d.geometry.SimplificationContraction.Average)

        ###Step 2: showing the new values for the refined meshes.
        #print(mesh_smp)

    #open3d.visualization.draw_geometries([mesh_smp])
    #mesh_smp.compute_vertex_normals()
    #open3d.visualization.draw_geometries([mesh_smp])

    if mesh_id in mesh_lower_outlier['id'] or mesh_id in mesh_upper_outlier['id']:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(average_size))


    ###Normalization, this happens for all meshes
    bounding_box = open3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh)

    ###get the x y z
    extent_length = np.asarray(open3d.geometry.AxisAlignedBoundingBox.get_extent(bounding_box))

    ###choose the longest side (x/y/z)
    scale_value = max(extent_length[0], extent_length[1], extent_length[2])

    ###use the longest one to scale
    mesh = mesh.scale(1 / scale_value, center=mesh.get_center())
    #print(np.asarray(mesh.vertices))
    ###get the bounding box for the scaled mesh
    #bounding_box_aftnorm = open3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh)

    ###this step is to show that all shapes are scaled
    #print(np.asarray(open3d.geometry.AxisAlignedBoundingBox.get_extent(
     #   bounding_box_aftnorm)))
    #print('data/normalized\\' + class_shape + '\\' + str(mesh_id) + '.off')
    open3d.io.write_triangle_mesh('C:\\Users\\User\\Documents\\GitHub\\Multimedia-retrieval\\project\\data\\LabeledDB_new'+ '\\' + class_shape + '\\' + str(mesh_id) + '.off', mesh)

directory = 'C:\\Users\\User\\Documents\\GitHub\\Multimedia-retrieval\\project\\data\\LabeledDB_new'
mesh_to_data(directory, 'normalized_result')

############################################
###Statistics
###Using histogram to check the distribution
###Undersample and oversample the outliers
############################################

mesh_df = pd.read_excel('C:\\Users\\User\\Documents\\GitHub\\Multimedia-retrieval\\project\\excel_file\\normalized_result.xlsx')
average_size = sum(mesh_df['num_faces'])/len(mesh_df)
print(average_size)

###Histogram for the meshes
mesh_size = mesh_df['num_faces']
plt.hist(mesh_size, bins=20)
plt.show()
