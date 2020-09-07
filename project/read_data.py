# library and set directory
import os
import sys
import open3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import HTML

# output a table with points （点型）
mesh_df = pd.DataFrame(
    columns=[
        "id",
        "class_shape",
        "num_faces",
        "num_vertices",
        "type_shape",
        "bounding_box_points",
        "extent_length",
        "extent_x",
        "extent_y",
        "extent_z",
    ]
)
directory = "data/LabeledDB_new"
dirs = os.fsencode(directory)
file_list = os.listdir(dirs)
for file in file_list:
    file_name = os.fsdecode(file)
    ###the class of the shape
    class_shape = file_name
    ###going through the meshes of the class
    file_path = os.fsencode(directory + "/" + class_shape)
    for off_file in os.listdir(file_path):
        meshes_file = os.fsdecode(off_file)
        ###Catch all the OFF files
        if meshes_file.endswith(".off"):

            ###Get id of the mesh
            sep = "."
            mesh_id = meshes_file.split(sep, 1)[0]

            mesh = open3d.io.read_triangle_mesh(
                directory + "/" + class_shape + "/" + meshes_file
            )
            ###the number of faces and vertices of the shape
            mesh_triangles = len(np.asarray(mesh.triangles))
            mesh_vertices = len(np.asarray(mesh.vertices))
            ###the type of faces
            faces_type = "triangles"
            ###the axis-aligned 3D bounding box of the shapes
            bounding_box = (
                open3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(
                    mesh
                )
            )
            bounding_box_points = np.asarray(
                open3d.geometry.AxisAlignedBoundingBox.get_box_points(bounding_box)
            )  # get bounding points
            extent_length = np.asarray(
                open3d.geometry.AxisAlignedBoundingBox.get_extent(bounding_box)
            )  # get length
            extent_x = extent_length[0]
            extent_y = extent_length[1]
            extent_z = extent_length[2]

            mesh_df = mesh_df.append(
                pd.DataFrame(
                    [
                        [
                            mesh_id,
                            class_shape,
                            mesh_triangles,
                            mesh_vertices,
                            faces_type,
                            bounding_box_points,
                            extent_length,
                            extent_x,
                            extent_y,
                            extent_z,
                        ]
                    ],
                    columns=mesh_df.columns,
                )
            )

mesh_df.to_excel("excel_file/results.xlsx", index=False)  # save an excel
