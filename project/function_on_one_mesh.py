###library and set directory
import os
import sys
import open3d
import pandas as pd
import numpy as np
from math import *
from git_clone import *
from preprocessing import *


def check_properties(name, mesh):
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(name)
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")

    geoms = [mesh]
    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        geoms.append(edges_to_lineset(mesh, edges, (1, 0, 0)))
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        geoms.append(edges_to_lineset(mesh, edges, (0, 1, 0)))
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        pcl = open3d.geometry.PointCloud(
            points=open3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
        pcl.paint_uniform_color((0, 0, 1))
        geoms.append(pcl)
    if self_intersecting:
        intersecting_triangles = np.asarray(
            mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print("  # visualize self-intersecting triangles")
        triangles = np.asarray(mesh.triangles)[intersecting_triangles]
        edges = [
            np.vstack((triangles[:, i], triangles[:, j]))
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ]
        edges = np.hstack(edges).T
        edges = open3d.utility.Vector2iVector(edges)
        geoms.append(edges_to_lineset(mesh, edges, (1, 0, 1)))
    open3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)


mesh = open3d.io.read_triangle_mesh("data/LabeledDB_new/Human/1.off")
mesh = basic_normalization(mesh)
mesh = mesh_samplng(mesh)
mesh = align_axis(mesh)

#mesh = flipping(mesh)
#help(open3d.geometry.PointCloud)
mesh.compute_vertex_normals()
#check_properties('human', mesh)
triangle = np.asarray(mesh.triangles)
vertex = np.asarray(mesh.vertices)


triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
cluster_area = np.asarray(cluster_area)

print(triangle_clusters)
print(cluster_n_triangles)
print(cluster_area)

longest = []
for i in range(2002):
    triangle_record = []
    for j in triangle:
        if i in j:
            triangle_record.append(j)
            if len(triangle_record) > len(longest):
                longest = triangle_record

#print(longest)
#print(len(longest))

