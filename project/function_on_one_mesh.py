###library and set directory
from git_clone import *
from preprocess_function import *
from operator import itemgetter
from step3_function import *

# def check_properties(name, mesh):
#     mesh.compute_vertex_normals()
#
#     edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
#     edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
#     vertex_manifold = mesh.is_vertex_manifold()
#     self_intersecting = mesh.is_self_intersecting()
#     watertight = mesh.is_watertight()
#     orientable = mesh.is_orientable()
#
#     print(name)
#     print(f"  edge_manifold:          {edge_manifold}")
#     print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
#     print(f"  vertex_manifold:        {vertex_manifold}")
#     print(f"  self_intersecting:      {self_intersecting}")
#     print(f"  watertight:             {watertight}")
#     print(f"  orientable:             {orientable}")
#
#     geoms = [mesh]
#     if not edge_manifold:
#         edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
#         geoms.append(edges_to_lineset(mesh, edges, (1, 0, 0)))
#     if not edge_manifold_boundary:
#         edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
#         geoms.append(edges_to_lineset(mesh, edges, (0, 1, 0)))
#     if not vertex_manifold:
#         verts = np.asarray(mesh.get_non_manifold_vertices())
#         pcl = open3d.geometry.PointCloud(
#             points=open3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
#         pcl.paint_uniform_color((0, 0, 1))
#         geoms.append(pcl)
#     if self_intersecting:
#         intersecting_triangles = np.asarray(
#             mesh.get_self_intersecting_triangles())
#         intersecting_triangles = intersecting_triangles[0:1]
#         intersecting_triangles = np.unique(intersecting_triangles)
#         print("  # visualize self-intersecting triangles")
#         triangles = np.asarray(mesh.triangles)[intersecting_triangles]
#         edges = [
#             np.vstack((triangles[:, i], triangles[:, j]))
#             for i, j in [(0, 1), (1, 2), (2, 0)]
#         ]
#         edges = np.hstack(edges).T
#         edges = open3d.utility.Vector2iVector(edges)
#         geoms.append(edges_to_lineset(mesh, edges, (1, 0, 1)))
#     open3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)


#mesh = open3d.io.read_triangle_mesh("data/backup/LabeledDB_new/mech/323.off")
mesh = trimesh.load_mesh("data/backup/LabeledDB_new/mech/323.off")
mesh = flipping_test(mesh, 'flip')
mesh.export('data\\Normalized\\mech\\323.off')
mesh = open3d.io.read_triangle_mesh('data\\Normalized\\mech\\323.off')
mesh = mesh_sampling(mesh)
mesh = basic_normalization(mesh)
mesh = align_axis(mesh)
mesh = scaling(mesh)
open3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
mesh = mesh_sampling(mesh)
mesh = basic_normalization(mesh)
mesh = align_axis(mesh)
mesh = scaling(mesh)

#sorted_triangle = sorted(triangle, key=itemgetter(0, 1))

# wrong_triangles = []
# for tri in triangle:
#     counter = 0
#     for match_triangle in triangle:
#         matches = list(set(tri) & set(match_triangle))
#         if len(matches) == 2:
#             #print("complete list: ")
#             #print(triangle, match_triangle)
#
#             ###If match, get the two points that create the edge
#             a = np.where(tri == matches[0])
#             b = np.where(tri == matches[1])
#             if (a[0] == 0 and b[0] == 2) or (a[0] == 2 and b[0] == 0):
#                 edge_triangle = [tri[2], tri[0]]
#             else:
#                 edge_triangle = [x for x in tri if x in matches]
#
#             c = np.where(match_triangle == matches[0])
#             d = np.where(match_triangle == matches[1])
#             if (c[0] == 0 and d[0] == 2) or (c[0] == 2 and d[0] == 0):
#                 edge_match = [match_triangle[2], match_triangle[0]]
#             else:
#                 edge_match = [y for y in match_triangle if y in matches]
#             print("matches: ")
#             print(edge_triangle, edge_match)
#             counter += 1
#             if counter == 3:
#                 next


