import pymesh
import numpy as np
from sklearn.decomposition import PCA


def fix_mesh(mesh, detail="normal", max_vertices=20000):
    bbox_min, bbox_max = mesh.bbox
    diag_len = np.linalg.norm(bbox_max - bbox_min)
    if type(detail) == str:
        if detail == "normal":
            target_len = diag_len * 5e-3
        elif detail == "high":
            target_len = diag_len * 2.5e-3
        elif detail == "low":
            target_len = diag_len * 1e-2
    else:
        target_len = detail
    print("Target resolution: {} mm".format(target_len))
    
    tol = 0.01
    mesh, info = pymesh.remove_duplicated_vertices(mesh, target_len * tol)
    print(info)

    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                                               preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices or mesh.num_vertices > max_vertices:
            break

        num_vertices = mesh.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1
        if count > 10: break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh


def compute_pca_alignement(M, N):
    """Compute the PCA rigid alignement between a mesh M and a template mesh N
    and return the rotation R, translation T and scale S matrices.
    args:
        - M: pymesh, the mesh to deform
        - N: pymesh, the template mesh
    return:
        - R: array(3,3), the rotation matrix to align M to N
        - T: array(3), the translation matrix to align M to N
        - S: array(3), the scale matrix to align M to N
    """
    pca_M = PCA(3).fit(M.vertices)
    pca_N = PCA(3).fit(N.vertices)

    norm_axis_M = pca_M.components_ / np.linalg.norm(pca_M.components_, axis=1)
    norm_axis_N = pca_N.components_ / np.linalg.norm(pca_N.components_, axis=1)
    R = np.einsum("ij,ik->ijk", norm_axis_M, norm_axis_N).sum(0)  # Rotation
    S = np.sqrt(pca_N.singular_values_ / pca_M.singular_values_)  # scale
    T = pca_N.mean_ - pca_M.mean_ @ R * S  # Translation
    return R, T, S


def merge_rigid_deformations(list_R, list_T, list_S):
    """Merge rigid deformations in one R, T, S"""
    assert len(list_R) == len(list_T) == len(list_S), "Not the same number of transformations"
    R, T, S = np.eye(3), np.zeros(3), np.ones(3)
    for i in range(len(list_R)):
        R = R @ list_R[i]
        S = S * list_S[i]
        T = T @ list_R[i]*list_S[i] + list_T[i]
    return R, T, S


def align_mesh(M, R, T, S):
    return pymesh.form_mesh(M.vertices @ R * S + T, M.faces)

def compute_laplacian(V, F):
    """
    Compute the Laplacian of the mesh
    
    args:
        V: (N x 3) a numpy array of vertices' positions    ex: [ (x0,y0,z0), (x1,y1,z1) ...]
        F: (M x 3) a numpy array of the faces of the triangular mesh (by vertex id) ex: [ (0,1,4), (2,4,5), ...]
    return:
        L: sparce_csr(N x N) the Laplacian matrix with cotangent weights
    """
    n = len(V)
    W_ij = np.empty(0)
    I = np.empty(0, np.int32)
    J = np.empty(0, np.int32)
    for i1, i2, i3 in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        vi1 = F[:, i1]
        vi2 = F[:, i2]
        vi3 = F[:, i3]
        u = V[vi2] - V[vi1]
        v = V[vi3] - V[vi1]

        cotan = (u * v).sum(axis=1) / np.linalg.norm(np.cross(u, v), -1)
        W_ij = np.append(W_ij, 0.5 * cotan)
        I = np.append(I, vi2)
        J = np.append(J, vi3)
        W_ij = np.append(W_ij, 0.5 * cotan)
        I = np.append(I, vi3)
        J = np.append(J, vi2)
    L = sp.sparse.csr_matrix((W_ij, (I, J)), shape=(n, n))
    L = L - sp.sparse.spdiags(L * np.ones(n), 0, n, n)
    L = L.tocsr()
    return L