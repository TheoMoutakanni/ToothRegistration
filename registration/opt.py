import numpy as np
import igl
import pymesh
from sklearn.neighbors import KDTree
from scipy.optimize import minimize, fmin_l_bfgs_b


def check_angle(vec1,vec2,thresh_angle):
    """Check if the angle between two vectors is less than a threshold angle
    Params:
        vec1: (Nxd) vector
        vec2: (Nxd) vector
        thresh_angle: threshold angle in degree
    Return:
        small_angle: (N) the boolean vector of angle less than threshold
    """
    if (len(vec1) == 0 or len(vec2) == 0):
        return np.ones((vec2.shape[0],1))

    crossProd = np.cross(vec1,vec2,1)
    dotProd = np.sum((vec1*vec2),1)
    angle = np.arctan2((np.sum((crossProd**2),1))**0.5, dotProd)/np.pi*180

    angle = angle%360

    return angle <= thresh_angle


def cost_fun(A, M, S, landmarks_M, landmarks_S, W_data, W_smooth, W_lm, dist_max, angle_max=60):
    """The cost function to optimize
    Params:
        A: (3x4xN) the transformation quaternions for x,y,z coordinates
        M: (N points mesh) the mesh to deform
        S: (M points mesh) the target mesh
        landmarks_M: (list of size k) the list of landmarks on mesh M aligned to landmarks_S
        landmarks_S: (list of size k) the list of landmarks on mesh S aligned to landmarks_M
        W_data: (float) the weight of the data loss
        W_smooth: (float) the weight of the smooth loss
        W_lm: (float) the weight of the landmarks loss loss
        dist_max: (float) the maximum distance to match two points
        angle_max: (float) the maximum angle between two point's normal to match them in degree
    Return:
        E_total: (float) the final loss
    """
    N = M.num_vertices
    A = A.reshape(3,4,N)
    
    #if AM is None:
    AM = pymesh.form_mesh(np.einsum("ijk,jk->ki",A, M.vertices.T), M.faces)
    #if (dist is None and is_valid_nn is None) or NN_M2S is None:
    tree = KDTree(S.vertices, leaf_size=40)
    dist, NN_M2S = tree.query(AM.vertices,1)
    NN_M2S = NN_M2S.flatten()
    dist = dist.flatten()
    #if is_valid_nn is None:
    AMn = igl.per_vertex_normals(AM.vertices, M.faces)
    Sn = igl.per_vertex_normals(S.vertices, S.faces)
    is_valid_nn = check_angle(AMn, Sn[NN_M2S], angle_max) & (dist <= dist_max)
    #is_valid_nn = np.array([True]*len(dist))

    E_smooth = 0
    nSmooth = 0
    for i in range(N):
        ring = M.get_vertex_adjacent_vertices(i)
        E_smooth += (np.linalg.norm(A[:,:,i:i+1] - A[:,:,ring])**2).sum()
        nSmooth += len(ring)
    E_smooth *= W_smooth/nSmooth

    if W_data > 0 and np.sum(is_valid_nn) > 0:
        E_data = (is_valid_nn/np.sum(is_valid_nn) * np.linalg.norm(AM.vertices - S.vertices[NN_M2S])**2).sum()
    else:
        E_data = 0.

    E_lm = (np.linalg.norm(np.einsum("ijk,jk->ki",A[:,:,landmarks_M], M.vertices[landmarks_M].T) - S.vertices[landmarks_S])**2).sum()
    if (len(landmarks_M) > 0):
        E_lm *= W_lm/len(landmarks_M)

    E_total = E_data + E_smooth + E_lm

    return E_total


def grad_cost_fun(A, M, S, landmarks_M, landmarks_S, W_data, W_smooth, W_lm, dist_max, angle_max=60):
    """The gradient of the cost function to optimize
    Params:
        A: (3x4xN) the transformation quaternions for x,y,z coordinates
        M: (N points mesh) the mesh to deform
        S: (M points mesh) the target mesh
        landmarks_M: (list of size k) the list of landmarks on mesh M aligned to landmarks_S
        landmarks_S: (list of size k) the list of landmarks on mesh S aligned to landmarks_M
        W_data: (float) the weight of the data loss
        W_smooth: (float) the weight of the smooth loss
        W_lm: (float) the weight of the landmarks loss loss
        dist_max: (float) the maximum distance to match two points
        angle_max: (float) the maximum angle between two point's normal to match them in degree
    Return:
        E_total: (3*4*N,1) the gradient of the loss
    """
    N = M.num_vertices
    A = A.reshape(3,4,N)
    
    #if AM is None:
    AM = pymesh.form_mesh(np.einsum("ijk,jk->ki", A, M.vertices.T), M.faces)
    #if (dist is None and is_valid_nn is None) or NN_M2S is None:
    tree = KDTree(S.vertices, leaf_size=40)
    dist, NN_M2S = tree.query(AM.vertices,1)
    NN_M2S = NN_M2S.flatten()
    dist = dist.flatten()
    #if is_valid_nn is None:
    AMn = igl.per_vertex_normals(AM.vertices, M.faces)
    Sn = igl.per_vertex_normals(S.vertices, S.faces)
    is_valid_nn = check_angle(AMn, Sn[NN_M2S], angle_max) & (dist <= dist_max)
    #is_valid_nn = np.array([True]*len(dist))

    G_data = 0
    if W_data > 0 and np.sum(is_valid_nn) > 0:
        G_data = W_data * is_valid_nn/np.sum(is_valid_nn) * 2 * np.einsum("ki,kj->jik", M.vertices, AM.vertices - S.vertices[NN_M2S])
    
    G_smooth = np.zeros((3,4,N))
    nSmooth = 0
    for i in range(N):
        ring = M.get_vertex_adjacent_vertices(i)
        G_smooth[:,:,i] += 2*(A[:,:,i:i+1] - A[:,:,ring]).sum(2)
        nSmooth += len(ring)
    G_smooth *= W_smooth/nSmooth

    G_lm = np.zeros((3,4,N))
    G_lm[:,:,landmarks_M] = 2 * np.einsum("ki,kj->jik", M.vertices[landmarks_M], np.einsum("ijk,jk->ki",A[:,:,landmarks_M], M.vertices[landmarks_M].T) - S.vertices[landmarks_S])
    if (len(landmarks_M) > 0):
        G_lm *= W_lm/len(landmarks_M)

    G = G_lm + G_smooth + G_data
    G = G.reshape([3*4*N,1])
    return G


def get_init_A(N, noise=0.01):
    A = np.array([np.concatenate([np.identity(3), np.zeros((3,1))],1) + noise*(2*np.random.random()-1) for _ in range(N)]).transpose(1,2,0)
    return A


def fit_mesh(M, S, landmarks_M=[], landmarks_S=[], W_data=1, W_smooth=1e+6, W_lm=1e-3, dist_max=0.4, angle_max=60, verbose=1, maxiter=400):
    N = M.num_vertices
    A = get_init_A(N)

    S = pymesh.form_mesh(S.vertices, S.faces)
    M = pymesh.form_mesh(np.concatenate([M.vertices, np.ones((N,1))], 1), M.faces)
    M.enable_connectivity()

    it = 1
    errPrev = np.inf
    err = 1e9
    eps_err = 1e-3

    # First phase with decreasing smoothing and landmarks weights

    fun_args = (M, S, landmarks_M, landmarks_S, W_data, W_smooth, W_lm, dist_max, angle_max)

    while (np.abs(errPrev - err) > eps_err and W_smooth >= W_data*1e3 and it <= maxiter):  
        errPrev = err
        A, err, _ = fmin_l_bfgs_b(cost_fun, A, grad_cost_fun, args=fun_args, maxiter=1)
        
        A = A.reshape(3,4,N)
        
        W_smooth *= 0.75
        W_lm *= 0.75
        fun_args = (M, S, landmarks_M, landmarks_S, W_data, W_smooth, W_lm, dist_max, angle_max)
        
        if verbose:
            AM = pymesh.form_mesh(np.einsum("ijk,jk->ki", A, M.vertices.T), M.faces)
            tree = KDTree(S.vertices, leaf_size=40)
            dist, NN_M2S = tree.query(AM.vertices,1)
            NN_M2S = NN_M2S.flatten()
            dist = dist.flatten()
            AMn = igl.per_vertex_normals(AM.vertices, M.faces)
            Sn = igl.per_vertex_normals(S.vertices, S.faces)
            is_valid_nn = check_angle(AMn, Sn[NN_M2S], angle_max) & (dist <= dist_max)
            
            dist = np.linalg.norm(AM.vertices - S.vertices[NN_M2S])**2
            distAll = (dist * is_valid_nn).sum()
            print('{} ERR: {} %_valid_nn: {} Dist: {}'.format(it, err, np.sum(is_valid_nn) / len(is_valid_nn), distAll))
        it += 1

    # Second phase

    def callbackF(A, *args):
        global it
        print(it, cost_fun(A, *fun_args))
        it += 1

    fun_args = (M, S, landmarks_M, landmarks_S, W_data, W_smooth, W_lm, dist_max, angle_max)
    res = minimize(cost_fun, A, args=fun_args, method='L-BFGS-B', jac=grad_cost_fun, callback=callbackF, options={'maxiter':maxiter-it, 'disp':True})

    final_A = res.x.reshape(3,4,N)

    AM = pymesh.form_mesh(np.einsum("ijk,jk->ki",final_A, M.vertices.T), M.faces)

    return AM, A