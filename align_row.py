import sys
import igl
import pymesh
import scipy as sp
import numpy as np
from meshplot import plot, subplot, interact
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
import os
from os.path import join as pjoin
from fix_mesh import fix_mesh


def main():
    args = sys.argv[1:]

    foldername = str(args[0])
    PER = str(args[1])
    outfolder = str(args[2])

    row1 = pymesh.merge_meshes(
        [pymesh.load_mesh(foldername + "/Segmentation_" + PER + "_{}{}.stl".format(j, i))
         for i in range(1, 8)
         for j in [1, 2]])
    row2 = pymesh.merge_meshes(
        [pymesh.load_mesh(foldername + "/Segmentation_" + PER + "_{}{}.stl".format(j, i))
         for i in range(1, 8)
         for j in [3, 4]])

    row1 = fix_mesh(row1)
    row2 = fix_mesh(row2)

    pca = PCA(3)
    pca.fit(row1.vertices)

    pca2 = PCA(3)
    pca2.fit(row2.vertices)

    # Translation
    T_row_pca = pca2.mean_ - pca.mean_

    norm_axis1 = pca.components_ / np.linalg.norm(pca.components_, axis=1)
    norm_axis2 = pca2.components_ / np.linalg.norm(pca2.components_, axis=1)

    # Rotation as in https://math.stackexchange.com/questions/1125203/finding-rotation-axis-and-angle-to-align-two-3d-vector-bases
    R_row_pca = np.einsum("ij,ik->ijk", norm_axis2, norm_axis1).sum(0)
    # scale
    S_row_pca = np.sqrt(pca2.singular_values_ / pca.singular_values_)

    aligned_row1 = pymesh.form_mesh((row1.vertices - pca.mean_) @ R_row_pca * S_row_pca + pca.mean_ + T_row_pca,
                                    row1.faces)


    R, T = igl.iterative_closest_point(aligned_row1.vertices, aligned_row1.faces, row2.vertices, row2.faces,
                                       num_samples=8000, max_iters=100)

    final_aligned_row1 = pymesh.form_mesh(aligned_row1.vertices @ R + T, row1.faces)








    # save
    #np.savetxt(outfolder + '/' + PER + '_pca_mean.txt', pca.mean_, delimiter=',')
    np.savez(outfolder +'/'+ PER + '_T_row.npz', pca.mean_, T_row_pca, T)
    np.savez(outfolder + '/' + PER + '_R_row.npz', R_row_pca, R)
    #np.savetxt(outfolder +'/'+ PER + '_T_row.npy', [pca.mean_, T_row_pca, T], delimiter=',')
    #np.savetxt(outfolder + '/' + PER + '_R_row.npy', [R_row_pca, R], delimiter=',')
    np.savetxt(outfolder + '/' + PER + '_S_row_pca.txt', S_row_pca, delimiter=',')



    #np.savetxt(outfolder +'/'+ PER + '_T.txt', T, delimiter=',')
    #np.savetxt(outfolder +'/'+ PER + '_R.txt', R, delimiter=',')

    pymesh.save_mesh_raw(outfolder +'/'+ PER +'_'+ "final_aligned_row1.stl", final_aligned_row1.vertices, final_aligned_row1.faces)
    pymesh.save_mesh_raw(outfolder + '/' + PER +'_'+ "row1.stl", row1.vertices, row1.faces)
    pymesh.save_mesh_raw(outfolder +'/'+ PER +'_'+ "row2.stl", row2.vertices, row2.faces)

    print('end')




if __name__ == "__main__":
    main()











