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
from utils import align_tooth

def main():
    args = sys.argv[1:]

    foldername = str(args[0])
    foldername2 = str(args[1])
    PER = str(args[2])
    outfolder = str(args[3])

    Rs = np.load(foldername+"/"+PER+'_R_row.npz')
    Ts = np.load(foldername+"/"+PER+'_T_row.npz')
    S_pca = np.loadtxt(foldername + "/" + PER + '_S_row_pca.txt', delimiter=',', dtype=np.float32)



    # row1
    for i in range(1, 8):
        for j in [1,2]:
            path = foldername2 + "/Segmentation_" + PER + "_{}{}.stl".format(j, i)
            toothmesh = pymesh.load_mesh(path)
            toothmesh.enable_connectivity()

            toothmesh = fix_mesh(toothmesh)

            final_aligned_toothmesh = align_tooth(toothmesh, Rs, Ts, S_pca)
            pymesh.save_mesh_raw(outfolder + '/' + PER + "_alignedtooth_" + str(j)+str(i) + '.stl', final_aligned_toothmesh.vertices, final_aligned_toothmesh.faces)


if __name__ == "__main__":
    main()
