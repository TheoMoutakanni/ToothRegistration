import argparse
from os.path import join as pjoin
import glob
from tqdm import tqdm

import igl
import pymesh
import numpy as np

from registration.utils import (compute_pca_alignement, merge_rigid_deformations,
                                align_mesh)

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="Data folder path")
parser.add_argument("template", type=str, help="Template folder path")
# parser.add_argument("-d", "--detail", type=str, default=0.2)
args = parser.parse_args()


template_upper_row = pymesh.merge_meshes(
    [pymesh.load_mesh(pjoin(args.template, "{}{}_clean.stl".format(j, i)))
     for i in range(1, 8)
     for j in [1, 2]])

template_lower_row = pymesh.merge_meshes(
    [pymesh.load_mesh(pjoin(args.template, "{}{}_clean.stl".format(j, i)))
     for i in range(1, 8)
     for j in [1, 2]])

folders = glob.glob(pjoin(args.data, 'scan*'))
for folder in tqdm(folders):

    upper_row = pymesh.merge_meshes(
        [pymesh.load_mesh(pjoin(folder, "{}{}_clean.stl".format(j, i)))
         for i in range(1, 8)
         for j in [1, 2]])
    lower_row = pymesh.merge_meshes(
        [pymesh.load_mesh(pjoin(folder, "{}{}_clean.stl".format(j, i)))
         for i in range(1, 8)
         for j in [1, 2]])

    # PCA alignement

    R_upper_pca, T_upper_pca, S_upper_pca = compute_pca_alignement(
        upper_row, template_upper_row)

    upper_row = align_mesh(upper_row, R_upper_pca, T_upper_pca, S_upper_pca)

    R_lower_pca, T_lower_pca, S_lower_pca = compute_pca_alignement(
        lower_row, template_lower_row)

    lower_row = align_mesh(lower_row, R_lower_pca, T_lower_pca, S_lower_pca)

    # ICP alignement

    R_upper_icp, T_upper_icp = igl.iterative_closest_point(
        upper_row.vertices, upper_row.faces,
        template_upper_row.vertices, template_upper_row.faces,
        num_samples=8000, max_iters=100)

    upper_row = align_mesh(upper_row, R_upper_icp, T_upper_icp, np.ones(3))

    R_lower_icp, T_lower_icp = igl.iterative_closest_point(
        lower_row.vertices, lower_row.faces,
        template_lower_row.vertices, template_lower_row.faces,
        num_samples=8000, max_iters=100)

    lower_row = align_mesh(lower_row, R_lower_icp, T_lower_icp, np.ones(3))

    R_upper, T_upper, S_upper = merge_rigid_deformations(
        [R_upper_pca, R_upper_icp],
        [T_upper_pca, T_upper_icp],
        [S_upper_pca, np.ones(3)])

    R_lower, T_lower, S_lower = merge_rigid_deformations(
        [R_lower_pca, R_lower_icp],
        [T_lower_pca, T_lower_icp],
        [S_lower_pca, np.ones(3)])


    # save
    np.savez(pjoin(folder, "defo_row.npz"),
        R_upper=R_upper, T_upper=T_upper, S_upper=S_upper,
        R_lower=R_lower, T_lower=T_lower, S_lower=S_lower)
