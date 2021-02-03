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
parser.add_argument("-row", "--defo_row", type=str, default="defo_row.npz")
args = parser.parse_args()


template_tooth_dict = {
    "{}{}".format(j, i): pymesh.load_mesh(pjoin(args.template, "{}{}_clean.stl".format(j, i)))
    for i in range(1, 8)
    for j in [1, 2, 3, 4]}

folders = glob.glob(pjoin(args.data, 'scan*'))
for folder in tqdm(folders):
    defo_row = np.load(pjoin(folder, args.defo_row))
    teeth_path = glob.glob(pjoin(folder, "*_clean.stl"))
    alignment_dict = {}
    for tooth_path in teeth_path:
        name = tooth_path.split('/')[-1].split('_')[0]
        if name[1] == '8':
            continue
        tooth = pymesh.load_mesh(tooth_path)

        if name[0] in [1, 2]:  # Upper row
            tooth = align_mesh(tooth,
                               defo_row['R_upper'],
                               defo_row['T_upper'],
                               defo_row['S_upper'])
        else:  # Lower row
            tooth = align_mesh(tooth,
                               defo_row['R_lower'],
                               defo_row['T_lower'],
                               defo_row['S_lower'])

        # PCA alignement

        R_pca, T_pca, S_pca = compute_pca_alignement(
            tooth, template_tooth_dict[name])

        tooth = align_mesh(tooth, R_pca, T_pca, S_pca)

        # ICP alignement

        R_icp, T_icp = igl.iterative_closest_point(
            tooth.vertices, tooth.faces,
            template_tooth_dict[name].vertices, template_tooth_dict[name].faces,
            num_samples=8000, max_iters=100)

        tooth = align_mesh(tooth, R_icp, T_icp, np.ones(3))

        # Final deformation

        R, T, S = merge_rigid_deformations(
            [R_pca, R_icp],
            [T_pca, T_icp],
            [S_pca, np.ones(3)])

        # save
        alignment_dict['R_{}'.format(name)] = R
        alignment_dict['T_{}'.format(name)] = T
        alignment_dict['S_{}'.format(name)] = S

    np.savez(pjoin(folder, "defo_teeth.npz"), **alignment_dict)
