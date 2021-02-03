import argparse
from os.path import join as pjoin
import glob
from tqdm import tqdm

import igl
import pymesh
import numpy as np

from registration.utils import align_mesh
from registration.opt import fit_mesh

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="Data folder path")
parser.add_argument("template", type=str, help="Template folder path")
parser.add_argument("-o", "--output", type=str, default="{}_fit.stl", help="Name of the output mesh")
parser.add_argument("-a", "--A", type=str, default="{}_A.npy", help="If set, save the deformation matrix")
parser.add_argument("-row", "--defo_row", type=str, default="defo_row.npz")
parser.add_argument("-tooth", "--defo_tooth", type=str, default="defo_teeth.npz")

parser.add_argument("--W_data", type=float, default=1.)
parser.add_argument("--W_smooth", type=float, default=1e+6)
parser.add_argument("--W_lm", type=float, default=1e-3)
parser.add_argument("--dist_max", type=float, default=0.4)
parser.add_argument("--angle_max", type=float, default=60.)
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--maxiter", type=int, default=400)

args = parser.parse_args()


template_tooth_dict = {
    "{}{}".format(j, i): pymesh.load_mesh(pjoin(args.template, "{}{}_clean.stl".format(j, i)))
    for i in range(1, 8)
    for j in [1, 2, 3, 4]}

folders = glob.glob(pjoin(args.data, 'scan*'))
for folder in tqdm(folders):
    defo_row = np.load(pjoin(folder, args.defo_row))
    defo_tooth = np.load(pjoin(folder, args.defo_tooth))
    teeth_path = glob.glob(pjoin(folder, "*_clean.stl"))
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
        tooth = align_mesh(tooth,
                           defo_tooth['R_{}'.format(name)],
                           defo_tooth['T_{}'.format(name)],
                           defo_tooth['S_{}'.format(name)])

        # PCA alignement

        AM, A = fit_mesh(template_tooth_dict[name], tooth,
                        W_data=args.W_data, W_smooth=args.W_smooth,
                        W_lm=args.W_lm, dist_max=args.dist_max,
                        angle_max=args.angle_max, verbose=args.verbose,
                        maxiter=args.maxiter)

        pymesh.save_mesh(pjoin(folder, args.output.format(name)),
                         AM, ascii=True)
        if args.A is not None:
            np.save(pjoin(folder, args.A.format(name)), A)
