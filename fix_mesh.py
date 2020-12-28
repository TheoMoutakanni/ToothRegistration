import argparse
from os.path import join as pjoin
import glob
from tqdm import tqdm

import pymesh
import numpy as np

from registration.utils import fix_mesh


parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="Data folder path")
parser.add_argument(
    "-o", "--output", default="{name}_clean.stl", type=str, help="output name")
parser.add_argument("-d", "--detail", type=str, default=0.2)
args = parser.parse_args()

try:
    args.detail = float(args.detail)
except ValueError:
    pass


folders = glob.glob(pjoin(args.data, 'scan*'))
for folder in tqdm(folders):
    teeth = glob.glob(pjoin(folder, "*.stl"))
    for tooth in teeth:
        name = tooth.split('/')[-1].split('.')[0]
        M = pymesh.load_mesh(tooth)
        M_clean = fix_mesh(M, detail=args.detail)
        pymesh.save_mesh(pjoin(folder, args.output.format(name=name)),
                         M, ascii=True)
