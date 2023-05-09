import os
import time
from typing import Literal, Optional

import numpy as np
from rdkit.DataStructs import ConvertToNumpyArray

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from rdkit import Chem
from rdkit.Chem import AllChem

BATCH_SIZE = 1000


def calc_morgan(mol, radius=4, n_bits=256, count=True, use_chirality=True):
    # if bad mol, return array of nan (so it can still fit in the array)
    if mol is None:
        return np.full(n_bits, np.nan)

    if count:
        _fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits, useChirality=use_chirality)
    else:
        _fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=use_chirality)

    fp = np.zeros(n_bits, dtype=np.int8)
    ConvertToNumpyArray(_fp, fp)
    return fp


def get_fps(smis, func: Literal["morgan"]):
    mols = [Chem.MolFromSmiles(smile) for smile in smis]
    mols = [m for m in mols if m is not None]  # can ghost remove but to bad
    if func == "morgan":
        return [calc_morgan(c) for c in mols]
    raise ValueError(f"cannot find func {func}")


class SmilesLoaderNew:
    def __init__(self, file_path, header=True, smi_col="SMILES", name_col=None, delimiter=","):
        self.file_path = file_path
        self.header = header
        self.smi_col = smi_col
        self.delimiter = delimiter
        self.name_col = name_col

    def __iter__(self):
        smiles_col_idx = 0
        name_col_idx = None
        with open(self.file_path, "r") as f:
            for i, line in enumerate(f):
                splits = line.split(self.delimiter)
                splits = [_.strip() for _ in splits]
                if i == 0 and self.header:
                    smiles_col_idx = splits.index(self.smi_col)
                    if self.name_col is not None:
                        name_col_idx = splits.index(self.name_col)
                    continue
                name = None if name_col_idx is None else splits[name_col_idx].strip()
                yield splits[smiles_col_idx].strip(), name

    def batches(self, batch_size: int = 100):
        smiles = []
        names = []
        for i, (smi, name) in self:
            smiles.append(smi)
            names.append(name)
            if (i + 1) % batch_size == 0:
                yield smiles, names
                smiles = []
                names = []
        if len(smiles) != 0:
            yield smiles, names


def _get_output_loc(ref_filename, prefix, suffix, out_dir):
    outfile_name = ref_filename
    outfile_name = outfile_name + "_" + suffix if suffix is not None else outfile_name
    outfile_name = prefix + "_" + outfile_name if prefix is not None else outfile_name
    outfile_name = outfile_name + ".dist"
    out_loc = os.path.join(out_dir, outfile_name)
    return out_loc


def _get_query_header(ii):
    string = ("#" * 60) + "\n"
    string += ("#" * 12) + "  Results for Query Database " + str(ii) + ("#" * 12) + "\n"
    string += ("#" * 60) + "\n"
    return string


def _chemical_distance_brute(ref_descriptor,
                             ref_names,
                             ref_smiles,
                             query_descriptor,
                             query_names,
                             query_smiles):
    query_descriptor = np.vstack(query_descriptor)
    if ref_names is None:
        ref_names = np.array(["NA"] * len(ref_smiles))
    dists = cdist(ref_descriptor, query_descriptor)
    ai = np.expand_dims(np.argmin(dists, axis=0), axis=0)
    names = ref_names[ai.squeeze()]
    d = np.take_along_axis(dists, ai, axis=0)
    res = np.vstack((query_names, query_smiles, d, ai, names)).T
    return res


def _chemical_distance_kdtree(tree,
                              tree_names,
                              tree_smiles,
                              in_descriptor,
                              in_names,
                              in_smiles,
                              k=1):
    descriptor = np.vstack(in_descriptor)
    if tree_names is None:
        tree_names = np.array(["NA"] * len(tree_smiles))
    d, ai = tree.query(descriptor, k=k)
    names = tree_names[ai.squeeze()]
    if k > 1:
        res = np.hstack((in_names.reshape(-1, 1), in_smiles.reshape(-1, 1), d, ai, names))
    else:
        res = np.vstack((in_names, in_smiles, d, ai, names)).T
    return res


def romanesco(
        query_file,
        reference_file,
        query_smi_col="SMILES",
        query_name_col=None,
        query_delimiter=",",
        reference_smi_col="SMILES",
        reference_name_col=None,
        reference_delimiter=",",
        n_neighbors: int = 1,
        algorithm: Literal["kdtree", "brute"] = "kdtree",
        out_dir: Optional[str] = None,
        outfile_prefix: Optional[str] = None,
        outfile_suffix: Optional[str] = None,
        print_res: bool = False,
        return_res: bool = True):
    if out_dir is not None:
        out_loc = _get_output_loc(os.path.basename(reference_file), outfile_prefix, outfile_suffix, out_dir)
    else:
        out_loc = False

    query = [(s, n) for s, n in SmilesLoaderNew(query_file, smi_col=query_smi_col,
                                                name_col=query_name_col, delimiter=query_delimiter)]
    query_smi = [_[0] for _ in query]
    query_name = [_[1] for _ in query]
    query_fps = get_fps(query_smi, func="morgan")

    if algorithm == "kdtree":
        q_tree = KDTree(query_fps)
        header = "\t".join(["Name", "SMILES"] +
                           [f"dist_{_}" for _ in range(n_neighbors)] +
                           [f"ref_idx_{_}" for _ in range(n_neighbors)] +
                           [f"ref_name_{_}" for _ in range(n_neighbors)]) + "\n"
    else:
        header = "\t".join(["Name", "SMILES"] +
                           [f"dist_{_}" for _ in range(1)] +
                           [f"ref_idx_{_}" for _ in range(1)] +
                           [f"ref_name_{_}" for _ in range(1)]) + "\n"

    if out_loc:
        with open(out_loc, "w") as f:
            f.write(header)

    all_results = []  # save results for return_res == True
    t0 = time.time()
    for i, (d_smiles, d_names) in enumerate(SmilesLoaderNew(reference_file,
                                                            smi_col=reference_smi_col,
                                                            name_col=reference_name_col,
                                                            delimiter=reference_delimiter).batches(BATCH_SIZE)):
        d_desc = get_fps(d_smiles, func="morgan")
        if algorithm == "kdtree":
            res = _chemical_distance_kdtree(q_tree, query_name, query_smi, d_desc, d_names, d_smiles, k=n_neighbors)
        else:
            res = _chemical_distance_brute(query_fps, query_name, query_smi, d_desc, d_names, d_smiles)
        res.astype(str)

        # if printing handling printing
        if print_res:
            print(_get_query_header(query_file))
            print("\n" + header)
            for row in res:
                print("\t".join(row))
            print("\n")

        # if outputting to file
        if out_loc:
            with open(out_loc, "a") as f:
                for row in res:
                    f.write("\t".join([str(_) for _ in row]) + "\n")

        # if returning the result
        if return_res:
            all_results.append(res)

    if return_res:
        return np.array(all_results).squeeze()
    else:
        return out_loc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Similarity search")
    parser.add_argument("--reference_file", type=str, required=True,
                        help="Location of SMILES to be used as the reference")
    parser.add_argument("--query_file", type=str, required=True,
                        help="Location of SMILES to be used as the query")
    parser.add_argument("--ref_smi_col", type=str, required=False, default='SMILES',
                        help="Name of SMILES column in reference_file")
    parser.add_argument("--query_smi_col", type=str, required=False, default='SMILES',
                        help="Name of SMILES column in query_file")
    parser.add_argument("--ref_name_col", type=str, required=False, default=None,
                        help="Name of column holding datapoint names in reference_file")
    parser.add_argument("--query_name_col", type=str, required=False, default=None,
                        help="Name of column holding datapoint names in query_file")
    parser.add_argument("--ref_delimiter", type=str, required=False, help="delimiter of reference_file", default='\t')
    parser.add_argument("--query_delimiter", type=str, required=False, help="delimiter of query_file", default='\t')
    parser.add_argument("--n_neighbors", type=int, required=False,
                        help="number of nearest neighbors to return (kdtree only)", default=1)
    parser.add_argument("--algorithm", type=str, required=False,
                        help="which algorithm to use (only for romanesco): ['kdtree', 'brute']", default='kdtree')
    parser.add_argument("--out_dir", type=str, required=False, help="directory location for output", default=None)
    parser.add_argument("--prefix", type=str, required=False, help="prefix of output files", default=None)
    parser.add_argument("--suffix", type=str, required=False, help="suffix of output files", default=None)
    parser.add_argument("--print_res", action="store_true", help="print result to console")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.getcwd()

    os.makedirs(args.out_dir, exist_ok=True)

    out_loc = romanesco(query_file=args.query_file, reference_file=args.reference_file,
                        query_smi_col=args.query_smi_col, reference_smi_col=args.ref_smi_col,
                        query_name_col=args.query_name_col, reference_name_col=args.ref_name_col,
                        query_delimiter=args.query_delimiter, reference_delimiter=args.ref_delimiter,
                        n_neighbors=args.n_neighbors, algorithm=args.algorithm, out_dir=args.out_dir,
                        outfile_prefix=args.prefix, outfile_suffix=args.suffix, print_res=args.print_res,
                        return_res=False)
