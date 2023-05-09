import os
from typing import Optional, List


def _read_dist_file(file_loc):
    with open(file_loc, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            yield [_.strip() for _ in line.split("\t")]


def _collect_dist_files(files: List, max_hits: int = 100000, max_hits_per_query: Optional[int] = None,
                        max_dist: float = 10.0, out_dir: Optional[str] = None):
    hits = {}
    for file in files:
        for row in _read_dist_file(file):
            if max_hits_per_query is None:
                hits.setdefault(-1, []).append((row[0], row[1], float(row[2]), row[3], row[4]))
            else:
                hits.setdefault(row[4], []).append((row[0], row[1], float(row[2]), row[3], row[4]))
    hits = {key: sorted(val, key=lambda x: x[2]) for key, val in hits.items()}
    hits = {key: val[:max_hits_per_query] if key != -1 else val[:max_hits] for key, val in hits.items()}
    hits = [(key, _val) for key, val in hits.items() for _val in val]
    hits = sorted(hits, key=lambda x: x[1][2])[:max_hits]
    hits = [(_1, _2) for _1, _2 in hits if _2[2] <= max_dist]

    if out_dir:
        outfile = files[0].replace(".dist", "_collected.dist")
        with open(os.path.join(out_dir, outfile), "w") as f:
            header = "\t".join(["Name", "SMILES"] +
                               [f"dist_{_}" for _ in range(1)] +
                               [f"ref_idx_{_}" for _ in range(1)] +
                               [f"ref_name_{_}" for _ in range(1)]) + "\n"
            f.write(header)
            for hit in hits:
                f.write(
                    "\t".join([str(hit[1][0]), str(hit[1][1]), str(hit[1][2]), str(hit[1][3]), str(hit[1][4])]))
                f.write("\n")
    else:
        return {key: val for key, val in hits}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Similarity search")
    parser.add_argument("--dist_files", required=True, nargs="+",
                        help='dist file locs for collections')
    parser.add_argument('--num_total', type=int, default=100000, required=False,
                        help='number of compounds to pick')
    parser.add_argument('--max_hits_per_query', type=int, default=1000, required=False,
                        help='number of hits for each reference')
    parser.add_argument('--max_dist', type=float, default=10.0, required=False,
                        help='maximum distance a hit can have')
    parser.add_argument('--outdir', type=str, required=True, help="output location")
    args = parser.parse_args()

    print(args.outdir)

    _collect_dist_files(files=args.dist_files, max_hits=args.num_total, max_hits_per_query=args.max_hits_per_query,
                        max_dist=args.max_dist, out_dir=args.outdir)
