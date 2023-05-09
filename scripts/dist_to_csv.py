import sys

file = sys.argv[1]

lines = []

with open(file, "r") as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        lines.append([_.strip() for _ in line.split("\t")])

with open(file.replace(".dist", ".csv"), "w") as f:
    f.write("SMILES,Name\n")
    for line in lines:
        f.write(f"{line[1]},{line[0]}\n")
