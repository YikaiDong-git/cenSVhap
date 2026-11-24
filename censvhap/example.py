"""Synthetic cohort with known answers, used by the example and the tests.

80 haplotypes (40 individuals) fall into four clades with topology ((A,B),(C,D)), 20 haplotypes
each. Background SVs define the topology: 30 shared by A+B, 30 by C+D, 30 private to each clade,
and 40 at 15% frequency across all haplotypes. Five test SVs are planted:

  TEST_single      all of A                          -> one component (A)
  TEST_two_origin  all of A and all of C             -> two components (A, C); A+C is not a branch
  TEST_scattered   3 haplotypes from each clade      -> unresolved (no branch reaches R^2 0.3)
  TEST_rare        8 haplotypes of A                 -> unresolved (first component < 10 carriers)
  TEST_residual    all of A + 3 of C + 3 of D        -> component A, then a 6-carrier residual

Usage: python -m censvhap.example OUT.vcf
"""
import sys

import numpy as np

CLADES = 'ABCD'
PER_CLADE = 20


def make_cohort(seed=1):
    """Return (presence haplotypes x SVs, sv_ids, haplotype names, clade label per haplotype)."""
    rng = np.random.default_rng(seed)
    n = PER_CLADE * len(CLADES)
    clade = np.repeat(np.arange(len(CLADES)), PER_CLADE)
    haplotypes = [f"IND{h // 2:02d}_hap{h % 2 + 1}" for h in range(n)]
    cols, ids = [], []

    def add(vec, name):
        cols.append(vec.astype(np.int8))
        ids.append(name)

    for k in range(30):
        add(np.isin(clade, (0, 1)), f"bg_AB_{k}")
        add(np.isin(clade, (2, 3)), f"bg_CD_{k}")
    for c in range(len(CLADES)):
        for k in range(30):
            add(clade == c, f"bg_{CLADES[c]}_{k}")
    for k in range(40):
        add(rng.random(n) < 0.15, f"bg_noise_{k}")

    def pick(c, m):
        return rng.choice(np.where(clade == c)[0], m, replace=False)

    add(clade == 0, "TEST_single")
    add(np.isin(clade, (0, 2)), "TEST_two_origin")
    v = np.zeros(n, bool)
    for c in range(len(CLADES)):
        v[pick(c, 3)] = True
    add(v, "TEST_scattered")
    v = np.zeros(n, bool)
    v[pick(0, 8)] = True
    add(v, "TEST_rare")
    v = clade == 0
    v[pick(2, 3)] = True
    v[pick(3, 3)] = True
    add(v, "TEST_residual")
    return np.column_stack(cols), ids, haplotypes, clade


def write_vcf(path, seed=1):
    presence, ids, haplotypes, _ = make_cohort(seed)
    with open(path, 'w') as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chrS,length=10000000>\n")
        fh.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">\n')
        fh.write('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="SV length">\n')
        fh.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position">\n')
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(haplotypes) + "\n")
        for j, sv in enumerate(ids):
            pos = 1000 + 1000 * j
            gts = "\t".join('1' if x else '0' for x in presence[:, j])
            fh.write(f"chrS\t{pos}\t{sv}\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;SVLEN=-500;END={pos + 500}\tGT\t{gts}\n")


if __name__ == "__main__":
    write_vcf(sys.argv[1])
