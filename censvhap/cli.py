"""censvhap: per-chromosome haplogroup resolution of centromeric SVs."""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .io import read_presence, to_diploid
from .tree import DISTANCES, build_tree, branch_partitions
from .resolve import resolve_svs

log = logging.getLogger("censvhap")


def run_chromosome(vcf, chrom, outdir, distance, min_carriers_input, max_freq, min_tips,
                   r2_threshold, min_component, min_carriers, individuals=None):
    presence, variants, haplotypes = read_presence(vcf, chrom, min_carriers_input, max_freq,
                                                   individuals)
    if presence is None:
        log.warning("%s: no SVs pass the input filters; skipped", chrom)
        return None
    # Haplotypes carrying none of the retained SVs have no defined distance; leave them out.
    keep = presence.sum(axis=1) > 0
    presence = presence[keep]
    haplotypes = [h for h, k in zip(haplotypes, keep) if k]
    log.info("%s: %d SVs x %d haplotypes", chrom, presence.shape[1], len(haplotypes))

    tree = build_tree(presence, haplotypes, distance)
    branches = branch_partitions(tree, haplotypes, min_tips)
    steps, comp, records, counts = resolve_svs(presence, variants, branches, r2_threshold,
                                               min_component, min_carriers)
    log.info("%s: %s", chrom, counts)
    write_outputs(Path(outdir) / chrom, chrom, tree, haplotypes, branches, steps, comp, records)
    return counts


def write_outputs(out, chrom, tree, haplotypes, branches, steps, comp, records):
    out.mkdir(parents=True, exist_ok=True)

    # Component genotypes per individual (sum of its two haplotypes).
    dosage, individuals = to_diploid(comp, haplotypes)
    an = 2 * len(individuals)
    for i, r in enumerate(records):
        ac = int(dosage[:, i].sum())
        af = ac / an if an else 0
        r.update(ac=ac, an=an, af=round(af, 6), maf=round(min(af, 1.0 - af), 6))
    np.savez_compressed(out / f"{chrom}_sv_genotypes.npz", genotypes=dosage.T)
    (out / f"{chrom}_sv_samples.txt").write_text('\n'.join(individuals) + '\n')
    pd.DataFrame(records).to_csv(out / f"{chrom}_sv_variants.tsv", sep='\t', index=False)

    df = pd.DataFrame(steps)
    df['carrier_indices'] = df['carrier_indices'].apply(lambda x: ','.join(map(str, x)) if x else '')
    df.to_csv(out / f"{chrom}_decomposition.tsv", sep='\t', index=False)

    pd.DataFrame([{k: b[k] for k in ('branch_id', 'length', 'n_tips')} for b in branches]).to_csv(
        out / f"{chrom}_branches.tsv", sep='\t', index=False)

    # Haplogroups: branches supported by more than ten monophyletic components.
    support = df.loc[df['type'] == 'monophyletic', 'branch_id'].value_counts()
    rows = []
    for b in branches:
        n = int(support.get(b['branch_id'], 0))
        if n > 10:
            carriers = [h for h, x in zip(haplotypes, b['vector']) if x == 1]
            rows.append({'haplogroup_id': b['branch_id'], 'monophyletic_svs': n,
                         'n_carriers': len(carriers), 'carriers': ",".join(carriers)})
    cols = ['haplogroup_id', 'monophyletic_svs', 'n_carriers', 'carriers']
    hg = pd.DataFrame(rows, columns=cols)
    if rows:
        hg = hg.sort_values('monophyletic_svs', ascending=False)
    hg.to_csv(out / f"{chrom}_validated_haplogroups.tsv", sep='\t', index=False)

    # Trees: the NJ tree as built, and the midpoint-rooted tree with per-branch support.
    tree.write(str(out / f"{chrom}_nj.nwk"), format='newick')
    support = support.to_dict()
    rooted = tree.root_at_midpoint()
    for i, node in enumerate(rooted.traverse(include_self=False)):
        n = support.get(f"B{i}", 0)
        if hasattr(node, 'support'):
            node.support = n
        else:
            node.name = f"{node.name}:{n}" if node.name else str(n)
    rooted.write(str(out / f"{chrom}_nj_annotated.nwk"), format='newick')


def main(argv=None):
    p = argparse.ArgumentParser(prog="censvhap", description=__doc__)
    p.add_argument('--vcf', required=True, help="multi-sample SV VCF, one column per haplotype "
                   "(names like HG002_hap1 / HG002_hap2 or _mat / _pat)")
    p.add_argument('--chrom', required=True, nargs='+', help="chromosome(s), e.g. chr17")
    p.add_argument('--out', required=True, help="output directory; one subfolder per chromosome")
    p.add_argument('--distance', default='cosine', choices=DISTANCES)
    p.add_argument('--individuals', help="optional file of individual IDs to keep, one per line")
    p.add_argument('--min-carriers-input', type=int, default=2,
                   help="skip SVs carried by fewer haplotypes (default 2)")
    p.add_argument('--max-freq', type=float, default=0.99,
                   help="skip SVs carried by more than this fraction of haplotypes (default 0.99)")
    p.add_argument('--min-tips', type=int, default=5,
                   help="test only branches with at least this many haplotypes on each side (default 5)")
    p.add_argument('--r2-threshold', type=float, default=0.3)
    p.add_argument('--min-component', type=int, default=10,
                   help="carriers required for the first component (default 10)")
    p.add_argument('--min-carriers', type=int, default=5,
                   help="carriers required for later components (default 5)")
    p.add_argument('--version', action='version', version=__version__)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    individuals = None
    if a.individuals:
        individuals = {l.strip() for l in open(a.individuals) if l.strip() and not l.startswith('#')}
    for chrom in a.chrom:
        run_chromosome(a.vcf, chrom, a.out, a.distance, a.min_carriers_input, a.max_freq,
                       a.min_tips, a.r2_threshold, a.min_component, a.min_carriers, individuals)


if __name__ == "__main__":
    main()
