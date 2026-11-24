"""Read a multi-haplotype SV VCF into a haplotype-by-SV presence matrix."""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import cyvcf2

HAPLOTYPE_NAME = re.compile(r'^([A-Za-z0-9]+)_(hap[12]|mat|pat)')


def parse_haplotype(name):
    """Split a haplotype column name (e.g. HG002_hap1, HG002_mat) into (individual, hap1|hap2)."""
    m = HAPLOTYPE_NAME.match(name)
    if m is None:
        return None
    return m.group(1), ('hap1' if m.group(2) in ('hap1', 'mat') else 'hap2')


def read_presence(vcf_path, chrom, min_carriers=2, max_freq=0.99, individuals=None):
    """Return (presence, variants, haplotypes) for one chromosome.

    Each VCF sample column is one haplotype assembly. A haplotype carries an SV when its genotype
    is heterozygous or homozygous alternate. SVs carried by fewer than `min_carriers` haplotypes or
    by more than int(H * max_freq) are skipped. Records at identical (pos, end, svtype, svlen) are
    collapsed to the one with the most carriers (first listed on ties).

    presence:   int8 array, haplotypes x SVs
    variants:   list of dicts, one per SV column
    haplotypes: list of haplotype names (rows of presence)
    """
    reader = cyvcf2.VCF(str(vcf_path), gts012=True)
    haplotypes = list(reader.samples)
    if individuals is not None:
        keep = [i for i, s in enumerate(haplotypes)
                if parse_haplotype(s) and parse_haplotype(s)[0] in individuals]
        haplotypes = [haplotypes[i] for i in keep]
        cols = np.array(keep, dtype=int)
    else:
        cols = np.arange(len(haplotypes))
    if len(haplotypes) == 0:
        raise ValueError(f"no haplotype columns of {vcf_path} match the requested individuals")

    n = len(haplotypes)
    max_count = int(n * max_freq)
    columns, variants = [], []
    indexed = any(Path(str(vcf_path) + s).exists() for s in ('.tbi', '.csi'))
    records = reader(chrom) if indexed else (v for v in reader if v.CHROM == chrom)
    for v in records:
        present = np.isin(v.gt_types[cols], (1, 2))
        ac = int(present.sum())
        if ac < min_carriers or ac > max_count:
            continue
        columns.append(present.astype(np.int8))
        variants.append({
            'chrom': v.CHROM, 'pos': v.POS,
            'end': v.INFO.get('END', v.POS),
            'id': v.ID or f"{v.CHROM}_{v.POS}",
            'svtype': v.INFO.get('SVTYPE', 'UNK'),
            'svlen': abs(v.INFO.get('SVLEN', 0) or 0),
            'ref': v.REF,
            'alt': ",".join(v.ALT) if v.ALT else '.',
            'ac': ac, 'an': n,
            'af': round(ac / n, 6),
            'maf': round(min(ac / n, 1.0 - ac / n), 6),
            'n_miss': 0,
        })
    reader.close()
    if not columns:
        return None, [], haplotypes

    columns, variants = collapse_duplicates(columns, variants)
    return np.vstack(columns).T, variants, haplotypes


def collapse_duplicates(columns, variants):
    """Keep one record per (pos, end, svtype, svlen): the one with the highest carrier count."""
    groups = defaultdict(list)
    for i, v in enumerate(variants):
        groups[(v['pos'], v['end'], v['svtype'], v['svlen'])].append(i)
    keep = sorted(max(idx, key=lambda i: variants[i]['ac']) for idx in groups.values())
    return [columns[i] for i in keep], [variants[i] for i in keep]


def to_diploid(presence, haplotypes):
    """Sum the two haplotypes of each individual: individuals x SVs dosage (0, 1, 2)."""
    rows = defaultdict(list)
    for i, h in enumerate(haplotypes):
        p = parse_haplotype(h)
        if p:
            rows[p[0]].append(i)
    individuals = sorted(rows)
    dosage = np.zeros((len(individuals), presence.shape[1]), dtype=np.int8)
    for k, ind in enumerate(individuals):
        dosage[k] = presence[rows[ind]].sum(axis=0)
    return dosage, individuals
