"""Haplotype distance, neighbor-joining tree, and the bipartitions defined by its branches."""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.tree import nj

DISTANCES = ('cosine', 'jaccard')


def haplotype_distances(presence, distance='cosine'):
    """Pairwise distance matrix between haplotype SV profiles (rows of `presence`)."""
    if distance == 'cosine':
        d = pdist(presence, metric='cosine')
    elif distance == 'jaccard':
        d = pdist(presence.astype(bool), metric='jaccard')
    else:
        raise ValueError(f"distance must be one of {DISTANCES}")
    return squareform(np.nan_to_num(d, nan=1.0))


def build_tree(presence, haplotypes, distance='cosine'):
    """Neighbor-joining tree (scikit-bio) over haplotype SV profiles."""
    return nj(DistanceMatrix(haplotype_distances(presence, distance), ids=haplotypes))


def branch_partitions(tree, haplotypes, min_tips=5):
    """Bipartitions from the internal branches of the midpoint-rooted tree.

    A branch is kept when the clade below it holds between `min_tips` and H - `min_tips`
    haplotypes. Branch ids are B<i>, i being the node's position in a pre-order traversal of the
    rooted tree, so they can be mapped back onto the written tree.
    Returns a list of dicts: branch_id, length, n_tips, vector (int8 indicator over haplotypes).
    """
    rooted = tree.root_at_midpoint()
    names = set(haplotypes)
    n = len(haplotypes)
    out = []
    for i, node in enumerate(rooted.traverse(include_self=False)):
        if node.is_tip():
            continue
        tips = node.subset() & names
        if min_tips <= len(tips) <= n - min_tips:
            out.append({
                'branch_id': f"B{i}",
                'length': node.length or 0.0,
                'n_tips': len(tips),
                'vector': np.array([1 if h in tips else 0 for h in haplotypes], dtype=np.int8),
            })
    return out
