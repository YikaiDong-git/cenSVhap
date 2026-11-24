"""Assign each SV's carriers to tree branches (haplogroup resolution).

For one SV with carrier indicator s over H haplotypes and branch bipartitions b_1..b_T:

  1. find the branch with the largest R^2(s, b_t) (squared Pearson correlation);
  2. if R^2 >= r2_threshold and the carriers shared with that branch number at least
     `min_component` (first pass) or `min_carriers` (later passes), record those shared carriers
     as a monophyletic component, remove the branch's haplotypes from s, and repeat;
  3. otherwise stop. Carriers left after at least one component form a residual component;
     an SV with no component is kept unresolved with its original carriers.

Every resolved SV also keeps its original (pre-resolution) record.

Output categories (column `sv_category`):
  monophyletic         one component, mapped to branch B<i>   id <sv>_B<i>
  residual             carriers left after the components      id <sv>_residual
  original_decomposed  the SV before resolution                id <sv>_original
  non_decomposed       unresolved SV, original carriers        id <sv>
"""
import numpy as np


def branch_r2(x, y):
    """Squared Pearson correlation of two indicator vectors; 0 when either is constant."""
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def resolve_svs(presence, variants, branches, r2_threshold=0.3, min_component=10, min_carriers=5):
    """Resolve every SV column of `presence` (haplotypes x SVs) against `branches`.

    Returns (steps, component_presence, component_variants, counts):
      steps               list of dicts, one per search step (sv_id, iteration, type, branch_id,
                          r2, n_carriers, carrier_indices)
      component_presence  int8 array, haplotypes x output records
      component_variants  list of dicts, one per output record (input fields + original_sv_id,
                          mapped_haplogroup, r2_score, sv_category; ac = haplotype carrier count)
      counts              dict of record counts per category
    """
    if not branches:
        raise ValueError("no branch passes the tip-count limits; nothing to resolve against")
    ct = np.column_stack([b['vector'] for b in branches])
    ids = [b['branch_id'] for b in branches]

    steps, vectors, records = [], [], []
    counts = dict(original_svs=presence.shape[1], decomposed_svs=0, monophyletic_svs=0,
                  non_decomposed_svs=0, residual_svs=0, original_decomposed_svs=0)

    for j in range(presence.shape[1]):
        base = variants[j]
        sv_id = base.get('id', f"SV_{j}")
        original = presence[:, j].copy()
        sv = original.copy()
        iteration = 0
        resolved = False

        def emit(vec, category, r2, ac, new_id=None, branch=None):
            info = base.copy()
            if new_id is not None:
                info['id'] = new_id
            info.update({'original_sv_id': sv_id, 'mapped_haplogroup': branch, 'r2_score': r2,
                         'sv_category': category, 'ac': ac})
            vectors.append(vec)
            records.append(info)

        def stop(r2):
            # Close the SV: residual if a component was found, otherwise unresolved.
            if resolved:
                emit(sv.copy(), 'residual', r2, int(sv.sum()), f"{sv_id}_residual", 'residual')
                counts['residual_svs'] += 1
            else:
                emit(original.copy(), 'non_decomposed', r2, int(original.sum()))
                counts['non_decomposed_svs'] += 1

        def step(kind, branch, r2, carriers):
            steps.append({'sv_id': sv_id, 'iteration': iteration, 'type': kind, 'branch_id': branch,
                          'r2': r2, 'n_carriers': int(carriers.sum()),
                          'carrier_indices': np.where(carriers > 0)[0].tolist()})

        while True:
            ac = int(sv.sum())
            if ac < min_carriers:
                if ac > 0:
                    step('residual_small', None, None, sv)
                    stop(None)
                elif not resolved:
                    stop(None)
                break

            r2 = np.array([branch_r2(sv, ct[:, t]) for t in range(ct.shape[1])])
            best = int(np.argmax(r2))
            best_r2 = float(r2[best])
            if best_r2 < r2_threshold:
                step('paraphyletic', None, best_r2, sv)
                stop(best_r2)
                break

            mono = sv & ct[:, best]
            n_mono = int(mono.sum())
            if iteration == 0 and n_mono < min_component:
                stop(best_r2)
                break
            if n_mono < min_carriers:
                step('paraphyletic', ids[best], best_r2, sv)
                stop(best_r2)
                break

            step('monophyletic', ids[best], best_r2, mono)
            emit(mono, 'monophyletic', best_r2, n_mono, f"{sv_id}_{ids[best]}", ids[best])
            counts['monophyletic_svs'] += 1
            if not resolved:
                resolved = True
                counts['decomposed_svs'] += 1
                emit(original.copy(), 'original_decomposed', None, int(original.sum()),
                     f"{sv_id}_original")
                counts['original_decomposed_svs'] += 1

            sv = sv & (~ct[:, best].astype(bool)).astype(np.int8)
            iteration += 1

    matrix = np.column_stack(vectors) if vectors else np.empty((presence.shape[0], 0), dtype=np.int8)
    return steps, matrix, records, counts
