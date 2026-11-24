import numpy as np
import pandas as pd
import pytest

from censvhap.cli import main
from censvhap.example import make_cohort, write_vcf
from censvhap.io import collapse_duplicates, read_presence, to_diploid
from censvhap.resolve import branch_r2, resolve_svs
from censvhap.tree import branch_partitions, build_tree


@pytest.fixture(scope="module")
def resolved():
    presence, ids, haplotypes, clade = make_cohort()
    variants = [{'id': i} for i in ids]
    tree = build_tree(presence, haplotypes, 'cosine')
    branches = branch_partitions(tree, haplotypes)
    steps, comp, records, counts = resolve_svs(presence, variants, branches)
    df = pd.DataFrame(records)
    df['carriers'] = [frozenset(np.where(comp[:, k])[0]) for k in range(comp.shape[1])]
    return df, clade


def components(df, sv):
    return df[(df.original_sv_id == sv) & (df.sv_category == 'monophyletic')]


def clade_set(clade, *cs):
    return frozenset(np.where(np.isin(clade, cs))[0])


def test_single_origin(resolved):
    df, clade = resolved
    c = components(df, 'TEST_single')
    assert len(c) == 1 and c.iloc[0].carriers == clade_set(clade, 0)
    assert (df.original_sv_id == 'TEST_single').sum() == 2      # component + original record


def test_two_origins_split(resolved):
    df, clade = resolved
    c = components(df, 'TEST_two_origin')
    assert sorted(len(x) for x in c.carriers) == [20, 20]
    assert set(c.carriers) == {clade_set(clade, 0), clade_set(clade, 2)}
    assert not (df[df.original_sv_id == 'TEST_two_origin'].sv_category == 'residual').any()


@pytest.mark.parametrize("sv", ["TEST_scattered", "TEST_rare"])
def test_unresolved(resolved, sv):
    df, _ = resolved
    rows = df[df.original_sv_id == sv]
    assert list(rows.sv_category) == ['non_decomposed']
    assert rows.iloc[0].id == sv


def test_residual(resolved):
    df, clade = resolved
    rows = df[df.original_sv_id == 'TEST_residual']
    assert components(df, 'TEST_residual').iloc[0].carriers == clade_set(clade, 0)
    res = rows[rows.sv_category == 'residual']
    assert len(res) == 1 and len(res.iloc[0].carriers) == 6
    assert res.iloc[0].carriers.isdisjoint(clade_set(clade, 0))


def test_branch_r2_is_squared_correlation():
    x = np.array([1, 1, 0, 0, 1, 0], np.int8)
    y = np.array([1, 0, 0, 0, 1, 1], np.int8)
    assert branch_r2(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2)
    assert branch_r2(x, np.ones(6, np.int8)) == 0.0


def test_collapse_keeps_most_carried():
    cols = [np.array([1, 0]), np.array([1, 1]), np.array([0, 1])]
    var = [dict(pos=1, end=5, svtype='DEL', svlen=4, ac=1),
           dict(pos=1, end=5, svtype='DEL', svlen=4, ac=2),
           dict(pos=9, end=9, svtype='INS', svlen=50, ac=1)]
    c, v = collapse_duplicates(cols, var)
    assert [x['ac'] for x in v] == [2, 1] and [x['pos'] for x in v] == [1, 9]


def test_diploid_sums_haplotypes():
    p = np.array([[1, 0], [1, 1], [0, 1]], np.int8)
    d, ind = to_diploid(p, ['B_hap2', 'A_hap1', 'B_hap1'])
    assert ind == ['A', 'B'] and d.tolist() == [[1, 1], [1, 1]]


def test_cli_end_to_end(tmp_path):
    vcf = tmp_path / "example.vcf"
    write_vcf(vcf)
    presence, variants, haps = read_presence(vcf, 'chrS')
    assert presence.shape == (80, 225)
    main(['--vcf', str(vcf), '--chrom', 'chrS', '--out', str(tmp_path / 'out')])
    out = tmp_path / 'out' / 'chrS'
    sv = pd.read_csv(out / 'chrS_sv_variants.tsv', sep='\t')
    g = np.load(out / 'chrS_sv_genotypes.npz')['genotypes']
    assert g.shape == (len(sv), 40)
    assert (g.sum(axis=1) == sv['ac']).all()
    assert set(sv[sv.original_sv_id == 'TEST_two_origin'].sv_category) == {'monophyletic', 'original_decomposed'}
    for f in ('decomposition.tsv', 'branches.tsv', 'validated_haplogroups.tsv', 'nj.nwk', 'nj_annotated.nwk'):
        assert (out / f'chrS_{f}').stat().st_size > 0
