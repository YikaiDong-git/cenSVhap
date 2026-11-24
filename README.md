# cenSVhap

Haplogroup resolution of centromeric structural variants on a haplotype neighbor-joining tree.

```bash
pip install .
censvhap --vcf cohort_svs.vcf.gz --chrom chr1 chr2 chrX --out resolved/
```

Input: a multi-sample VCF with one column per haplotype (`<id>_hap1`, `<id>_hap2`).
Output per chromosome: `<chr>_sv_variants.tsv`, `<chr>_sv_genotypes.npz`, `<chr>_sv_samples.txt`
(the input of [cenSVimp](https://github.com/YikaiDong-git/cenSVimp)). Options: `censvhap --help`.

License: MIT. Contact: yikai.dong@emory.edu, josh.weinstock@emory.edu

## Citation

Dong Y, Cutler D, Weinstock JS. Common centromeric variation associates with molecular and complex
traits in humans. *medRxiv* 2026. [10.64898/2026.10.09.26365154](https://doi.org/10.64898/2026.10.09.26365154)

```bibtex
@article{dong2026centromeric,
  title   = {Common centromeric variation associates with molecular and complex traits in humans},
  author  = {Dong, Yikai and Cutler, David and Weinstock, Joshua S.},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.09.26365154}
}
```
