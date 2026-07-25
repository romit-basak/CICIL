# Data Licenses & Attribution

Source, license, and citation for every third-party corpus under `Dataset/`. This data
wasn't created by this project — keep this file attached if it's ever moved or
repackaged.

## Retrieval-bank corpora (`Dataset/*.jsonl`)

Built by `byuild_corpora.py` from AmericasNLP 2021 + 2023 train+dev splits (each
language's `.es` file paired with its target-side file), deduplicated on exact
(spanish, target) pairs. Format: one JSON object per line,
`{"spanish_caption": ..., "target_caption": ...}`.

| File | Pairs | Source repo | License | Cite |
|---|---|---|---|---|
| `bribri.jsonl` | 8,297 | `AmericasNLP/americasnlp2021` + `americasnlp2023`, `bribri-spanish/` | Not explicitly stated (no LICENSE file in the source repos) | Feldman & Coto-Solano (2020), "Neural Machine Translation Models with Back-Translation for the Extremely Low-Resource Indigenous Language Bribri" |
| `guarani.jsonl` | 15,494 | `AmericasNLP/americasnlp2021` + `americasnlp2023`, `guarani-spanish/` | Not explicitly stated | AmericasNLP 2021/2023 shared task overview papers |
| `nahuatl.jsonl` | 16,119 | `AmericasNLP/americasnlp2021` + `americasnlp2023`, `nahuatl-spanish/` | Not explicitly stated | AmericasNLP 2021/2023 shared task overview papers |
| `wixarika.jsonl` | 9,940 | `AmericasNLP/americasnlp2021` + `americasnlp2023`, `wixarika-spanish/` | Underlying corpus is **CC BY-NC 4.0** (large-corpus file: CC BY-NC-SA 4.0), per the original source `pywirrarika/wixarikacorpora` | Mager et al. (2018); cite `pywirrarika/wixarikacorpora` |
| `maya.jsonl` | 14,332 | `github.com/alemol/yua-es-ccc` — YUA-ES Communicative Contexts Corpus (Molina-Villegas et al., 2026; CentroGeo + SEDECULTA, Yucatán state government) | **CC-BY-4.0**, explicit, purpose-built for MT reuse | Molina-Villegas, A., Chavarrea Chim, M.E., Hau Ucán, S.L., Cano Chan, D.E., Moo Batun, A.d.R., Hau Caamal, G. (2026). *Yua-es communicative contexts corpus.* Zenodo. https://doi.org/10.5281/zenodo.20631792 — and per the source README, "if you use these data for any AmericasNLP task, you must cite this corpus." The raw TSV + upstream `LICENSE` file are kept alongside as `Dataset/yua_es_ccc_v1.tsv` / `Dataset/YUA_ES_CCC_LICENSE` for provenance. |

**On "not explicitly stated":** the AmericasNLP GitHub repos carry no LICENSE file or
repo-level terms — only a request to cite the original papers. That isn't the same as
public domain; strictly it defaults to all-rights-reserved. But the shared task
organizers themselves redistribute these exact files on public GitHub repos under that
same no-license-file, cite-the-source convention — hosting them here (an academic,
non-commercial project, following the same norm) matches established practice in this
subfield rather than deviating from it.

**Wixárika is under an explicit non-commercial restriction (CC BY-NC 4.0):** our use
(non-commercial coursework/paper) is squarely inside what that license permits. One
caveat worth naming: this repo is public, so in principle a third party could pull the
file and use it commercially, which the license wouldn't permit them to do — the same
exposure the original `pywirrarika/wixarikacorpora` repo already carries by being public
itself. Not a risk introduced by this project, just worth being aware of.

**Maya is the cleanest file in the set (CC BY-4.0, no NC restriction):** cloned directly
from `github.com/alemol/yua-es-ccc` (verified live, 2026-07-25) — a real, purpose-built
open release (v1.0.0, June 2026) explicitly designed for MT reuse, unlike the other four
files' "cite the source, no license file" status. Its own README requires citing the
Zenodo record if used for any AmericasNLP-related task (done above).

## Ruled out for Yucatec Maya (found during the search, not usable)

- **MayanV** (`github.com/transducens/mayanv`, CC0) — covers 15 other Mayan languages,
  does not include Yucatec Maya at all.
- **jw.org / JW300-derived Yucatec Maya corpus** — a large (~263k sentence) Spanish-Maya
  corpus exists (used internally by the MayanV paper's authors for model training, not
  publicly released) sourced from jw.org. Explicitly avoided: this is the same source
  behind JW300, and Masakhane was forced to stop using JW300 for African-language MT
  after the Jehovah's Witnesses organization confirmed its site prohibits text/data
  mining and denied a formal request for permission. The YUA-ES-CCC paper itself flags
  this same corpus as legally non-distributable for exactly this reason.
- **AmericasNLP 2024 ST2 (educational materials) Maya data** — real and CC-BY-4.0-adjacent
  (it's the labeled subset of YUA-ES-CCC used for that shared task), but the released
  task format is Maya→Maya morphosyntactic transformation (`Source`/`Change`/`Target`
  columns, e.g. negation/interrogative transforms), **not** Spanish-paired — useless for
  retrieval as released. Use YUA-ES-CCC's own parallel release instead (which is what we
  did).
- **CPLM** (Sierra et al., 2020) — has a Yucatec Maya portion, but no explicit license and
  no downloadable dataset (query-interface only, per the corpus's own paper).

## Considered but not currently used

- **MultiScript30k** (Driggers-Ellis et al., 2025) — a synthetic, NLLB-generated Guaraní
  corpus (~30k pairs); the paper is CC BY 4.0. `STAGE2_HANDOFF.md` listed this as a
  possible additional Guaraní source, but it is **not** mixed into `guarani.jsonl` —
  that file is AmericasNLP 2021+2023 train/dev only, per `byuild_corpora.py`. Noting
  this so nobody assumes MultiScript30k pairs are already included.
- **py-elotl** (`elotl.corpus.load('axolotl')`) — an alternative Nahuatl source, not used
  here (`nahuatl.jsonl` comes from AmericasNLP instead). If swapped in later: the
  `py-elotl` *package* is MPL 2.0, but no explicit license was found for the axolotl
  corpus data itself — same "cite it, no stated license" situation as the AmericasNLP
  files above.

## Provenance note

`byuild_corpora.py` (repo root) is the conversion script that produced these files.
Its `REPO_2021`/`REPO_2023` constants point at a local Downloads folder, so it isn't
runnable as-is in this repo — it's kept as a record of exactly how the data was built
(train+dev concatenated, deduplicated by exact (spanish, target) pair), in case it needs
re-running against a different corpus snapshot. Its own `OUT_DIR` constant
(`data/corpora`) doesn't match where the files actually live (`Dataset/`, repo root) —
they were added via a direct GitHub upload rather than by running the script inside a
checked-out copy of this repo. `src/stage2/build_index.py` points at the real location,
`Dataset/`.
