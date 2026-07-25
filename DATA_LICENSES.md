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
| *(no file)* | — | Yucatec Maya | — | **No retrieval-bank source found.** Not in AmericasNLP 2021's language list; flagged as an open gap in `STAGE2_HANDOFF.md`, still unresolved. |

**On "not explicitly stated":** the AmericasNLP GitHub repos carry no LICENSE file or
repo-level terms — only a request to cite the original papers. That isn't the same as
public domain; strictly it defaults to all-rights-reserved. But the shared task
organizers themselves redistribute these exact files on public GitHub repos under that
same no-license-file, cite-the-source convention — hosting them here (an academic,
non-commercial project, following the same norm) matches established practice in this
subfield rather than deviating from it.

**Wixárika is the one file under an explicit restriction (CC BY-NC 4.0):** our use
(non-commercial coursework/paper) is squarely inside what that license permits. One
caveat worth naming: this repo is public, so in principle a third party could pull the
file and use it commercially, which the license wouldn't permit them to do — the same
exposure the original `pywirrarika/wixarikacorpora` repo already carries by being public
itself. Not a risk introduced by this project, just worth being aware of.

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
