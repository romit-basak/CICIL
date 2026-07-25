"""Stage 2 — sweep driver: the k in {3,5,8} x query-arm grid in one command.

Wires together translate.py's ``--k`` and ``--query-arm`` switches so the
moment a language's retrieval bank lands (an entry is added to build_index.py's
``CORPORA`` dict and the index is built), this is the one command that
produces every prediction file run_ablations.py knows how to read:

    uv run python -m src.stage2.run_sweep                    # everything
    uv run python -m src.stage2.run_sweep --dry-run           # validate wiring, no API calls, no auth needed
    uv run python -m src.stage2.run_sweep --lang wixarika      # one language (e.g. the pilot)
    uv run python -m src.stage2.run_sweep --lang wixarika --dry-run

    # Sweep a specific Stage 1 backend (e.g. once the distilled adapter's
    # outputs cover all 5 languages, not just Wixárika):
    uv run python -m src.stage2.run_sweep --backend smolvlm

Then:
    uv run python -m src.stage2.run_ablations

Runs 3 arms per language (skipping generic+cultural: generic-mode records
carry no cultural_annotations, so that combination would silently collapse to
generic+text and just burn API calls/time for a duplicate result):

  generic       + text     query   (baseline pipeline)
  cultural-vqa  + cultural query   (full treatment / headline pipeline)
  cultural-vqa  + text     query   (RQ1 ablation control: same prompt mode,
                                     same k, same index -- only the retrieval
                                     query changes. This isolates "does
                                     culturally-indexed retrieval beat vanilla
                                     text retrieval?" from Stage 1's own
                                     generic-vs-cultural-vqa prompting effect.)

each at k = 3, 5, 8 -- 9 runs per language, 45 across all 5 languages. Any
language without a built index (see build_index.CORPORA) still runs -- it
just falls back to zero-shot translation, same as translate.py does today --
so this script is safe to run right now, before Nandita's banks land, and
again unchanged after they do.

``--backend`` is a single global choice applied to every run in the grid (not
an added Cartesian dimension) -- it defaults to "ollama" rather than the
distilled "smolvlm" backend on purpose: this script's whole point is isolating
the retrieval-query effect as one clean variable, and defaulting to a
different Stage 1 backend would silently entangle two separate ablations in
the one script designed to keep them apart. It's also not populated for 4/5
languages yet (see STAGE1_HANDOFF.md) -- pass --backend smolvlm explicitly
once that's ready.
"""

from __future__ import annotations

import argparse
import time

from .translate import BACKEND_CHOICES, LANGUAGES, ensure_vertex_credentials, translate_language

# (mode, query_arm) pairs worth actually running. generic+cultural is omitted:
# see module docstring.
RUN_CONFIGS: list[tuple[str, str]] = [
    ("generic", "text"),
    ("cultural-vqa", "cultural"),
    ("cultural-vqa", "text"),
]

K_VALUES = [3, 5, 8]


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2: run the full k x query-arm ablation sweep in one command"
    )
    p.add_argument(
        "--lang", default="all",
        help="Language code (guarani/bribri/maya/wixarika/nahuatl) or 'all' (default)",
    )
    p.add_argument(
        "--backend", default="ollama", choices=BACKEND_CHOICES,
        help="Which Stage 1 backend's outputs to sweep. Applied uniformly to every "
             "run in the grid (not looped/multiplied). Default: ollama.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate load/retrieve/prompt for every run in the grid, no API calls, no auth needed.",
    )
    p.add_argument(
        "--sleep-between-runs", type=float, default=1.0,
        help="Pause (s) between runs, on top of translate.py's own per-call API pause. Default: 1.0",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.dry_run:
        ensure_vertex_credentials()

    langs = LANGUAGES if args.lang == "all" else [args.lang]
    if args.lang != "all" and args.lang not in LANGUAGES:
        raise SystemExit(f"Unknown --lang {args.lang!r}. Expected one of {LANGUAGES + ['all']}.")

    total = len(langs) * len(RUN_CONFIGS) * len(K_VALUES)
    done = 0
    tag = " [DRY-RUN]" if args.dry_run else ""
    print(
        f"Sweep: {len(langs)} lang(s) x {len(RUN_CONFIGS)} arm(s) x {len(K_VALUES)} "
        f"k-value(s), backend={args.backend} = {total} runs{tag}\n"
    )

    for lang in langs:
        for mode, query_arm in RUN_CONFIGS:
            for k in K_VALUES:
                done += 1
                print(f"[{done}/{total}] {lang} | mode={mode} | query-arm={query_arm} "
                      f"| k={k} | backend={args.backend}{tag}")
                translate_language(
                    lang, mode, k,
                    dry_run=args.dry_run,
                    backend=args.backend,
                    query_arm=query_arm,
                )
                if not args.dry_run:
                    time.sleep(args.sleep_between_runs)

    if args.dry_run:
        print("\nDry-run sweep complete -- check predictions/_dryrun/ for sample prompts.")
    else:
        print("\nSweep complete.")
        print("Score everything with:\n    uv run python -m src.stage2.run_ablations")


if __name__ == "__main__":
    main()
