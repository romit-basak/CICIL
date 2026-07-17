"""Authoritative preliminary ChrF++ numbers, in one place with provenance.

These are the results reported in the preliminary paper (``acl2023/cicil_prelim.tex``,
Tables 2 & 3) and cross-checked against ``STAGE1_HANDOFF.md``. They are hard-coded
here because the dev *gold references* live under ``data/`` (gitignored, CC BY-NC),
so the end-to-end scores cannot be recomputed from this repo alone. When the dataset
is mounted, ``src.stage1.evaluate`` reproduces them from ``predictions/``.

Keeping the numbers in a module (rather than inline in the plotting script) means
the figure, any tables, and future analyses all read the *same* source of truth.
Update this file when a run changes, and every downstream artifact updates with it.
"""

from __future__ import annotations

from dataclasses import dataclass

# Display order = descending official baseline (best-resourced first). Keeps the
# figure legible and groups the two degenerate low-resource cases at the right.
LANGUAGE_ORDER = ["Guaraní", "Yucatec Maya", "Wixárika", "Nahuatl", "Bribri"]


@dataclass(frozen=True)
class LangResult:
    language: str
    official_baseline: float | None  # shared-task MT baseline; None = no MT baseline
    generic: float                   # RQ1 control  (no cultural prompting)
    cultural_vqa: float              # RQ1 treatment (+ cultural annotations)

    @property
    def delta(self) -> float:
        """Cultural-VQA minus generic — the RQ1 headline per language."""
        return round(self.cultural_vqa - self.generic, 2)


# End-to-end dev ChrF++ at k=5 (Table 3 of the prelim paper; STAGE1_HANDOFF.md).
# n = 50 per language (Wixárika dev uses the 20-image pilot upstream).
RESULTS: dict[str, LangResult] = {
    "Guaraní":      LangResult("Guaraní",      20.82, 21.26, 20.70),
    "Yucatec Maya": LangResult("Yucatec Maya", None,  19.25, 19.56),
    "Wixárika":     LangResult("Wixárika",     17.77, 8.97,  9.19),
    "Nahuatl":      LangResult("Nahuatl",      11.53, 13.96, 15.52),
    "Bribri":       LangResult("Bribri",       7.57,  4.91,  4.41),
}

# Below this, per-language ChrF++ differences at n<=50 are not meaningful
# (stated in the paper's Limitations; used to grey out noise-level deltas).
NOISE_BAND = 0.6


def ordered_results() -> list[LangResult]:
    return [RESULTS[name] for name in LANGUAGE_ORDER]


def mean_delta() -> float:
    deltas = [r.delta for r in RESULTS.values()]
    return round(sum(deltas) / len(deltas), 2)


if __name__ == "__main__":
    print(f"{'language':<14}{'base':>7}{'generic':>9}{'cultural':>10}{'Δ':>8}")
    print("-" * 48)
    for r in ordered_results():
        base = f"{r.official_baseline:.2f}" if r.official_baseline is not None else "—"
        flag = "" if abs(r.delta) >= NOISE_BAND else "  (noise)"
        print(f"{r.language:<14}{base:>7}{r.generic:>9.2f}{r.cultural_vqa:>10.2f}{r.delta:>+8.2f}{flag}")
    print("-" * 48)
    print(f"{'mean Δ':<14}{'':>7}{'':>9}{'':>10}{mean_delta():>+8.2f}")
