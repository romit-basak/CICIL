"""v4.4: fix the aggregation flaw by RE-ASSEMBLING frozen v4.3 transcripts.

The v4.3 validation isolated the last structural failure: assembly trusts
verdict tokens over rationale content, and confidence over caution. hch_005:
the answerer wrote "puente colgante o pasarela" in three rationales but its
one confident SI was for "sendero" -- the final said sendero (inventing
"hormigón"). Same mechanism let a late confident SI beat an early "el texto
parece superpuesto" in the adversarial museum test, and let the 2B witness's
hallucinated "pañuelo" merge into grn_025's final instead of being omitted.

Assembly is a pure function of recorded state (base, base2, ocr, action, qa
transcript), all stored in the --out-jsonl records -- so v4.4 re-runs ONLY
the assembler over the frozen v4.3 transcripts. No VLM, no interrogation:
~1 Gemini call per record. This doubles as the clean ablation for the paper:
same evidence, two aggregation policies, diff the captions.

New aggregation rules (on top of v4.3's):
  1. rationales outrank verdict tokens -- if an answer's REASON describes
     something different from its one-word verdict, trust the reason;
  2. under disagreement between answers on the same point, prefer the most
     CAUTIOUS reading or omit the point;
  3. witness conflicts unresolved by the interrogation are OMITTED, never
     merged (don't take one object from each witness).

Run:
    uv run python -m analysis.human_eval.reassemble_v44 \
        outputs/agent_rag_pilot_curve_gemini_v43.jsonl \
        outputs/agent_rag_dev_probe_v43.jsonl
    # writes <input stem>_v44.jsonl next to each input
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.human_eval.prototype_agent_rag import (
    CULTURE_NAMES_ES,
    call_gemini_raw,  # noqa: F401 -- via make_llm_call("gemini", ...)
    make_llm_call,
    scrub_meta,
)

ASSEMBLER_PROMPT_V44 = """\
Escribe la descripción final en español (máximo 40 palabras) de una imagen de la cultura {culture_name}.

Descripción base del modelo de visión (HIPÓTESIS, no hechos):
"{base_caption}"

Segunda descripción independiente por otro modelo de visión (también HIPÓTESIS):
{base2_block}

Texto legible transcrito de la imagen (hecho directo — si nombra un lugar o evento, úsalo tal cual; nunca lo sustituyas por otro lugar):
{ocr_block}

Acción principal observada (hecho directo — la actividad suele ser lo más importante de la imagen; inclúyela):
{action_block}

Preguntas de verificación visual y las respuestas del modelo de visión mirando la imagen:
{qa_block}

Reglas de agregación (las más importantes):
- Las respuestas valen por su RAZONAMIENTO, no por su primera palabra. Si la razón de una respuesta describe otra cosa que su veredicto SI/NO (p. ej. un veredicto afirma "sendero" pero varias razones describen "un puente colgante o pasarela"), confía en lo que las razones DESCRIBEN, no en los veredictos.
- Bajo desacuerdo entre respuestas sobre el mismo punto, prefiere la lectura más CAUTELOSA, o omite el punto si no hay lectura segura.
- Si las dos descripciones base se contradicen en un objeto o estructura y el interrogatorio no lo resolvió con claridad, OMITE el punto en disputa. Nunca fusiones objetos tomados de testigos distintos (no añadas un objeto que solo un testigo vio y ninguna respuesta confirmó).
- No inventes materiales, colores o detalles que ninguna fuente menciona.

Reglas heredadas:
- Nombra un concepto cultural SOLO si su verificación fue SI (usa "posiblemente" si fue INCIERTO). Si fue NO o no hubo verificación, no lo menciones.
- No añadas ningún dato cultural que no venga de una verificación, ni frases decorativas del tipo "refleja/evoca/representa la cultura {culture_name}".
- No añadas nombres de lugares, estados o regiones que no aparezcan en el texto legible o en una respuesta.
- NUNCA identifiques a una persona de la imagen con un nombre propio; un texto legible con nombre propio que etiqueta OTRA cosa (museo, monumento, calle, evento) no identifica a la persona — menciónalo como texto visible.
- Escribe una DESCRIPCIÓN, no un informe: nada de "no hay evidencia de..." ni menciones del proceso de verificación.

Responde SOLO con la descripción final, una sola oración o dos, sin explicación.
"""


ADJUDICATION_PROMPT = """\
Eres el juez de un interrogatorio visual sobre una imagen de la cultura {culture_name}. Tienes dos descripciones (HIPÓTESIS de dos modelos de visión distintos), texto legible y acción observada (hechos directos), y un interrogatorio de preguntas y respuestas.

Descripción A: "{base_caption}"
Descripción B: {base2_block}
Texto legible (hecho): {ocr_block}
Acción observada (hecho): {action_block}

Interrogatorio:
{qa_block}

Tu tarea: produce una LISTA DE HECHOS RESUELTOS. Para cada punto en disputa (entre las dos descripciones, o entre respuestas del interrogatorio), decide así:
- El RAZONAMIENTO de las respuestas pesa más que su veredicto de una palabra: si un veredicto dice "sendero" pero tres razones describen "un puente colgante o pasarela", el hecho resuelto es lo que las razones describen.
- Bajo desacuerdo sin resolución clara, marca el punto como OMITIR.
- Nunca inventes materiales, colores o detalles que ninguna fuente menciona.

Responde SOLO con líneas, sin explicación adicional:
HECHO: <hecho resuelto>
OMITIR: <punto en disputa sin resolución>
(los hechos directos — texto legible, acción — van como HECHO tal cual)
"""

CAPTION_FROM_FACTS_PROMPT = """\
Escribe la descripción final en español (máximo 40 palabras) de una imagen de la cultura {culture_name}, usando EXCLUSIVAMENTE los hechos resueltos siguientes. No añadas nada que no esté en la lista; ignora los puntos marcados OMITIR.

{facts}

Reglas: nombra conceptos culturales solo si un hecho los afirma (con "posiblemente" si el hecho está hedgeado); nunca identifiques personas con nombre propio (un texto que etiqueta un museo/monumento/evento no identifica a la persona); nada de frases meta ni decorativas. Responde SOLO con la descripción, una o dos oraciones.
"""


WHITELIST_RULE = """

CONCEPTOS CULTURALES PERMITIDOS — los ÚNICOS términos culturales específicos (artesanías, prendas, danzas, sitios) que puedes nombrar en la descripción, y solo si el interrogatorio los apoyó (SI, o "posiblemente" con INCIERTO):
{whitelist}

Las respuestas del interrogatorio pueden CONFIRMAR o NEGAR estos conceptos, pero NUNCA introducir términos culturales nuevos. Si una respuesta nombra un término cultural que NO está en la lista (p. ej. el nombre de una prenda o una danza), NO lo uses: describe el objeto de forma genérica ("traje tradicional", "falda con volantes"). La cultura {culture_name} en sí siempre puede mencionarse.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--v45", action="store_true",
                        help="v4.5: assembler may only NAME cultural concepts "
                             "present in the retrieved snippets (recomputed "
                             "deterministically from the stored base caption) "
                             "-- closes the parametric-injection leak found in "
                             "the Argentine-dress probe ('traje guasú', a "
                             "fabricated garment name introduced by an ANSWER, "
                             "asserted as fact by the v4.3/v4.4 assemblers)")
    parser.add_argument("--llm", default="gemini", choices=["gemini", "ollama"],
                        help="assembler LLM: gemini, or ollama (qwen2.5:7b) so "
                             "the fully-local arm's finals stay fully local")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--two-stage", action="store_true",
                        help="v4.4b: adjudicate disputed points into a resolved-"
                             "facts list first, then caption from the facts "
                             "(CoVe-style reason-then-write; targets verdict "
                             "dominance that the prompt rules alone don't fix)")
    args = parser.parse_args()

    from src.stage2.translate import ensure_vertex_credentials
    ensure_vertex_credentials()

    suffix = "_v45" if args.v45 else ("_v44b" if args.two_stage else "_v44")
    if args.llm == "ollama":
        suffix += "-local"
    llm_call = make_llm_call(args.llm, args.ollama_model)
    banks: dict = {}
    if args.v45:
        from src.stage1.rag_context import TextBank

    for path in args.files:
        out_path = path.with_name(path.stem + suffix + ".jsonl")
        done = set()
        if out_path.exists():
            done = {json.loads(l)["id"] for l in out_path.open(encoding="utf-8") if l.strip()}
        with out_path.open("a", encoding="utf-8") as out:
            for rec in (json.loads(l) for l in path.open(encoding="utf-8") if l.strip()):
                if rec["id"] in done:
                    continue
                qa_lines = [f"- Concepto candidato: {qa['concept']}\n  Pregunta: {qa['question']}\n  Respuesta: {qa['answer']}"
                            for rd in rec["rounds"] for qa in rd["qa"]]
                common = dict(
                    culture_name=CULTURE_NAMES_ES[rec["culture"]],
                    base_caption=rec["base"], base2_block=rec.get("base2", "(no disponible)"),
                    ocr_block=rec["ocr"], action_block=rec["action"],
                    qa_block="\n".join(qa_lines) or "(ninguna verificación -- usa solo las descripciones base)")
                record = {**rec, "final_v43": rec["final"]}
                if args.v45:
                    culture = rec["culture"]
                    if culture not in banks:
                        banks[culture] = TextBank(culture)
                    hits = [h for h in banks[culture].retrieve(rec["base"], k=5)
                            if h["score"] >= 0.30]
                    whitelist = "\n".join(f"- {h['title']}: {h['extract'][:200]}"
                                          for h in hits) or "- (ninguno: no nombres ningún concepto cultural específico)"
                    prompt = ASSEMBLER_PROMPT_V44.format(**common) + WHITELIST_RULE.format(
                        whitelist=whitelist, culture_name=common["culture_name"])
                    final = scrub_meta(llm_call(prompt))
                    record.update(version="v4.5-whitelist", final=final,
                                  whitelist=[h["title"] for h in hits])
                elif args.two_stage:
                    facts = llm_call(ADJUDICATION_PROMPT.format(**common))
                    final = scrub_meta(llm_call(CAPTION_FROM_FACTS_PROMPT.format(
                        culture_name=common["culture_name"], facts=facts)))
                    record.update(version="v4.4b-two-stage", facts=facts, final=final)
                else:
                    final = scrub_meta(llm_call(ASSEMBLER_PROMPT_V44.format(**common)))
                    record.update(version="v4.4-reassembly", final=final)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{rec['id']}\n  v4.3: {rec['final'][:120]}\n  {suffix[1:]}: {final[:120]}")
        print(f"-> {out_path.name}")


if __name__ == "__main__":
    main()
