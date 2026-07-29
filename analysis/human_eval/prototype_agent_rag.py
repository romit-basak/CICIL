"""Prototype v4: multi-agent VQA -- an LLM questioner frames the verification.

v3 (prototype_verify_rag.py) showed the 2B VLM can *verify* a match it cannot
spontaneously *assert*: one closed question ("Wikipedia says X looks like
[feature] -- do you see it?") finally produced correct naming where two
open-synthesis designs failed. But v3's questions are hand-coded (patterns ->
textile lace; nothing for huts, terrain, vegetation...), and hand-coding can't
cover the cultural taxonomy.

v4 replaces the hand-coded question bank with an LLM questioner (Gemini Flash,
already in the stack for Stage 2): it reads the VLM's own generic caption plus
per-culture Wikipedia snippets retrieved on that caption, decides which vague
parts are worth probing, and writes up to 3 closed SI/NO/INCIERTO questions,
each tied to a named candidate concept. The VLM answers them against the
image (narrow perception only -- its one reliable skill per v3). Gemini then
assembles the final Spanish caption from the base + Q/A record, under explicit
instructions to preserve the base's directly-observed specifics and add a
concept only when the VLM confirmed it. The 2B synthesis bottleneck is gone
entirely -- Gemini writes the sentence.

Trade-off to acknowledge in the paper: with --llm gemini, Gemini enters Stage
1, so "Stage 1 is a small local model" is no longer cleanly true -- and even
though only text crosses the API (never the image), an LLM that both frames
the questions and writes the caption has extracted what it wanted, so the
sovereignty claim belongs to the --llm ollama rung only.

v4.2 (2026-07-29), after the manual image audit of the v4.1 gemini-config
pilot run found the 2B base caption to be the dominant unrecoverable error
source (hallucinated drum/boat/firewood scenes survived as "protected facts";
the 7B's in-rationale corrections -- "pescado", "en la mano" -- never reached
the caption because they weren't answers to questions):
  1. the ANSWERER (7B) writes the base observation by default (--base
     answerer); the SmolVLM file base remains as --base file for comparison;
  2. a standing ACTION question ("what are the people doing?") joins OCR --
     the audit found actions were structurally unprobed (embroidering,
     feeding calves, loading a truck all went unasked);
  3. neutral-phrasing rule: questions must not presuppose base-claimed
     object identities ("what is she holding?", never "does the kerchief she
     holds have X?" -- the kerchief was a base hallucination that a truthful
     SI about the fabric laundered into the final);
  4. answers outrank the base: the assembler is instructed that when an
     answer's rationale contradicts the base description, the answer wins;
  5. no unverified cultural flourishes ("evoca la vida rural wixárika"
     appeared on a sheep photo with zero verifications).
Plus four rules from scoring the partial v4.1 LOCAL run (whose naive direct
questions beat Gemini on the hardest image: "is he playing a drum?" -> "NO,
he's in a WHEELCHAIR" -> near-gold caption, while Gemini's feature-level
probes never questioned the hallucinated drum at all):
  6. the questioner treats the base as a HYPOTHESIS -- load-bearing unusual
     claims get verified first with a direct question;
  7. direct is-it-traditional-dress questions are explicitly allowed;
  8. no-repetition rule (the local questioner asked the same sandal question
     5x and never stopped voluntarily -- all 6 images hit the cap);
  9. assembler: no meta-language ("no hay evidencia de..." is not a caption)
     and no place names absent from OCR/answers (it invented "Nayarit").

v4.3 (2026-07-30), after auditing the v4.2 pilot run:
  10. NEVER name individuals (questioner + assembler): retrieval injected a
      real Huichol artist's name onto an unidentified man ("posiblemente
      José Benítez Sánchez"). Only exception: a name appearing in the
      image's own legible text. Paper limitation: culturally significant
      public figures who SHOULD be named are under-described; who gets
      named is a community question, not a hyperparameter.
  11. TWO WITNESSES: the 2B's file caption is passed alongside the 7B base
      as an independent second description; contradictions (bridge vs
      "steep path", hch_005) become priority verification targets. Neither
      witness is trusted; this also gives the fine-tuned SmolVLM a real
      role again.
  12. ACTION question extended to held/worn objects (targets the grn_025
      "pañuelo": she holds her spread SKIRT, never asked).
  13. Code-enforced stopping: duplicate questions are dropped; an all-repeat
      batch forces a stop (the 7B questioner never stops voluntarily).
  14. scrub_meta(): report-style sentences ("no hay evidencia de...")
      removed from captions in code; the prompt rule alone slipped once.

Multi-pass mode assembles a caption after EVERY round (read-only: the
questioner never sees these, so the trajectory matches an uninstrumented run).
This buys three things at one assembler call per round: (a) the round-1
caption doubles as the single-pass baseline, so one run yields the whole
quality-vs-rounds curve; (b) interpretability -- what could the model already
say, and what did each later probe surface?; (c) hallucination provenance --
a wrong claim is pinned to the specific round (question + answer) where it
first entered, instead of being diagnosed from the final caption alone (cf.
the grn_025 Itauguá case, which took manual image-forensics to localize).

Qualitative only: 5 known cases (2 successes-to-beat, 3 failure modes), no
training, no production-code changes. Base captions are reused from the
existing generic-arm outputs (deterministic; avoids an extra VLM load).

Run:
    uv run python -m analysis.human_eval.prototype_agent_rag --questions-only
        # Gemini side only (no VLM load -- safe while another SmolVLM job runs)
    uv run python -m analysis.human_eval.prototype_agent_rag
        # full loop (needs the M4 free: SmolVLM answers the questions)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch  # noqa: F401 -- see rag_context.py: must load before faiss on macOS

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

CASES = [
    # (culture, image_id) -- what each isolates:
    ("guarani", "grn_019"),   # can the questioner rediscover the ñandutí probe v3 hand-coded?
    ("guarani", "grn_025"),   # are poster facts (Corrientes) preserved, not overwritten?
    ("wixarika", "hch_021"),  # bare canyon: does it avoid inventing a question/answer? (Wirikuta retraction case)
    ("maya", "yua_001"),      # sleeping cat: zero cultural content -- best output is honest silence
    ("bribri", "bzd_042"),    # wooden artifact: CBIR failure case; can text-side do better?
]

QUESTIONER_PROMPT = """\
Eres un experto en la cultura {culture_name}. Un modelo de visión pequeño describió una imagen así:

"{base_caption}"

Fragmentos de Wikipedia sobre esta cultura, recuperados a partir de esa descripción, con su puntuación de similitud (0-1; por debajo de ~0.40 la relación suele ser casual, no real):
{snippets}

Tu tarea: identifica las partes VAGAS de la descripción que podrían corresponder a un concepto cultural específico de los fragmentos (p. ej. "patrones geométricos en la tela" podría ser un tipo de encaje; "una colina" podría ser un sitio sagrado identificable por rasgos concretos). Para cada una, escribe UNA pregunta de verificación cerrada y puramente VISUAL que el modelo de visión pueda responder mirando la imagen, con SI, NO o INCIERTO. La pregunta debe describir el rasgo visual diagnóstico del concepto, no pedir conocimiento cultural.

Reglas estrictas:
- Máximo 3 preguntas. CERO es una respuesta válida: si la descripción no contiene nada culturalmente ambiguo, o los fragmentos no ofrecen ningún candidato plausible, devuelve una lista vacía.
- No formules preguntas a partir de fragmentos con puntuación baja (<0.40) salvo que el vínculo visual con la descripción sea inequívoco. Una pregunta plausible sobre un concepto irrelevante es peor que ninguna: el modelo de visión tiende a responder SI a preguntas genéricas ("¿hay montañas?"), lo que fabricaría una atribución falsa.
- Pero lo contrario también es un error: si la descripción menciona un rasgo visual genérico (p. ej. "patrones geométricos", "tejido", "bordados") Y un fragmento con puntuación ≥0.45 describe un concepto con un rasgo visual diagnóstico concreto, DEBES formular esa pregunta — no la omitas por prudencia. Ejemplo ilustrativo de otra cultura: descripción dice "una vasija con dibujos en espiral", fragmento "[0.52] Cerámica shipibo: ...patrones laberínticos kené que cubren toda la superficie..." → pregunta: "¿Los dibujos de la vasija forman líneas laberínticas continuas que cubren toda la superficie?"
- NUNCA cuestiones hechos ya específicos y observados directamente (texto legible de carteles, nombres de lugares leídos en la imagen, fechas). Esos se conservan tal cual.
- No inventes conceptos que no estén en los fragmentos.

Responde SOLO con JSON válido, sin markdown:
[{{"vague_part": "...", "concept": "...", "question": "..."}}]
"""

OCR_QUESTION = (
    "¿Hay algún texto legible en la imagen (carteles, letreros, pancartas, "
    "fechas, nombres de lugares)? Si es así, transcríbelo EXACTAMENTE como "
    "aparece, y nada más. Si no hay ningún texto legible, responde "
    "únicamente: NINGUNO."
)

# Examples deliberately DOMAIN-REMOTE (bicycle/computer/garden never appear in
# the datasets): the v4.3 run showed a dataset-drawn example ("sosteniendo el
# borde de su falda extendida", taken from grn_025) gets parroted verbatim
# into other images' extractions where it is false (hch_012, hch_017).
ACTION_QUESTION = (
    "¿Qué está haciendo la persona o las personas de la imagen, si las hay, "
    "y qué sostienen o llevan puesto? Describe brevemente la acción principal "
    "y los objetos sostenidos (p. ej. reparando una bicicleta, escribiendo en "
    "una computadora, regando las plantas de un jardín). Si no hay personas, "
    "responde únicamente: NINGUNA."
)

ITERATIVE_QUESTIONER_PROMPT = """\
Eres un experto en la cultura {culture_name}. Estás interrogando, pregunta a pregunta, a un modelo de visión que está mirando una imagen que tú no puedes ver. Su descripción inicial:

"{base_caption}"

Texto legible que el modelo de visión transcribió de la imagen (hecho observado — no lo cuestiones, pero puede informar tus preguntas):
{ocr_block}

Acción principal que el modelo de visión observó (hecho observado):
{action_block}

Segunda descripción INDEPENDIENTE de la misma imagen, por un modelo de visión distinto (menos fiable en general, pero a veces ve lo que el otro no — un puente que el otro llamó sendero):
{base2_block}

Si las dos descripciones se CONTRADICEN en algo importante (el tipo de estructura, el objeto principal, la escena), resolver esa contradicción con una pregunta directa es prioridad máxima, junto con verificar las afirmaciones centrales de la base.

Fragmentos de Wikipedia sobre esta cultura, recuperados a partir de esa descripción, con su puntuación de similitud (0-1; por debajo de ~0.40 la relación suele ser casual, no real):
{snippets}

Interrogatorio hasta ahora:
{qa_block}

Decide UNA de dos opciones:
1. Formula la SIGUIENTE TANDA de preguntas de verificación cerradas (1 a 3; puramente visuales, respondibles con SI/NO/INCIERTO mirando la imagen, cada una describiendo el rasgo visual diagnóstico de su concepto candidato). Reacciona a las respuestas anteriores: un NO descarta ese concepto — prueba otro candidato plausible o termina; un SI puede merecer una pregunta de seguimiento más específica que afine el hallazgo; un INCIERTO puede merecer una reformulación con un rasgo más fácil de ver.
2. Termina el interrogatorio, si no queda ningún candidato plausible (puntuación ≥0.40 y vínculo visual real con la descripción) sin verificar, o si ya tienes lo necesario.

Mismas reglas que siempre: nunca cuestiones hechos ya observados directamente (texto legible, lugares leídos en la imagen); no inventes conceptos fuera de los fragmentos; una pregunta genérica sobre un concepto irrelevante es peor que ninguna. Si la descripción menciona un rasgo visual genérico y un fragmento con puntuación ≥0.45 describe un rasgo diagnóstico concreto, DEBES sondearlo antes de terminar.

Regla de fraseo NEUTRO: nunca presupongas en la pregunta un objeto o identidad tomado de la descripción base que aún no haya sido confirmado. Pregunta "¿Qué sostiene la mujer?" o "¿El objeto que sostiene la mujer tiene [rasgo]?", nunca "¿El pañuelo que sostiene la mujer tiene [rasgo]?" — si la base se equivocó de objeto, una respuesta afirmativa sobre el rasgo blanquearía el error.

PRIMERA PRIORIDAD — verificar la base: la descripción base es una HIPÓTESIS, no un hecho. Si contiene una afirmación central o inusual (un instrumento musical, un animal exótico, una acción llamativa, un vehículo), verifícala primero con una pregunta directa ("¿El hombre está tocando un tambor?") — una respuesta NO suele venir con la corrección correcta. También es legítima y productiva la pregunta directa sobre vestimenta: "¿La persona lleva vestimenta tradicional {culture_name}?".

Regla de NO REPETICIÓN: nunca vuelvas a formular una pregunta ya respondida, ni una variante trivial de ella. Dos INCIERTO sobre el mismo detalle significan que la imagen no lo resuelve — cambia de tema o termina. Cada ronda debe aportar información nueva.

Regla de NOMBRES PROPIOS: nunca preguntes si una persona de la imagen es un individuo concreto con nombre y apellido (artista, líder, personaje), aunque un fragmento de Wikipedia lo mencione. Los fragmentos describen la cultura en general, no a las personas de esta foto. Si el TEXTO LEGIBLE contiene un nombre de persona, una pregunta prioritaria es a QUÉ etiqueta ese texto: ¿es un gafete o pancarta que identifica a la persona fotografiada, o etiqueta otra cosa (un monumento, un museo, una calle, un evento)?

Responde SOLO con JSON válido, sin markdown. Una de estas dos formas:
[{{"question": "...", "concept": "...", "vague_part": "..."}}]
{{"done": true, "reason": "..."}}
"""

ASSEMBLER_PROMPT = """\
Escribe la descripción final en español (máximo 40 palabras) de una imagen de la cultura {culture_name}.

Descripción base del modelo de visión (sus datos directamente observados son fiables; sus atribuciones culturales no):
"{base_caption}"

Texto legible transcrito de la imagen (hecho directo — si nombra un lugar o evento, úsalo tal cual; nunca lo sustituyas por otro lugar):
{ocr_block}

Acción principal observada (hecho directo — la actividad suele ser lo más importante de la imagen; inclúyela):
{action_block}

Segunda descripción independiente por otro modelo de visión (si contradice la base y el interrogatorio NO resolvió la contradicción, omite el punto en disputa):
{base2_block}

Preguntas de verificación visual y las respuestas del modelo de visión mirando la imagen:
{qa_block}

Reglas estrictas:
- Conserva los hechos concretos observados de la base (objetos, colores, texto legible, lugares leídos en la imagen). No los reemplaces ni contradigas — EXCEPTO cuando una respuesta del interrogatorio los contradiga.
- Las RESPUESTAS del interrogatorio prevalecen sobre la descripción base: si la base dice "pollo" y una respuesta menciona que es pescado, escribe pescado; si la base describe un objeto que una respuesta niega, elimínalo.
- Nombra un concepto cultural SOLO si su verificación fue SI (usa "posiblemente" si fue INCIERTO). Si fue NO o no hubo verificación, no lo menciones.
- No añadas ningún dato cultural que no venga de una verificación, ni frases decorativas del tipo "refleja/evoca/representa la cultura {culture_name}" — la pertenencia cultural la da el contexto del dataset, no la descripción.
- No añadas nombres de lugares, estados o regiones que no aparezcan en el texto legible o en una respuesta del interrogatorio.
- NUNCA identifiques a una persona de la imagen con un nombre propio (ni con "posiblemente"), salvo que el texto legible de la propia imagen sea claramente una etiqueta DE ESA PERSONA (un gafete, una pancarta con su nombre y su foto) — y eso solo si una verificación lo confirmó. Un texto con nombre propio que etiqueta OTRA cosa (un monumento, un museo, una calle, un cartel de evento) NUNCA identifica a la persona fotografiada: menciónalo como texto visible, no como identidad. Un fragmento de Wikipedia que mencione a un artista o líder NO significa que la persona fotografiada sea esa persona.
- Escribe una DESCRIPCIÓN, no un informe: nunca uses frases meta como "no hay evidencia de...", "no se observan elementos...", ni menciones el proceso de verificación. Si algo no se verificó, simplemente no lo menciones.
- Corrige atribuciones culturales erróneas de la base (esta imagen es de la cultura {culture_name}).

Responde SOLO con la descripción final, una sola oración o dos, sin explicación.
"""

CULTURE_NAMES_ES = {
    "guarani": "guaraní (Paraguay/Argentina)",
    "wixarika": "wixárika / huichol (México)",
    "maya": "maya yucateco (México)",
    "bribri": "bribri (Costa Rica)",
    "nahuatl": "náhuatl (México)",
}


def make_llm_call(llm: str, ollama_model: str):
    """The questioner/assembler LLM is pluggable: 'gemini' (hybrid rung) or
    'ollama' (fully-local rung -- e.g. qwen2.5:3b, a text LLM in the same
    parameter class as the VLM but with all of it devoted to reasoning).
    With --llm ollama, nothing derived from the image leaves the machine."""
    if llm == "gemini":
        return call_gemini_raw

    def call_ollama(prompt: str, max_tokens: int = 1024) -> str:
        import ollama

        resp = ollama.chat(
            model=ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "seed": 1234, "num_ctx": 8192,
                     "num_predict": max_tokens},
        )
        return resp["message"]["content"].strip()

    return call_ollama


def call_gemini_raw(prompt: str, max_tokens: int = 1024) -> str:
    """Full-text Gemini call. translate.call_gemini can't be reused here: it
    returns only the FIRST non-empty line (a guard for line-aligned translation
    output -- it reduced our pretty-printed JSON to '['), applies the
    translation system prompt, and caps output at 128 tokens.

    Retries with backoff: multi-pass fires 10-20 calls per image and a Vertex
    429 RESOURCE_EXHAUSTED killed the first v4.3 pilot run mid-image."""
    import time

    from google.genai import types

    from src.stage2.translate import GEMINI_MODEL, GEMINI_SEED, _get_client

    for attempt, delay in enumerate((0, 15, 45, 120)):
        if delay:
            print(f"    (Gemini retry {attempt}, sleeping {delay}s)")
            time.sleep(delay)
        try:
            response = _get_client().models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,  # near-deterministic; extraction, not translation
                    seed=GEMINI_SEED,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    max_output_tokens=max_tokens,
                ),
            )
            return (response.text or "").strip()
        except Exception as e:  # noqa: BLE001 -- one 429 must not kill a 2h run
            if attempt == 3:
                raise
            print(f"    WARNING: Gemini error ({str(e)[:80]}), retrying")
    return ""


META_LANGUAGE_RE = re.compile(
    r"no hay evidencia|no se observa|no se pueden? (?:identificar|determinar|ver)|"
    r"no es posible determinar|sin evidencia de", re.I)

# Decorative culture-flourish clause, either as a whole sentence
# ("Posiblemente refleja la cultura X.") or as a trailing participial clause
# glued to a content sentence (", evocando la cultura X (México)"). The
# no-flourish prompt rule leaks stochastically under prompt stacking
# (2 of 20 in the v4.5-local pilot finals); enforce in code like scrub_meta.
FLOURISH_CLAUSE_RE = re.compile(
    r",?\s*(?:en\s+un\s+(?:entorno|contexto|ambiente)\s+)?(?:posiblemente\s+)?"
    r"(?:evocando|reflejando|representando|que\s+(?:posiblemente\s+)?"
    r"(?:refleja|refleje|evoca|evoque|representa|represente))\s+"
    r"(?:elementos\s+de\s+)?la\s+cultura[^.!?]*", re.I)
FLOURISH_SENTENCE_RE = re.compile(
    r"^\s*(?:posiblemente\s+)?(?:refleja|evoca|representa)\s+(?:elementos\s+"
    r"de\s+)?la\s+cultura", re.I)


def scrub_meta(caption: str) -> str:
    """Drop report-style sentences and decorative culture-flourish clauses
    from a caption (both prompt rules slipped stochastically in local runs;
    enforce them in code)."""
    sentences = re.split(r"(?<=[.!?])\s+", caption)
    kept = [FLOURISH_CLAUSE_RE.sub("", s) for s in sentences
            if not META_LANGUAGE_RE.search(s)
            and not FLOURISH_SENTENCE_RE.search(s)]
    out = " ".join(k.strip() for k in kept if k.strip()).strip()
    return out or caption


def norm_question(q: str) -> str:
    """Normalized form for duplicate detection (v4.1-local asked the same
    sandal question 5x verbatim; the prompt rule alone didn't stop it)."""
    return re.sub(r"[^a-záéíóúñü ]", "", q.lower()).strip()


def parse_questions(raw: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        items = json.loads(text)
        return [q for q in items if isinstance(q, dict) and q.get("question")][:3]
    except (json.JSONDecodeError, TypeError):
        print(f"  WARNING: questioner returned non-JSON, treating as 0 questions: {raw[:120]!r}")
        return []


def parse_decision(raw: str) -> dict:
    """One iterative-round decision: {'questions': [...]} (a batch of 1-3) or
    {'done': ...}. Unparseable output ends the interrogation (conservative
    default). A single bare question dict is tolerated as a batch of one."""
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        decision = json.loads(text)
        if isinstance(decision, list):
            batch = [q for q in decision if isinstance(q, dict) and q.get("question")][:3]
            if batch:
                return {"questions": batch}
        elif isinstance(decision, dict):
            if decision.get("done"):
                return decision
            if decision.get("question"):
                return {"questions": [decision]}
    except (json.JSONDecodeError, TypeError):
        pass
    print(f"  WARNING: unparseable round decision, stopping: {raw[:120]!r}")
    return {"done": True, "reason": "(unparseable output)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--questions-only", action="store_true",
                        help="run only the questioner LLM (no SmolVLM load)")
    parser.add_argument("--llm", default="gemini", choices=["gemini", "ollama"],
                        help="questioner/assembler LLM: gemini (hybrid) or "
                             "ollama (fully local)")
    parser.add_argument("--ollama-model", default="qwen2.5:3b")
    parser.add_argument("--answerer", default="ollama", choices=["smolvlm", "ollama"],
                        help="VLM that answers the verification questions. "
                             "Default ollama (qwen2.5vl:7b): the 2026-07-28 "
                             "decoy control showed the 2B's SI/NO verdict "
                             "token carries no signal (6/8 SI on absent "
                             "concepts; 82/82 SI in the full verify run), "
                             "while the local 7B discriminates cleanly (8/8 "
                             "NO on decoys, 3/4 SI + 1 justified INCIERTO on "
                             "true claims). SmolVLM stays as the base "
                             "captioner either way.")
    parser.add_argument("--multi-pass", action="store_true",
                        help="iterative interrogation: one question per round, "
                             "questioner sees each answer before deciding the "
                             "next (requires the VLM; ignored in "
                             "--questions-only)")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--base", default="answerer", choices=["answerer", "file"],
                        help="who writes the base observation: the answerer "
                             "VLM (v4.2 default -- the audit showed the 2B "
                             "file base is the dominant error source) or the "
                             "pre-generated SmolVLM file (v4.1 behavior)")
    parser.add_argument("--pilot", action="store_true",
                        help="run ALL 20 wixarika pilot images instead of the "
                             "5 dev cases -- the pilot has gold Spanish, so "
                             "per-round captions give a measured "
                             "quality-vs-rounds curve")
    parser.add_argument("--dev-sample", type=int, default=0, metavar="N",
                        help="run N seeded-random dev images per non-wixarika "
                             "culture (guarani/maya/bribri/nahuatl) -- the "
                             "round-4 cross-culture arm. Dev has no Spanish "
                             "gold, so these rows are human-eval only")
    parser.add_argument("--out-jsonl", type=Path, default=None,
                        help="append one record per image (base, ocr, per-round "
                             "qa + captions, final) for scoring/auditing")
    parser.add_argument("--image", type=Path, default=None,
                        help="run on one arbitrary image file (adversarial "
                             "tests); pairs with --culture for the bank")
    parser.add_argument("--culture", default="wixarika",
                        choices=list(CULTURE_NAMES_ES),
                        help="culture bank to use with --image")
    args = parser.parse_args()

    from src.stage1.rag_context import TextBank

    llm_call = make_llm_call(args.llm, args.ollama_model)
    if args.llm == "gemini":
        from src.stage2.translate import ensure_vertex_credentials
        ensure_vertex_credentials()

    backend = None
    if not args.questions_only:
        from src.stage1.backends import get_backend
        if args.answerer == "smolvlm":
            backend = get_backend("smolvlm", adapter=str(OUTPUTS / "adapters" / "distill_full"))
        else:
            backend = get_backend("ollama")  # qwen2.5vl:7b

    from src.stage1.data_io import load_split
    text_banks: dict[str, TextBank] = {}

    split = "pilot" if args.pilot else "dev"
    if args.image:
        cases = [(args.culture, args.image.stem)]
    elif args.pilot:
        cases = [("wixarika", e.id) for e in load_split("wixarika", "pilot")]
    elif args.dev_sample:
        # Same seed as build_round4_pilot so the sample is reproducible.
        rng = random.Random(20260731)
        cases = []
        for culture in ("guarani", "maya", "bribri", "nahuatl"):
            ids = sorted(e.id for e in load_split(culture, "dev"))
            cases.extend((culture, i)
                         for i in sorted(rng.sample(ids, args.dev_sample)))
    else:
        cases = CASES

    # Resume: skip ids already recorded (the first v4.3 pilot run died at
    # image 11 on a Vertex 429; rerunning must not duplicate the first 10).
    if args.out_jsonl and args.out_jsonl.exists():
        done = {json.loads(l)["id"] for l in args.out_jsonl.open(encoding="utf-8")
                if l.strip()}
        if done:
            print(f"[resume] skipping {len(done)} already in {args.out_jsonl.name}")
            cases = [(c, i) for c, i in cases if i not in done]

    for culture, image_id in cases:
        if args.image:
            image_path = args.image
        else:
            ex = next(e for e in load_split(culture, split) if e.id == image_id)
            image_path = ex.image_path
        if culture not in text_banks:
            text_banks[culture] = TextBank(culture)

        print(f"\n{'=' * 70}\n{image_id} ({culture})\n{'=' * 70}")
        if args.base == "answerer" and backend is not None:
            from src.stage1 import vqa_prompts
            base = backend.caption(image_path, vqa_prompts.GENERIC_PROMPT)
            base_source = "answerer"
        else:
            base_path = OUTPUTS / f"{culture}_{split}_generic_smolvlm.jsonl"
            base = next(json.loads(l)["generated_spanish"]
                        for l in base_path.open(encoding="utf-8")
                        if json.loads(l)["id"] == image_id)
            base_source = "smolvlm-file"
        print(f"STEP 0 -- base caption ({base_source}):\n  {base}")

        # Second witness: the 2B's independent description of the same image
        # (on disk for dev + wixarika pilot). Neither witness is trusted --
        # disagreements become priority verification targets (the hch_005
        # bridge that the 7B parsed as a "steep path" but the 2B got right).
        base2_block = "(no disponible)"
        if base_source == "answerer":
            try:
                base2_path = OUTPUTS / f"{culture}_{split}_generic_smolvlm.jsonl"
                base2_block = next(json.loads(l)["generated_spanish"]
                                   for l in base2_path.open(encoding="utf-8")
                                   if json.loads(l)["id"] == image_id)
                print(f"STEP 0.5 -- second witness (smolvlm):\n  {base2_block}")
            except (FileNotFoundError, StopIteration):
                pass

        hits = text_banks[culture].retrieve(base, k=5)
        print("STEP 1 -- retrieval on base caption:")
        for h in hits:
            print(f"  {h['score']:.2f} -- {h['title']}")
        # Hard floor in CODE, not just the prompt: in the v4.1 run the Gemini
        # questioner broke its own <0.40 rule and built a leading question from
        # a 0.21 snippet ("Sierra Madre Oriental"), which the answerer's
        # inevitable SI to "are there rock formations?" turned into a fabricated
        # specific claim -- the hch_021/Wirikuta failure through a new door.
        # 0.30 keeps mid-band descriptive candidates but drops the junk tail.
        hits = [h for h in hits if h["score"] >= 0.30]
        snippets = "\n".join(f"- [{h['score']:.2f}] {h['title']}: {h['extract'][:250]}"
                             for h in hits) or "- (ninguno: ningún fragmento supera el umbral de similitud)"

        def ask_vlm(q: dict) -> str:
            return backend.caption(
                image_path,
                q["question"] + " Responde únicamente con una palabra: SI, NO, o "
                                "INCIERTO, seguida de una breve razón.")

        qa_lines: list[str] = []

        # OCR via the ANSWERER (the 7B reads reliably; the 2B doesn't -- see
        # the v3 grn_025 finding). Extracted once, passed to questioner and
        # assembler as a given fact.
        ocr_block = "(no extraído)"
        action_block = "(no extraído)"
        if backend is not None:
            ocr = backend.caption(image_path, OCR_QUESTION)
            ocr_block = ("(ninguno)" if re.search(r"^\s*ninguno", ocr, re.I)
                         or len(ocr.strip()) < 3 else ocr.strip()[:300])
            print(f"STEP 1.5 -- legible text (answerer): {ocr_block}")
            action = backend.caption(image_path, ACTION_QUESTION)
            action_block = ("(ninguna persona)" if re.search(r"^\s*ninguna", action, re.I)
                            or len(action.strip()) < 3 else action.strip()[:300])
            print(f"STEP 1.6 -- main action (answerer): {action_block}")

        rounds_log: list[dict] = []
        stop_reason = None

        if args.multi_pass and not args.questions_only:
            # Iterative "20 questions" (capped): a batch of 1-3 probes per
            # round; the questioner sees all answers before deciding the next
            # batch or stopping.
            transcript: list[str] = []
            asked: set[str] = set()
            for round_i in range(1, args.max_rounds + 1):
                decision = parse_decision(llm_call(ITERATIVE_QUESTIONER_PROMPT.format(
                    culture_name=CULTURE_NAMES_ES[culture], base_caption=base,
                    snippets=snippets, ocr_block=ocr_block, action_block=action_block,
                    base2_block=base2_block,
                    qa_block="\n".join(transcript) or "(aún ninguna pregunta)")))
                if decision.get("done"):
                    stop_reason = decision.get("reason", "?")
                    print(f"STEP 2.{round_i} -- questioner ({args.llm}) stops: "
                          f"{stop_reason}")
                    break
                # Code-enforced no-repetition: drop already-asked questions;
                # if the whole batch is repeats, the questioner is out of
                # ideas -- force the stop it won't take itself.
                fresh = [q for q in decision["questions"]
                         if norm_question(q["question"]) not in asked]
                if not fresh:
                    stop_reason = "(forced: batch repeated earlier questions)"
                    print(f"STEP 2.{round_i} -- {stop_reason}")
                    break
                asked.update(norm_question(q["question"]) for q in fresh)
                round_qa: list[dict] = []
                for q in fresh:
                    answer = ask_vlm(q)
                    round_qa.append({"concept": q.get("concept", "?"),
                                     "question": q["question"], "answer": answer})
                    transcript.append(f"P ({q.get('concept', '?')}): "
                                      f"{q['question']}\nR: {answer}")
                    qa_lines.append(f"- Concepto candidato: {q.get('concept', '?')}\n"
                                    f"  Pregunta: {q['question']}\n  Respuesta: {answer}")
                    print(f"STEP 2.{round_i} -- [{q.get('concept', '?')}] "
                          f"{q['question']}\n  VLM: {answer}")
                # Per-round caption: probes the state without feeding back into
                # the interrogation (the questioner only ever sees the Q/A
                # transcript), so the trajectory matches an uninstrumented run
                # and the round-1 caption doubles as the single-pass result --
                # one run yields the whole quality-vs-rounds curve.
                round_caption = scrub_meta(llm_call(ASSEMBLER_PROMPT.format(
                    culture_name=CULTURE_NAMES_ES[culture], base_caption=base,
                    ocr_block=ocr_block, action_block=action_block,
                    base2_block=base2_block, qa_block="\n".join(qa_lines))))
                rounds_log.append({"round": round_i, "qa": round_qa,
                                   "caption": round_caption})
                print(f"  CAPTION after round {round_i}: {round_caption}")
            else:
                print(f"STEP 2 -- hit --max-rounds cap ({args.max_rounds})")
        else:
            questions = parse_questions(llm_call(QUESTIONER_PROMPT.format(
                culture_name=CULTURE_NAMES_ES[culture], base_caption=base, snippets=snippets)))
            print(f"STEP 2 -- questioner ({args.llm}) framed {len(questions)} question(s):")
            for q in questions:
                print(f"  [{q.get('concept', '?')}] {q['question']}\n"
                      f"    (targets: {q.get('vague_part', '?')})")

            if args.questions_only:
                continue

            for q in questions:
                answer = ask_vlm(q)
                qa_lines.append(f"- Concepto candidato: {q.get('concept', '?')}\n"
                                f"  Pregunta: {q['question']}\n  Respuesta: {answer}")
                print(f"STEP 3 -- VLM answer [{q.get('concept', '?')}]: {answer}")

        final = scrub_meta(llm_call(ASSEMBLER_PROMPT.format(
            culture_name=CULTURE_NAMES_ES[culture], base_caption=base,
            ocr_block=ocr_block, action_block=action_block, base2_block=base2_block,
            qa_block="\n".join(qa_lines) or "(ninguna verificación -- usa solo la base)")))
        print(f"STEP 4 -- assembled final caption ({args.llm}):\n  {final}")

        if args.out_jsonl:
            with args.out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "id": image_id, "culture": culture, "split": split,
                    "llm": args.llm, "answerer": args.answerer,
                    "version": "v4.3", "base_source": base_source,
                    "base": base, "base2": base2_block,
                    "ocr": ocr_block, "action": action_block,
                    "rounds": rounds_log, "stop_reason": stop_reason,
                    "final": final,
                }, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 70}\nDone. Qualitative comparison only -- no new eval numbers.")


if __name__ == "__main__":
    main()
