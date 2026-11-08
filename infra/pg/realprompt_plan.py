"""REALPROMPT plan, frozen before any launch (studies/x/realprompt/PREREG.md). The job and the scorer both read it.

Each pole gets the prompts of one domain; the push is the NEURONRW residual push (neuron_rw_core.ResidualSteer,
alpha_abs = dose * the layer's Arm 0 residual RMS). The six code neurons also get a random control neuron from the
same layer with the same sign, drawn by `draw_control` (numpy default_rng([SEED, layer, neuron]), redrawn while equal
to the target); the indices below are its output, written down before launch and re-checked by the dry run.
"""
from __future__ import annotations

import re

import numpy as np

SEED = 20261025
DOMAINS = ("code", "email", "news", "essay", "story")
PROMPTS = {                                  # [0] is the user's domain prompt, [1] the second in-domain opener
    "code": ["Here is the code:\n```", "Here is the program:\n```"],
    "email": ["Here is the email:\n", "Here is the email:\n\nSubject:"],
    "news": ["Here is the news report:\n", "Here is the news report:\n\nHeadline:"],
    "essay": ["Here is the essay:\n", "Here is the essay:\n\nTitle:"],
    "story": ["Here is the story:\n", "Here is the story:\n\nChapter 1\n\n"],
}
DOSES = {"Q4B": [0, 4, 8, 16, 32], "G4": [0, 2, 4, 8, 16], "O7": [0, 2, 4, 8, 16]}   # residual-RMS units
N_SAMPLES = 10
SAMPLING = {"temperature": 0.7, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0, "max_new_tokens": 128}
MAX_ROWS = 160
D_MLP = {"Q4B": 9728, "G4": 10240}            # banked config.json (weights bucket); O7 has no control neuron
MODELS = {                                    # studies/neuronrw/SPEC.md §9.4 and plan/arm2_<M>.json
    "Q4B": {"repo": "Qwen/Qwen3-4B-Base", "branch": None, "tokname": "qwen3",
            "revision_expected": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
            "rms": "s3://pg-lineage-769063704696/results/neurons/rw/arm0_q4b/1/out/arm0/rms.json"},
    "G4": {"repo": "google/gemma-3-4b-pt", "branch": None, "tokname": "gemma3",
           "revision_expected": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
           "rms": "s3://pg-lineage-769063704696/results/neurons/rw/arm0_g4/1/out/arm0/rms.json"},
    "O7": {"repo": "allenai/Olmo-3-1025-7B", "branch": "stage1-step1413814", "tokname": "olmo3",
           "revision_expected": "373bad25002f1624757a73235c5ca844c6375c25",
           "rms": "s3://pg-lineage-769063704696/results/neurons/rw/arm0_o7/3/out/arm0/rms.json"},
}

# The six code neurons: NEURONOFF SPEC §5.4 regexes, verbatim; the random same-layer control (draw_control).
CODE = {
    "L11N5238+": {"model": "Q4B", "label": "SQL FOREIGN KEY", "regex": r"(?i)\bforeign\s+key\b", "ctrl": "L11N3497+"},
    "L27N2924-": {"model": "Q4B", "label": "Angular", "regex": r"\bAngular(JS)?\b|@angular/", "ctrl": "L27N7222-"},
    "L08N3398-": {"model": "Q4B", "label": "HTML title attribute",
                  "regex": r"""\stitle\s*=\s*["']|\btitle\s+attribute\b|attr\(\s*['"]title""", "ctrl": "L08N3836-"},
    "L07N10202-": {"model": "G4", "label": "React useEffect", "regex": r"\buseEffect\b", "ctrl": "L07N2158-"},
    "L06N599+": {"model": "G4", "label": "Python import ... as",
                 "regex": r"\bimport\s+[\w.]+\s+as\s+\w+|\bfrom\s+[\w.]+\s+import\s+\w+\s+as\s+\w+", "ctrl": "L06N3187+"},
    "L08N7823+": {"model": "G4", "label": "SQL REFERENCES", "regex": r"""(?i)\breferences\s+[\w.`"\[\]]+\s*\(""",
                  "ctrl": "L08N6357+"},
}

# Every pole: (pole, domain, source, theme). source: page (the user's page), arm1 (NEURONRW Arm 1 target), best
# (curate/BEST.md top 25). theme: a regex written from the pole's re-explained label before any sample of this run
# was read; it ranks the shown samples and gives a descriptive count, never a test. Code poles use their construct.
POLES = {
    "Q4B": [
        ("L32N6216+", "essay", "page", r"(?i)\boutline|\bthesis|\bdraft|\bdiagram|\btimeline|\bbrainstorm"),
        ("L32N9698-", "news", "page", r"(?i)\bstroke|\bcancer|spinal cord|\bdisease|\billness|\bdiagnos|\bpatient|"
                                      r"\btumou?r|\bdementia|\balzheimer|\bparkinson"),
        ("L23N1238-", "essay", "page", r"(?i)vice versa|other way (a)?round"),
        ("L28N5262+", "news", "page", r"(?i)\bfail(ed|ing|s)? to\b|\bunable to\b|\bcould ?n[o']?t\b|\bbarely\b"),
        ("L11N5238+", "code", "arm1", None),
        ("L27N2924-", "code", "arm1", None),
        ("L08N3398-", "code", "arm1", None),
        ("L12N7367-", "essay", "best", r"(?i)\buv\b|\bunwrap|\bunroll|\bflatten|\bflat\b|cut open|\bsurface|\bplane\b|\bfold"),
        ("L09N1795+", "essay", "best", r"(?i)cross-?tab|\bexploratory|\bconfirmatory|\bdescriptive|\bstatistic|"
                                       r"factor analysis|\bvariance|\bregression"),
        ("L07N4960-", "news", "best", r"(?i)\bmet with|\battended|in attendance|\bvisited|\bminister|\bpresident|"
                                      r"\bpremier|\bdelegation|\bceremon|\bdignitar"),
        ("L26N4718-", "email", "best", r"(?i)the information|personal data|e-?mail address|\bprivacy|data protection|"
                                       r"\bterms\b|you provide"),
        ("L31N3418+", "email", "best", r"(?i)\bpropert|\brental|\blandlord|\btenant|real estate|\binvestment"),
        ("L34N2004-", "news", "best", r"(?i)\bdevelopment|\bsurprising|\bshocking|\bstunning|\bpromising|turn of events"),
        ("L10N8041-", "essay", "best", r"(?i)[一-鿿぀-ヿ가-힯]|\buniversity|\bdepartment|"
                                       r"\bdistrict|\bprefecture"),
        ("L24N6912-", "essay", "best", r"(?i)[Ѐ-ӿ֐-ۿऀ-ॿ぀-ヿ一-鿿]|"
                                       r"\btranslat|\blatin\b|original title"),
        ("L24N3091+", "essay", "best", r"(?i)\bhow to\b|\bable to\b|in order to|\bwhen you\b|\bsteps? to\b"),
        ("L29N9473-", "story", "best", r"(?i)\bgo(es)? to\b|\bwent to\b|\bhead(ed|ing)?\b|\bproceed|\brush(ed)?\b|"
                                       r"\bturn(ed)? to\b|\btowards?\b|\bnavigat"),
        ("L31N5699+", "essay", "best", r"(?i)\bheld\b|\bhold(s|ing)?\b|\bbelief|\bviews?\b|\bopinion"),
    ],
    "G4": [
        ("L11N1220+", "news", "page", r"(?i)forward-looking|undue reliance|no assurance|\badversely|uncertaint|"
                                      r"safe harbou?r|risk factor"),
        ("L15N9089+", "email", "page", r"(?i)\bbusiness|\bcooperation|\bpartner|\benterprise|\bcompany"),
        ("L28N5924+", "news", "page", r"(?i)\bindia"),
        ("L07N10202-", "code", "page+arm1", None),
        ("L07N10202+", "news", "page", r"(?i)\bclass of\b|\bcohort|\bbatch\b|\bincoming\b|\binaugural|\bgraduat"),
        ("L06N599+", "code", "arm1", None),
        ("L08N7823+", "code", "arm1", None),
        ("L18N3736-", "essay", "best", r"(?i)in my opinion|in my view|in our view|\bi think\b|in particular"),
        ("L15N2237-", "news", "best", r"(?i)\broyal|\bprince|\bqueen\b|\bking\b|\bmeghan|\bharry\b|\bdiana\b|\bpalace|"
                                      r"tell-all|\bmemoir|book deal|\bbiograph"),
        ("L04N7359+", "news", "best", r"(?i)\bamateur"),
        ("L03N10205+", "story", "best", r"(?i)\blet (it|them|him|her|me|us|you|the \w+) go\b|\blet go\b"),
        ("L16N2951-", "email", "best", r"(?i)\bcontainer|\bpackag|\bresidue|safety data|\bml\b|fl\.? oz|\boz\b|\bproduct"),
        ("L24N1919+", "news", "best", r"(?i)\bplates?\b|\blicen[cs]e|\bregistration|\bvehicle"),
        ("L27N10083-", "story", "best", r"(?i)\bwrit|\bwrote\b|\boverwrit|\brewrit"),
        ("L21N6835-", "essay", "best", r"(?i)\bbefore\b|\bprior to\b|\bpublished|\bconstructed|\bbuilt\b|\bexisted"),
        ("L04N8650+", "email", "best", r"(?i)\bjerseys?\b|\bshirt|t-shirt|\bsweater|\bjumper|\bapparel|\bcamiseta"),
        ("L04N4329+", "story", "best", r"(?i)\bbikini|\bswimwear|\bswimsuit|\bbeach"),
        ("L26N9472-", "essay", "best", r"(?i)\b(us|you|those) \w+ (folks?|people|types|majors|owners)\b|\bfolks?\b"),
        ("L14N6393+", "news", "best", r"(?i)\bshooting|\bshot\b|\bguns?\b|\bgunfire|\bfirearm|ar-15|\brifle|semiautomatic"),
        ("L01N4035-", "news", "best", r"(?i)\bamenit|\bplanning|\b[AM]\d{1,4}\b|\bcouncil|\bpostcode"),
        ("L15N2237+", "news", "best", r"(?i)\banalyst|\brating|price target|target price|\bupgrade|\bdowngrade|"
                                      r"\bconsensus|\boutperform"),
    ],
    "O7": [
        ("L26N6019+", "news", "page", r"(?i)\bbroadcast|\btelevis|live[- ]?stream|\bspectator|\bstadium|press box|"
                                      r"\bterrace|\bcourt\b|\bgames?\b"),
        ("L07N7004-", "essay", "page", r"(?i)medical school|\bmedicine\b|\bmbbs\b|\bjournal|\bresidency|\bclinical|"
                                       r"\bphysician|\bcmaj\b"),
        ("L03N7540+", "news", "page", r"\b[A-Z][a-z]+ [A-Z][a-z]{2,}\b"),
    ],
}

# Code rule (frozen): a completion is code when it has a closing fence (a newline followed by ```; the opening fence
# is the prompt's) or at least CODE_MIN_PUNCT characters of CODE_PUNCT in the decoded text of its first CODE_HEAD
# tokens.
CODE_PUNCT = frozenset("{}[]()<>;=")
CODE_MIN_PUNCT = 8
CODE_HEAD = 64
CODE_SHARE_MIN = 0.80          # d* = the smallest positive dose whose push samples are at least this share code
GAP_BAR = 0.15                 # registered: push minus control presence at d* >= GAP_BAR
ROOM_MAX = 0.85                # control presence at d* above this leaves no room for the bar: not scorable
CODE_POS_MIN, CODE_NEG_MAX = 0.80, 0.20      # code-rule controls on dose-0 samples (code prompts / prose prompts)
COS_MIN, RATIO_BAND = 0.95, (0.9, 1.1)       # realized change at decode step 1 (G-gen bounds), median per cell
N_BOOT = 10_000
SHOW_TOKENS = 60
REPEAT_CUT = 0.5               # distinct-bigram ratio below this flags a loop (neuron_rw_analysis.REPEAT_CUT)


def parse_pole(p: str) -> tuple:
    m = re.fullmatch(r"L(\d+)N(\d+)([+-])", p)
    assert m, f"bad pole id {p!r}"
    return int(m[1]), int(m[2]), 1 if m[3] == "+" else -1


def draw_control(model_key: str, pole: str) -> str:
    layer, neuron, sign = parse_pole(pole)
    rng = np.random.default_rng([SEED, layer, neuron])
    c = neuron
    while c == neuron:
        c = int(rng.integers(0, D_MLP[model_key]))
    return f"L{layer:02d}N{c}{'+' if sign == 1 else '-'}"


def prompt_index(domain: str, j: int) -> int:
    """The seed's prompt index: fixed across models, 2 * domain position + j."""
    return 2 * DOMAINS.index(domain) + j


def directions(model_key: str) -> list:
    """Plan order: the model's poles, then the controls of its code neurons (in POLES order)."""
    out = []
    for pole, dom, src, theme in POLES[model_key]:
        assert dom in DOMAINS, dom
        c = CODE.get(pole)
        if dom == "code":
            assert c is not None and c["model"] == model_key, f"{pole}: code pole without a CODE entry"
        out.append({"pole": pole, "role": "pole", "domain": dom, "source": src, "target": None,
                    "theme": c["regex"] if c else theme})
    for pole, dom, _, _ in POLES[model_key]:
        if dom == "code":
            out.append({"pole": CODE[pole]["ctrl"], "role": "control", "domain": "code", "source": "control",
                        "target": pole, "theme": CODE[pole]["regex"]})
    ids = [d["pole"] for d in out]
    assert len(ids) == len(set(ids)), f"duplicate pole in {model_key}"
    return out


def is_code(completion: str, head_text: str) -> bool:
    return "\n```" in completion or sum(ch in CODE_PUNCT for ch in head_text) >= CODE_MIN_PUNCT
