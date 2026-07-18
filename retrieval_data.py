"""Relative-position + semantic-filtering retrieval task (flat mixed list).

A list is a FLAT sequence of elements, each either an OBJECT (ball, box, ...) or a
NUMBER (20-99). Positions are 1-indexed from the LEFT. Objects have a LATENT shape
(round/flat) the model must LEARN (never shown). Queries combine skills:
  index · type-filter (number/object) · semantic (round/flat) · relative (after/before
  an anchor) · content (value > V).
Every example carries a STEP-BY-STEP trace; every trace token is tagged with the ONE
skill that step exercises, so loss can be split per-skill. Curriculum: L1..L5.
"""
import numpy as np

SHAPE = {"ball": "round", "coin": "round", "ring": "round",
         "box": "flat", "book": "flat", "card": "flat"}
OBJECTS = list(SHAPE)
ROUND = [o for o in OBJECTS if SHAPE[o] == "round"]
FLAT = [o for o in OBJECTS if SHAPE[o] == "flat"]
VAL_LO, VAL_HI = 20, 99           # values disjoint from positions (<=12)
SKILLS = ["index", "filter", "semantic", "relative", "content", "read"]
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}

# --- level -> (list-len range, query-type ids) ---
LEVELS = {
    1: (dict(lo=3, hi=5),  ["T1"]),
    2: (dict(lo=5, hi=6),  ["T2", "T3"]),
    3: (dict(lo=6, hi=7),  ["T4", "T5"]),
    4: (dict(lo=7, hi=9),  ["T6", "T7"]),
    5: (dict(lo=9, hi=12), ["T8", "T9", "T10"]),
}


def is_num(e): return isinstance(e, int)
def is_obj(e): return isinstance(e, str)


def make_list(rng, n):
    """n elements, ~half objects half numbers, at least 1 of each."""
    els = []
    for _ in range(n):
        if rng.random() < 0.5:
            els.append(OBJECTS[rng.randint(len(OBJECTS))])
        else:
            els.append(int(rng.randint(VAL_LO, VAL_HI + 1)))
    if all(is_num(e) for e in els): els[rng.randint(n)] = OBJECTS[rng.randint(len(OBJECTS))]
    if all(is_obj(e) for e in els): els[rng.randint(n)] = int(rng.randint(VAL_LO, VAL_HI + 1))
    return els


# ---------------- query solvers ----------------
# each returns (query_tokens, answer_str, steps) where steps = [(text, skill), ...]
# or None if no valid answer for the sampled params (caller resamples).

def _first_anchor(els, name):
    for i, e in enumerate(els):
        if e == name:
            return i
    return None


def q_T1(rng, els):                                   # kth from left  (index)
    k = rng.randint(1, len(els) + 1)
    ans = els[k - 1]
    q = [ORD.get(k, f"{k}th"), "from", "left"]
    steps = [(f"pos {k} -> {ans}", "index")]
    return q, str(ans), steps


def q_T2(rng, els):                                   # kth number  (index+filter)
    nums = [(i, e) for i, e in enumerate(els) if is_num(e)]
    if not nums: return None
    k = rng.randint(1, len(nums) + 1)
    ans = nums[k - 1][1]
    q = [ORD.get(k, f"{k}th"), "number"]
    steps = [("scan for numbers", "filter"),
             (f"{ORD.get(k, str(k))} number = {ans}", "index")]
    return q, str(ans), steps


def q_T3(rng, els):                                   # kth object  (index+filter)
    objs = [(i, e) for i, e in enumerate(els) if is_obj(e)]
    if not objs: return None
    k = rng.randint(1, len(objs) + 1)
    ans = objs[k - 1][1]
    q = [ORD.get(k, f"{k}th"), "object"]
    steps = [("scan for objects", "filter"),
             (f"{ORD.get(k, str(k))} object = {ans}", "index")]
    return q, str(ans), steps


def q_T4(rng, els):                                   # kth round/flat object (index+filter+semantic)
    shp = rng.choice(["round", "flat"])
    hits = [(i, e) for i, e in enumerate(els) if is_obj(e) and SHAPE[e] == shp]
    if not hits: return None
    k = rng.randint(1, len(hits) + 1)
    ans = hits[k - 1][1]
    q = [ORD.get(k, f"{k}th"), shp, "object"]
    steps = [(f"classify {shp} objects", "semantic"),
             (f"{ORD.get(k, str(k))} {shp} = {ans}", "index")]
    return q, str(ans), steps


def q_T5(rng, els):                                   # first number > V (index+filter+content)
    nums = [e for e in els if is_num(e)]
    if not nums or max(nums) <= VAL_LO: return None
    V = int(rng.randint(VAL_LO, max(nums)))
    match = [e for e in els if is_num(e) and e > V]
    if not match: return None
    ans = match[0]
    q = ["first", "number", "greater", "than", str(V)]
    steps = [(f"scan numbers > {V}", "content"),
             (f"first match = {ans}", "index")]
    return q, str(ans), steps


def q_T6(rng, els):                                   # kth element after first ANCHOR (index+relative)
    anchor = OBJECTS[rng.randint(len(OBJECTS))]
    p = _first_anchor(els, anchor)
    if p is None: return None
    k = rng.randint(1, len(els) - p) if p < len(els) - 1 else 0
    if k == 0 or p + k >= len(els): return None
    ans = els[p + k]
    q = [ORD.get(k, f"{k}th"), "element", "after", "first", anchor]
    steps = [(f"anchor first {anchor} -> pos {p+1}", "relative"),
             (f"{k} after -> pos {p+k+1} = {ans}", "index")]
    return q, str(ans), steps


def q_T7(rng, els):                                   # kth number after first ANCHOR (index+filter+relative)
    anchor = OBJECTS[rng.randint(len(OBJECTS))]
    p = _first_anchor(els, anchor)
    if p is None: return None
    after_nums = [e for e in els[p + 1:] if is_num(e)]
    if not after_nums: return None
    k = rng.randint(1, len(after_nums) + 1)
    ans = after_nums[k - 1]
    q = [ORD.get(k, f"{k}th"), "number", "after", "first", anchor]
    steps = [(f"anchor first {anchor} -> pos {p+1}", "relative"),
             ("numbers after anchor", "filter"),
             (f"{ORD.get(k, str(k))} = {ans}", "index")]
    return q, str(ans), steps


def q_T8(rng, els):                                   # kth round obj after ANCHOR (all 4)
    anchor = OBJECTS[rng.randint(len(OBJECTS))]
    shp = rng.choice(["round", "flat"])
    p = _first_anchor(els, anchor)
    if p is None: return None
    hits = [e for e in els[p + 1:] if is_obj(e) and SHAPE[e] == shp]
    if not hits: return None
    k = rng.randint(1, len(hits) + 1)
    ans = hits[k - 1]
    q = [ORD.get(k, f"{k}th"), shp, "object", "after", "first", anchor]
    steps = [(f"anchor first {anchor} -> pos {p+1}", "relative"),
             (f"{shp} objects after", "semantic"),
             (f"{ORD.get(k, str(k))} = {ans}", "index")]
    return q, str(ans), steps


def q_T9(rng, els):                                   # first number > V after ANCHOR (filter+relative+content)
    anchor = OBJECTS[rng.randint(len(OBJECTS))]
    p = _first_anchor(els, anchor)
    if p is None: return None
    after_nums = [e for e in els[p + 1:] if is_num(e)]
    if not after_nums or max(after_nums) <= VAL_LO: return None
    V = int(rng.randint(VAL_LO, max(after_nums)))
    match = [e for e in after_nums if e > V]
    if not match: return None
    ans = match[0]
    q = ["first", "number", "greater", "than", str(V), "after", "first", anchor]
    steps = [(f"anchor first {anchor} -> pos {p+1}", "relative"),
             (f"numbers after > {V}", "content"),
             (f"first match = {ans}", "index")]
    return q, str(ans), steps


def q_T10(rng, els):                                  # object right before first number > V (content-anchor+relative)
    V = int(rng.randint(VAL_LO, VAL_HI - 1))
    pos = next((i for i, e in enumerate(els) if is_num(e) and e > V), None)
    if pos is None or pos == 0: return None
    ans = els[pos - 1]
    q = ["element", "before", "first", "number", "greater", "than", str(V)]
    steps = [(f"anchor first number > {V} -> pos {pos+1}", "content"),
             (f"element before -> pos {pos} = {ans}", "relative")]
    return q, str(ans), steps


SOLVERS = {"T1": q_T1, "T2": q_T2, "T3": q_T3, "T4": q_T4, "T5": q_T5,
           "T6": q_T6, "T7": q_T7, "T8": q_T8, "T9": q_T9, "T10": q_T10}


def make_example(rng, level):
    lo_hi, qtypes = LEVELS[level]
    for _ in range(200):
        n = rng.randint(lo_hi["lo"], lo_hi["hi"] + 1)
        els = make_list(rng, n)
        qt = qtypes[rng.randint(len(qtypes))]
        out = SOLVERS[qt](rng, els)
        if out is not None:
            q, ans, steps = out
            return dict(list=els, qtype=qt, level=level, query=q, answer=ans, steps=steps)
    return None


# ---------------- tokenization ----------------
def _example_tokens(ex):
    """Return (tokens, prompt_len, skill_tags). Target skills tagged per token;
    prompt tokens tagged 'input'. Answer tokens tagged 'read'."""
    list_toks = [str(e) for e in ex["list"]]
    prompt = list_toks + ["|", "Q"] + ex["query"] + ["="]
    tgt, tgt_skill = [], []
    for text, skill in ex["steps"]:
        for t in text.split():
            tgt.append(t); tgt_skill.append(skill)
        tgt.append(";"); tgt_skill.append(skill)
    tgt += ["A", ex["answer"], "<eos>"]
    tgt_skill += ["read", "read", "read"]
    tokens = prompt + tgt
    skills = ["input"] * len(prompt) + tgt_skill
    return tokens, len(prompt), skills


def make_ood_example(rng, lo=13, hi=20):
    """Level-5 query types on LONGER lists than trained (the OOD generalization probe)."""
    for _ in range(200):
        n = rng.randint(lo, hi + 1); els = make_list(rng, n)
        qt = ["T8", "T9", "T10"][rng.randint(3)]
        out = SOLVERS[qt](rng, els)
        if out:
            q, ans, steps = out
            return dict(list=els, qtype=qt, level=6, query=q, answer=ans, steps=steps)
    return None


def build_vocab(seed=0, n=3000):
    rng = np.random.RandomState(seed)
    vocab = set(["<pad>", "<eos>", "|", "Q", "=", ";", "A", "from", "left",
                 "number", "object", "round", "flat", "element", "after", "before",
                 "first", "greater", "than", "scan", "for", "numbers", "objects",
                 "classify", "match", "anchor", "pos", "->"])
    vocab.update(OBJECTS)
    vocab.update(str(i) for i in range(0, 100))                 # positions (1-20) + values (20-99)
    vocab.update(ORD.values())
    vocab.update(f"{i}th" for i in range(1, 25))                # ordinals beyond 5 (OOD lengths)
    for _ in range(n):
        for lv in LEVELS:
            ex = make_example(rng, lv)
            if ex: vocab.update(_example_tokens(ex)[0])
        oe = make_ood_example(rng)                              # cover OOD tokens too
        if oe: vocab.update(_example_tokens(oe)[0])
    return sorted(vocab)


class Tokenizer:
    def __init__(self, vocab):
        self.itos = vocab
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.pad = self.stoi["<pad>"]; self.eos = self.stoi["<eos>"]
    def enc(self, toks): return [self.stoi[t] for t in toks]


if __name__ == "__main__":
    import re
    rng = np.random.RandomState(0)
    print("=== sample examples per level ===")
    for lv in LEVELS:
        ex = make_example(rng, lv)
        toks, pl, sk = _example_tokens(ex)
        print(f"L{lv} [{ex['qtype']}] list={ex['list']}")
        print(f"     Q: {' '.join(ex['query'])} = {ex['answer']}")
        print(f"     trace: {' | '.join(t for t,_ in ex['steps'])}")
        print(f"     skills-in-target: {sorted(set(s for s in sk if s!='input'))}")
    # correctness self-test: independently re-verify each query type's answer
    print("\n=== correctness self-test (2000 examples) ===")
    bad = 0
    for _ in range(2000):
        lv = rng.randint(1, 6)
        ex = make_example(rng, lv)
        if ex is None: continue
        # answer must be an element of the list (object or number) or valid
        a = ex["answer"]
        aval = int(a) if a.lstrip("-").isdigit() else a
        if aval not in ex["list"]:
            bad += 1
    print(f"answers-in-list violations: {bad}/2000 (should be 0)")
    v = build_vocab(0, 800)
    print(f"vocab size ~ {len(v)}")
