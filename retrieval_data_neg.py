"""Extension: negative numbers + a new 'sign' skill (find the first NEGATIVE number).
Reuses the comparison machinery (a number's sign = compare-to-0), so we can test whether a new
skill installs by REUSING the existing 'content' (>V) circuit. New negative-number tokens are added
to the vocab; the 42 ternary matrices are vocab-independent so they can be fine-tuned from the ckpt.
"""
import numpy as np
import retrieval_data as D

NEG_LO, NEG_HI = -30, -1                      # negative-number range (disjoint tokens from positives 20..99)
ORD = D.ORD

def make_list_neg(rng, n, p_neg=0.4):
    """flat list, ~half objects/half numbers; numbers are negative w.p. p_neg."""
    els = []
    for _ in range(n):
        if rng.rand() < 0.5:
            els.append(D.OBJECTS[rng.randint(len(D.OBJECTS))])
        elif rng.rand() < p_neg:
            els.append(int(rng.randint(NEG_LO, NEG_HI + 1)))
        else:
            els.append(int(rng.randint(D.VAL_LO, D.VAL_HI + 1)))
    # ensure >=1 object and >=1 number
    if not any(isinstance(e, str) for e in els): els[rng.randint(n)] = D.OBJECTS[rng.randint(len(D.OBJECTS))]
    if not any(isinstance(e, int) for e in els): els[rng.randint(n)] = int(rng.randint(D.VAL_LO, D.VAL_HI + 1))
    return els

def q_sign(rng, els):
    """'first negative number' -> the first number < 0. Skill = 'sign' (reuses comparison)."""
    negs = [(i, e) for i, e in enumerate(els) if isinstance(e, int) and e < 0]
    if not negs:
        return None
    pos, val = negs[0]
    query = ["first", "negative", "number"]
    steps = [("scan for negative numbers", "sign"),
             (f"first negative -> pos {pos+1} = {val}", "sign")]
    return query, str(val), steps

def make_neg_example(rng):
    for _ in range(200):
        n = rng.randint(5, 10)
        els = make_list_neg(rng, n)
        out = q_sign(rng, els)
        if out:
            q, ans, steps = out
            return dict(list=els, qtype="Tsign", level=3, query=q, answer=ans, steps=steps)
    return None

def build_vocab_neg(base_vocab):
    """extend an existing vocab APPEND-ONLY: original token IDs must not shift, or a model
    whose embedding rows are copied by old ID is scrambled (orig-task acc -> 0)."""
    extra = set(str(v) for v in range(NEG_LO, NEG_HI + 1))   # -30..-1
    extra |= {"negative", "positive"}
    return list(base_vocab) + sorted(t for t in extra if t not in set(base_vocab))


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    base = D.build_vocab(0, 500)
    vneg = build_vocab_neg(base)
    print(f"base vocab {len(base)} -> with negatives {len(vneg)}  (+{len(vneg)-len(base)} new tokens)")
    tok = D.Tokenizer(vneg)
    print("\n--- 3 sign-task examples ---")
    for _ in range(3):
        ex = make_neg_example(rng)
        print(f"list: {ex['list']}")
        print(f"  Q: {' '.join(ex['query'])}  -> A: {ex['answer']}")
        for t, s in ex["steps"]: print(f"     [{s}] {t}")
        toks, pl, sk = D._example_tokens(ex)
        _ = tok.enc(toks)                       # verify all tokens in vocab
        print(f"  tokenizes OK ({len(toks)} tokens)")
    # verify original examples still tokenize with the extended vocab
    oe = D.make_example(rng, 5); toks, pl, sk = D._example_tokens(oe); tok.enc(toks)
    print("\noriginal-task example still tokenizes with extended vocab: OK")
