"""ONE-new-token sign task (user's design): the ONLY new vocab item is '-'.
Negative numbers are COMPOSITIONAL: two tokens '-' 'MAG', with MAG drawn from the SAME range as
positive values (20..99) — so the only signal of negativity is the preceding '-' token. A single
256-dim embedding cannot memorize per-number answers; detecting negatives requires relational
binding ('-' followed by a number), i.e. genuine circuit work.
Query: 'first - number' (reuses '-' as the query word: still one new token total).
"""
import numpy as np
import retrieval_data as D

VLO, VHI = D.VAL_LO, D.VAL_HI          # 20..99 for BOTH positive values and negative magnitudes

def build_vocab_sign(base_vocab):
    assert "-" not in base_vocab
    return list(base_vocab) + ["-"]     # exactly ONE new token

def make_sign_list(rng, n, p_neg=0.35):
    """elements: objects, positive ints, or negative ints (rendered later as '-','MAG').
    Rejection-sample until the list has >=1 negative AND >=1 object (no in-place patching —
    an insert can silently overwrite the only member of the other class)."""
    while True:
        els = []
        for _ in range(n):
            r = rng.rand()
            if r < 0.45:
                els.append(D.OBJECTS[rng.randint(len(D.OBJECTS))])
            elif rng.rand() < p_neg:
                els.append(-int(rng.randint(VLO, VHI + 1)))
            else:
                els.append(int(rng.randint(VLO, VHI + 1)))
        if any(isinstance(e, int) and e < 0 for e in els) and any(isinstance(e, str) for e in els):
            return els

def make_sign_example(rng):
    n = rng.randint(5, 10)
    els = make_sign_list(rng, n)
    negs = [(i, e) for i, e in enumerate(els) if isinstance(e, int) and e < 0]
    pos, val = negs[0]
    return dict(list=els, qtype="Tsign", level=3, query=["first", "-", "number"],
                answer=val, pos=pos + 1)

def sign_tokens(ex):
    """(tokens, prompt_len) in the standard format; negatives expand to ['-','MAG']."""
    toks = []
    for e in ex["list"]:
        if isinstance(e, int) and e < 0: toks += ["-", str(-e)]
        else: toks.append(str(e))
    toks += ["|", "Q"] + list(ex["query"]) + ["="]
    pl = len(toks)
    mag = str(-ex["answer"])
    toks += ["scan", "for", "-", "numbers", ";",
             "first", "-", "->", "pos", str(ex["pos"]), "=", "-", mag, ";",
             "A", "-", mag, "<eos>"]
    return toks, pl


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    base = D.build_vocab(0, 300)
    v = build_vocab_sign(base)
    print(f"vocab {len(base)} -> {len(v)}  (ONE new token: '-')")
    tok = D.Tokenizer(v)
    for _ in range(3):
        ex = make_sign_example(rng)
        toks, pl = sign_tokens(ex)
        tok.enc(toks)
        print(f"\nlist: {ex['list']}")
        print(f"tokens: {' '.join(toks)}")
        print(f"  answer entity: {ex['answer']} at pos {ex['pos']}  (tokenizes OK)")
