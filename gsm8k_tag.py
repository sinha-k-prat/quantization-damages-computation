"""GSM8K token tagger for the real-model quantization study.
Labels every token of a solution as:
  computation  - a number the model must CALCULATE (right of '=' or after '####')
  copy         - a number copied verbatim from the question (retrieval)
  language     - everything else (fluent text baseline)
Also exposes the gold final answer (after '####') for task-success (exact-match) scoring.
The '<<a op b=c>>' calculator spans are stripped from model input but used to sanity-check labels.
"""
import re

NUM = r'-?\$?\d[\d,]*\.?\d*'

def gold_answer(answer_raw):
    m = re.search(r'####\s*(' + NUM + r')', answer_raw)
    return m.group(1).replace(",", "").replace("$", "").strip() if m else None

def clean_solution(answer_raw):
    """what the model actually sees: <<...>> calculator hints removed."""
    return re.sub(r'<<[^>]*?>>', '', answer_raw)

def char_labels(question, answer_raw):
    """return (clean_text, [(start,end,label,value)]) labeling each number span."""
    given = set(re.findall(r'\d[\d,]*\.?\d*', question))
    given = {g.replace(",", "") for g in given}
    results = {m.group(1).replace(",", "") for m in re.finditer(r'<<[^>]*?=\s*(' + NUM + r')\s*>>', answer_raw)}
    clean = clean_solution(answer_raw)
    spans = []
    for m in re.finditer(NUM, clean):
        s, e = m.span(); val = m.group().replace(",", "").replace("$", "")
        pre = clean[max(0, s - 8):s]
        if re.search(r'(####|=)\s*\$?\s*$', pre):
            lab = "computation"                    # produced as a result
        elif val in results:
            lab = "computation"                    # a computed value reused
        elif val in given:
            lab = "copy"                           # taken from the question
        else:
            lab = "copy"                           # unlabeled number -> treat as non-computation
        spans.append((s, e, lab, val))
    return clean, spans

def tag_tokens(question, answer_raw, tokenizer):
    """tokenize the clean solution; return [(token_str, label)] with char-offset alignment."""
    clean, spans = char_labels(question, answer_raw)
    enc = tokenizer(clean, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    out = []
    for tid, (a, b) in zip(ids, offs):
        lab = "language"
        if b > a:                                  # real span
            for (s, e, l, v) in spans:
                if a < e and b > s:                # token overlaps a number span
                    lab = l; break
        out.append((tokenizer.decode([tid]), lab, tid))
    return clean, out


if __name__ == "__main__":
    # a couple of REAL gsm8k items (fallback if no dataset download)
    EX = [
        ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
         "How many clips did Natalia sell altogether in April and May?",
         "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
         "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"),
        ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
         "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n"
         "Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"),
    ]
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test[:3]")
        EX = [(d["question"], d["answer"]) for d in ds]
        print("loaded real GSM8K from datasets\n")
    except Exception as e:
        print("dataset download failed (%s); using hardcoded real examples\n" % type(e).__name__)

    tok = None
    for name in ["Qwen/Qwen2.5-0.5B", "gpt2"]:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name); print("tokenizer:", name, "\n"); break
        except Exception as e:
            print("tokenizer", name, "unavailable:", type(e).__name__)
    if tok is None:
        raise SystemExit("no tokenizer available offline; will use Qwen tokenizer on the GPU box")

    for q, a in EX[:2]:
        print("Q:", q[:90], "...")
        print("gold answer (#### , for success EM):", gold_answer(a))
        clean, toks = tag_tokens(q, a, tok)
        from collections import Counter
        print("token-type counts:", dict(Counter(l for _, l, _ in toks)))
        # show the computation + copy tokens explicitly
        comp = [t for t, l, _ in toks if l == "computation"]
        cop = [t for t, l, _ in toks if l == "copy"]
        print("COMPUTATION tokens:", comp)
        print("COPY tokens:", cop)
        print("-" * 70)
