"""
Stage 2 High-Quality Reasoning & In-Context Rule-Invention Corpus
=================================================================
Generates clean, structured problems across 3 domains where the Language Model
must REASON and INVENT rules in-context (with ZERO pre-labeled box hints!):

  1. In-Context Mystery Rule Invention:
     Shows 2 worked input->output examples of a hidden compositional machine
     (combining shifts, swaps, or conditional filters), then gives an unseen input
     where the model must deduce the hidden rule inside `<think>...</think>` and
     predict `<ans>...</ans>`.
  2. Stateful Python Code Execution (Variables, Clocks, Swaps, Branches):
     Multi-statement Python traces with distractor comments -> predict final values!
  3. Multi-Step Deductive Entity & Dial Stories:
     Natural-language state & conditional trading stories -> deduce final owner/value!
"""

import random


FILLER_NOTES = [
    "note ok.",
    "check pass.",
    "log idle.",
    "keep state.",
    "wait step.",
    "noop line.",
]


def _make_mystery_rule_example(rng: random.Random) -> str:
    """Domain 1: Model sees 2 examples of an UNSEEN rule and must induce it on a 3rd input!"""
    rule_type = rng.choice(["swap_ends_shift", "even_swap_pair", "clock_add_pair", "reverse_plus"])

    def apply_rule(arr: list[int]) -> tuple[list[int], str]:
        a = list(arr)
        if rule_type == "swap_ends_shift":
            a[0], a[2] = a[2], a[0]
            a = [(x + 1) % 8 for x in a]
            hint = "rule swap ends then +1 mod 8"
        elif rule_type == "even_swap_pair":
            if a[0] % 2 == 0:
                a[1], a[2] = a[2], a[1]
                hint = f"rule first={a[0]} even -> swap mid,last"
            else:
                hint = f"rule first={a[0]} odd -> keep order"
        elif rule_type == "clock_add_pair":
            step = a[0]
            a[1] = (a[1] + step) % 12
            a[2] = (a[2] + step) % 12
            hint = f"rule tick +{step} mod 12"
        else:
            a = [a[2], a[1], a[0]]
            a[0] = (a[0] + 2) % 8
            hint = "rule reverse then first +2 mod 8"
        return a, hint

    ex1_in = [rng.randint(0, 7) for _ in range(3)]
    ex2_in = [rng.randint(0, 7) for _ in range(3)]
    test_in = [rng.randint(0, 7) for _ in range(3)]

    ex1_out, _ = apply_rule(ex1_in)
    ex2_out, _ = apply_rule(ex2_in)
    test_out, thought = apply_rule(test_in)

    fmt = lambda v: f"{v[0]},{v[1]},{v[2]}"
    return (
        f"induce: [{fmt(ex1_in)}] -> [{fmt(ex1_out)}] ; "
        f"[{fmt(ex2_in)}] -> [{fmt(ex2_out)}] ; "
        f"[{fmt(test_in)}] -> <think> {thought} </think> <ans> {fmt(test_out)} </ans>"
    )


def _make_python_trace_example(rng: random.Random, n_ops: int) -> str:
    """Domain 2: Stateful Python snippet with clock math, swaps, conditionals, and comments."""
    a = rng.randint(0, 7)
    b = rng.randint(0, 7)
    c = rng.randint(0, 7)
    clk = rng.randint(0, 11)

    lines = [f"py: set a= {a} ; set b= {b} ; set c= {c} ; set clk= {clk}"]
    for _ in range(n_ops):
        op = rng.choices(
            ["filler", "tick", "swap_ab", "swap_bc", "if_clk_swap", "write"],
            weights=[0.35, 0.20, 0.12, 0.12, 0.13, 0.08],
        )[0]
        if op == "filler":
            lines.append(rng.choice(FILLER_NOTES))
        elif op == "tick":
            dh = rng.choice([1, 2, 5, 11])
            clk = (clk + dh) % 12
            lbl = "-1" if dh == 11 else f"+{dh}"
            lines.append(f"tick clk {lbl}")
        elif op == "swap_ab":
            a, b = b, a
            lines.append("swap a,b")
        elif op == "swap_bc":
            b, c = c, b
            lines.append("swap b,c")
        elif op == "if_clk_swap":
            if clk % 2 == 0:
                a, b = b, a
            lines.append("if clk even then swap a,b")
        elif op == "write":
            var = rng.choice(["a", "b", "c"])
            val = rng.randint(0, 7)
            if var == "a":
                a = val
            elif var == "b":
                b = val
            else:
                c = val
            lines.append(f"set {var}= {val}")

    parity = "even" if clk % 2 == 0 else "odd"
    program = " ; ".join(lines)
    return f"{program} -> <think> clk={clk} {parity} -> a={a},b={b},c={c} </think> <ans> {a},{b},{c},{clk} </ans>"


def _make_deductive_story_example(rng: random.Random, n_events: int) -> str:
    """Domain 3: Natural-language entity & dial deduction puzzle."""
    vals = [rng.randint(1, 7) for _ in range(3)]  # Ava, Ben, Zoe
    dial = rng.randint(0, 11)
    story = [f"story: set Ava= {vals[0]} , set Ben= {vals[1]} , set Zoe= {vals[2]} , set dial= {dial}."]

    for _ in range(n_events):
        ev = rng.choices(["chatter", "turn", "trade01", "trade12", "cond_trade"], weights=[0.35, 0.20, 0.15, 0.15, 0.15])[0]
        if ev == "chatter":
            story.append(rng.choice(FILLER_NOTES))
        elif ev == "turn":
            step = rng.choice([1, 2, 5])
            dial = (dial + step) % 12
            story.append(f"tick dial +{step}.")
        elif ev == "trade01":
            vals[0], vals[1] = vals[1], vals[0]
            story.append("swap Ava,Ben.")
        elif ev == "trade12":
            vals[1], vals[2] = vals[2], vals[1]
            story.append("swap Ben,Zoe.")
        else:
            if vals[0] % 2 == 0:
                vals[1], vals[2] = vals[2], vals[1]
            story.append("if Ava even then swap Ben,Zoe.")

    text = " ".join(story)
    return (
        f"{text} query -> <think> dial={dial} Ava={vals[0]} -> Ben={vals[1]} Zoe={vals[2]} </think> "
        f"<ans> {vals[0]},{vals[1]},{vals[2]},{dial} </ans>"
    )


def build_reasoning_splits(seed: int = 42, n_train: int = 1800, n_val: int = 200, n_long_test: int = 200):
    """
    Strictly separates train, in-distribution test, and long-horizon extrapolation test splits!
    """
    rng = random.Random(seed)

    def sample_doc(long_mode: bool = False, is_train: bool = False) -> str:
        if long_mode:
            # 3.5x more state transitions & distractors than training!
            if rng.random() < 0.55:
                return _make_python_trace_example(rng, n_ops=rng.randint(18, 26))
            else:
                return _make_deductive_story_example(rng, n_events=rng.randint(16, 24))
        kind = rng.random()
        if kind < 0.34:
            return _make_mystery_rule_example(rng)
        elif kind < 0.70:
            n_o = rng.randint(1, 8) if is_train else rng.randint(4, 8)
            return _make_python_trace_example(rng, n_ops=n_o)
        else:
            n_e = rng.randint(1, 7) if is_train else rng.randint(4, 7)
            return _make_deductive_story_example(rng, n_events=n_e)

    train_docs = [sample_doc(False, True) for _ in range(n_train)]
    val_docs = [sample_doc(False, False) for _ in range(n_val)]
    long_docs = [sample_doc(True, False) for _ in range(n_long_test)]
    return train_docs, val_docs, long_docs
