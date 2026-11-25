"""
Stage 2 Runner: Training & Inspecting the Self-Inventing SuperBPE + CoBPE Reasoning LM
======================================================================================
1. Trains the Two-Stage SuperBPE + CoBPE Tokenizer from scratch on the training corpus
   and prints learned multi-word "superwords" & compression gains.
2. Trains two Language Models head-to-head on pure next-(base, modifier) prediction:
   - Baseline: Standard Gated-DeltaNet LM (same SuperBPE+CoBPE, no Wreath gears or TRM)
   - Proposed: Self-Inventing NCM-TRM Dual-Gear Wreath LM (ZERO box/clock labels!)
3. Evaluates:
   - Unseen Mystery-Rule Invention (`induce:`), Python Execution (`py:`), Deductive Stories
   - 3.5x Long-Horizon Zero-Shot Extrapolation in 8-Bit FP8 (C = 4)
   - Autonomous Circuit Discovery: inspects which tokens made the LM invent swaps & clock angles!
"""

import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp1_adaptive_locking import snap_to_exact_signed_perm
from stage2.reasoning_corpus import build_reasoning_splits
from stage2.self_reasoning_lm import SelfInventingWreathLM
from stage2.tokenizer_super_cobpe import SuperCoBPETokenizer


def collate_docs(docs: list[str], tok: SuperCoBPETokenizer):
    enc = [tok.encode(d) for d in docs]
    max_len = max(len(b) for b, _ in enc)
    B = len(docs)
    base_t = torch.zeros(B, max_len, dtype=torch.long)  # 0 == <PAD>
    mod_t = torch.zeros(B, max_len, dtype=torch.long)
    ans_mask = torch.zeros(B, max_len, dtype=torch.bool)

    think_start_id = tok.base_to_id["<think>"]
    eos_id = tok.base_to_id["<EOS>"]

    for i, (b_ids, m_ids) in enumerate(enc):
        L = len(b_ids)
        base_t[i, :L] = torch.tensor(b_ids, dtype=torch.long)
        mod_t[i, :L] = torch.tensor(m_ids, dtype=torch.long)
        in_reasoning = False
        for j, bid in enumerate(b_ids):
            if bid == think_start_id:
                in_reasoning = True
            if in_reasoning and bid != eos_id:
                ans_mask[i, j] = True
    return base_t, mod_t, ans_mask


def extract_answer_span(text: str) -> str:
    if "<ans>" in text and "</ans>" in text:
        return text.split("<ans>", 1)[1].split("</ans>", 1)[0].strip().replace(" ", "")
    return ""


def generate_autoregressive(
    model: SelfInventingWreathLM, prompt_b: list[int], prompt_m: list[int], tok: SuperCoBPETokenizer, max_new: int = 28, fp8: bool = False
) -> str:
    """True zero-hint autoregressive generation: model writes its own <think>...</think><ans>...</ans>!"""
    b_seq = list(prompt_b)
    m_seq = list(prompt_m)
    eos_id = tok.base_to_id["<EOS>"]
    ans_end_id = tok.base_to_id["</ans>"]

    with torch.no_grad():
        for _ in range(max_new):
            bt = torch.tensor([b_seq], dtype=torch.long)
            mt = torch.tensor([m_seq], dtype=torch.long)
            b_log, m_log, _ = model(bt, mt, target_base_ids=None, deadzone=0.05, fp8_eval=fp8)
            next_b = int(b_log[0, -1].argmax().item())
            # Condition modifier on chosen next_b
            next_m = int(m_log[0, -1].argmax().item())
            b_seq.append(next_b)
            m_seq.append(next_m)
            if next_b in (eos_id, ans_end_id):
                break
    return tok.decode(b_seq, m_seq)


def evaluate_lm_autoregressive(model: SelfInventingWreathLM, docs: list[str], tok: SuperCoBPETokenizer, fp8: bool = False):
    """
    Strictly honest evaluation: feeds ONLY the problem prompt up to `<think>`,
    and checks whether the model's own generated `<think>...<ans>...</ans>` gets the exact answer!
    """
    model.eval()
    think_id = tok.base_to_id["<think>"]
    exact_matches = 0
    char_matches = 0
    total_chars = 0
    domain_stats = {"induce:": [0, 0], "py:": [0, 0], "story:": [0, 0]}
    samples = []

    for idx, doc_str in enumerate(docs):
        b_ids, m_ids = tok.encode(doc_str)
        # Cut strictly right after `<think>` so ZERO reasoning or answer tokens are leaked!
        cut = b_ids.index(think_id) + 1
        gen_text = generate_autoregressive(model, b_ids[:cut], m_ids[:cut], tok, max_new=26, fp8=fp8)

        pred_ans = extract_answer_span(gen_text)
        true_ans = extract_answer_span(doc_str)

        is_exact = int(pred_ans == true_ans and len(true_ans) > 0)
        exact_matches += is_exact

        # Also measure per-value slot accuracy inside `a,b,c(,clk)`
        p_parts = pred_ans.split(",")
        t_parts = true_ans.split(",")
        for k_i, tv in enumerate(t_parts):
            total_chars += 1
            if k_i < len(p_parts) and p_parts[k_i] == tv:
                char_matches += 1

        for prefix in domain_stats:
            if doc_str.startswith(prefix):
                domain_stats[prefix][0] += is_exact
                domain_stats[prefix][1] += 1

        if idx < 3:
            samples.append((doc_str, gen_text, true_ans, pred_ans))

    slot_acc = (char_matches / max(1, total_chars)) * 100.0
    seq_acc = (exact_matches / max(1, len(docs))) * 100.0
    dom_pct = {k: (v[0] / max(1, v[1])) * 100.0 for k, v in domain_stats.items() if v[1] > 0}
    return slot_acc, seq_acc, dom_pct, samples


def inspect_self_invented_circuits(model: SelfInventingWreathLM, tok: SuperCoBPETokenizer):
    """Probes what gears & clock rotations the LM invented WITHOUT any supervision!"""
    probe_text = (
        "py: set a= 2 ; set b= 5 ; set c= 1 ; set clk= 4 ; "
        "note ok. ; tick clk +1 ; swap a,b ; if clk even then swap a,b -> <think>"
    )
    b_ids, m_ids = tok.encode(probe_text)
    bt = torch.tensor([b_ids])
    mt = torch.tensor([m_ids])

    model.eval()
    with torch.no_grad():
        _ = model(bt, mt, deadzone=0.05, fp8_eval=False)
        G_seq = model.last_G_swap[0]                   # [L, 6, 6]
        th_deg = torch.rad2deg(model.last_theta[0])    # [L, 4]

    eye6 = torch.eye(6)
    print("\n--- AUTONOMOUS CIRCUIT DISCOVERY: What Did the LM Invent on Its Own? ---")
    for pos, (bid, mid) in enumerate(zip(b_ids, m_ids)):
        piece = ("" if mid == 0 else tok.id_to_mod[mid]) + tok.id_to_base[bid]
        if any(kw in piece for kw in ["swap", "tick", "+1", "note", "even"]):
            G_snap = snap_to_exact_signed_perm(G_seq[pos : pos + 1])[0]
            swapped = not torch.equal(G_snap, eye6)
            max_ang = th_deg[pos].abs().max().item()
            print(
                f"  Supertoken [{pos:2d}] {repr(piece):18s} -> Swapper Active: {str(swapped):5s} | Peak Dial Rotation: {max_ang:5.1f}°"
            )


def run_stage2():
    print("=" * 92)
    print("STAGE 2: Self-Inventing SuperBPE + CoBPE Reasoning Language Model Benchmark")
    print("=" * 92)

    train_docs, val_docs, long_docs = build_reasoning_splits(seed=42, n_train=3200, n_val=60, n_long_test=60)

    tok = SuperCoBPETokenizer(subword_merges=85, superword_merges=65)
    stats = tok.train(train_docs)
    sample_superwords = [a + b for a, b in tok.stage2_merges[:10]]
    print(
        f"[SuperBPE + CoBPE] Base Vocab: {stats['base_vocab_size']} | Modifiers: {stats['modifier_vocab_size']} | "
        f"Token Compression vs Standard BPE: -{stats['compression_vs_bpe_pct']:.1f}% fewer tokens!"
    )
    print(f"  Sample Space-Bridging Superwords Learned: {sample_superwords[:8]}\n")

    # Pre-collate mini-batches for fast training
    batch_size = 32
    batches = []
    for idx in range(0, len(train_docs), batch_size):
        batches.append(collate_docs(train_docs[idx : idx + batch_size], tok))

    trained_ours = None
    for label, use_wreath in [
        ("1. Standard Gated-DeltaNet LM (SuperBPE+CoBPE Baseline)", False),
        ("2. Self-Inventing NCM-TRM Wreath LM (Ours, Autonomous!)", True),
    ]:
        torch.manual_seed(2026)
        model = SelfInventingWreathLM(
            base_vocab=stats["base_vocab_size"],
            mod_vocab=stats["modifier_vocab_size"],
            d_model=96,
            d_swap=8,
            d_v=36,
            n_dials=4,
            chunk_c=4,
            use_wreath_trm=use_wreath,
        )
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=1e-4)
        total_steps = 22 * len(batches)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=3e-4)

        recent_ans_acc = 0.10
        trig_step: int | None = None
        t0 = time.time()

        step = 0
        for _epoch in range(22):
            for b_ids, m_ids, a_mask in batches:
                step += 1
                model.train()
                if use_wreath:
                    if trig_step is None and recent_ans_acc >= 0.70:
                        trig_step = step
                    if trig_step is None:
                        lam = 0.0
                        ns = 0.025 if recent_ans_acc < 0.50 else 0.012
                    else:
                        prog = min(1.0, (step - trig_step) / 200.0)
                        lam = 0.25 * prog
                        ns = 0.005 * (1.0 - 0.8 * prog)
                    dz = 0.05 if trig_step is not None else 0.0
                else:
                    lam, ns, dz = 0.0, 0.0, 0.0

                inp_b, inp_m = b_ids[:, :-1], m_ids[:, :-1]
                tgt_b, tgt_m = b_ids[:, 1:], m_ids[:, 1:]
                non_pad = tgt_b != 0

                b_logits, m_logits, mag_loss = model(
                    inp_b, inp_m, target_base_ids=tgt_b, noise_std=ns, deadzone=dz, fp8_eval=False
                )

                # Pure LM Next-Token Loss (weighted 2.5x on <think>/<ans> tokens vs prompt copy)
                ans_region = a_mask[:, 1:]
                tok_weights = torch.where(ans_region, 3.0, 1.0) * non_pad.float()
                ce_b = F.cross_entropy(b_logits.reshape(-1, stats["base_vocab_size"]), tgt_b.reshape(-1), reduction="none")
                ce_m = F.cross_entropy(m_logits.reshape(-1, stats["modifier_vocab_size"]), tgt_m.reshape(-1), reduction="none")
                lm_loss = ((ce_b + 0.5 * ce_m) * tok_weights.reshape(-1)).sum() / tok_weights.sum().clamp(min=1.0)
                loss = lm_loss + lam * mag_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

                with torch.no_grad():
                    if ans_region.any():
                        acc_now = (b_logits.argmax(-1)[ans_region] == tgt_b[ans_region]).float().mean().item()
                        recent_ans_acc = 0.92 * recent_ans_acc + 0.08 * acc_now

        dt = time.time() - t0
        val_slot, val_seq, val_dom, val_samples = evaluate_lm_autoregressive(model, val_docs, tok, fp8=False)
        long_slot, long_seq, _, _ = evaluate_lm_autoregressive(model, long_docs, tok, fp8=True)

        dom_str = " | ".join([f"{k} {v:5.1f}%" for k, v in val_dom.items()])
        print(f"{label} ({n_params:,} params | {dt:.1f}s)")
        print(f"  Zero-Hint Autoregressive Reasoning -> Slot Acc: {val_slot:5.1f}% | Exact Full Answer: {val_seq:5.1f}% [{dom_str}]")
        print(f"  3.5x Long-Horizon FP8 (C=4) Rollout -> Slot Acc: {long_slot:5.1f}% | Exact Full Answer: {long_seq:5.1f}%\n")

        if use_wreath:
            trained_ours = model
            print("--- SAMPLE LIVE AUTOREGRESSIVE THOUGHTS & ANSWERS GENERATED BY OUR MODEL ---")
            for s_i, (orig, gen, t_ans, p_ans) in enumerate(val_samples):
                think_part = "<think>" + gen.split("<think>", 1)[1] if "<think>" in gen else gen[-60:]
                prompt_part = orig.split("<think>", 1)[0].strip()
                print(f"  [Example {s_i+1}] Prompt : {prompt_part}")
                print(f"              Model  : {think_part}")
                print(f"              Check  : Pred='{p_ans}' vs True='{t_ans}'\n")

    inspect_self_invented_circuits(trained_ours, tok)


if __name__ == "__main__":
    run_stage2()
