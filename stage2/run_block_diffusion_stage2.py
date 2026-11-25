"""
Stage 2 (v2) Runner: True Block Diffusion + NCM + TRM + Multi-Head (H=8) Wreath LM on GPU
=========================================================================================
1. Splits each document cleanly at `<think>`:
   - Left-padded Prompt (`p_base, p_mod`) encoded via GPU Chunk-Parallel Wreath Scan (`H=8` heads)
   - Fixed 24-supertoken Target Block (`<think> ... </think> <ans> ... </ans>`) generated IN PARALLEL
     from `<MASK>` across 4 passes (`Pass 0 NCM Skeleton -> TRM Pass 1 -> Pass 2 -> Pass 3`)!
2. Enforces the Exp #6 Dependency-Ordered Magnet Cascade:
   - Clock dead-zone & 200-step magnet trigger at `>= 70%`
   - Downstream Swap-Box magnet held OFF until `Clock >= 92%`!
3. Benchmarks Standard Multi-Head DeltaNet Block-Diffusion LM vs. Proposed Wreath-TRM LM
   across `induce:`, `py:`, `story:`, and `3.5x` Long-Horizon 8-Bit `FP8` (`C = 4`)!
"""

import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp1_adaptive_locking import snap_to_exact_signed_perm
from stage2.block_diffusion_wreath_lm import BlockDiffusionWreathLM
from stage2.reasoning_corpus import build_reasoning_splits
from stage2.tokenizer_super_cobpe import SuperCoBPETokenizer


BLOCK_LEN = 32


def extract_answer_span(text: str) -> str:
    if "<ans>" in text and "</ans>" in text:
        return text.split("<ans>", 1)[1].split("</ans>", 1)[0].strip().replace(" ", "")
    return ""


def collate_block_diffusion(docs: list[str], tok: SuperCoBPETokenizer, block_len: int = BLOCK_LEN):
    """
    Splits each doc into:
      - Left-padded Prompt ending right before `<think>` (so `[:, -1]` is the exact final prompt token!)
      - Right-padded `block_len` Response Block `<think> ... </think> <ans> ... </ans> <EOS>`
    """
    think_id = tok.base_to_id["<think>"]
    prompts_b, prompts_m = [], []
    targets_b, targets_m = [], []

    for d in docs:
        b_ids, m_ids = tok.encode(d)
        cut = b_ids.index(think_id)
        pb, pm = b_ids[:cut], m_ids[:cut]
        tb, tm = b_ids[cut : cut + block_len], m_ids[cut : cut + block_len]
        if len(tb) < block_len:
            pad_n = block_len - len(tb)
            tb = tb + [0] * pad_n
            tm = tm + [0] * pad_n
        prompts_b.append(pb)
        prompts_m.append(pm)
        targets_b.append(tb)
        targets_m.append(tm)

    max_p = max(len(x) for x in prompts_b)
    # Round max_p up to multiple of 4 for clean GPU chunk alignment!
    if max_p % 4 != 0:
        max_p += 4 - (max_p % 4)

    B = len(docs)
    p_base = torch.zeros(B, max_p, dtype=torch.long)
    p_mod = torch.zeros(B, max_p, dtype=torch.long)
    for i, (pb, pm) in enumerate(zip(prompts_b, prompts_m)):
        # LEFT-PAD prompt so `[:, -1]` is ALWAYS the final prompt token!
        Lp = len(pb)
        p_base[i, max_p - Lp :] = torch.tensor(pb, dtype=torch.long)
        p_mod[i, max_p - Lp :] = torch.tensor(pm, dtype=torch.long)

    t_base = torch.tensor(targets_b, dtype=torch.long)
    t_mod = torch.tensor(targets_m, dtype=torch.long)
    return p_base, p_mod, t_base, t_mod


def evaluate_block_diffusion(
    model: BlockDiffusionWreathLM,
    docs: list[str],
    tok: SuperCoBPETokenizer,
    device: str,
    fp8: bool = False,
    snap_swap: bool = False,
    snap_clk: bool = False,
):
    model.eval()
    n_passes = 4 if model.use_wreath_trm else 2
    pass_exact = [0] * n_passes
    slot_ok = 0
    slot_tot = 0
    dom_stats = {"induce:": [0, 0], "py:": [0, 0], "story:": [0, 0]}
    samples = []

    batch_sz = 64
    with torch.no_grad():
        for start in range(0, len(docs), batch_sz):
            sub = docs[start : start + batch_sz]
            pb, pm, _, _ = collate_block_diffusion(sub, tok)
            pb, pm = pb.to(device), pm.to(device)

            b_logs, m_logs, _, _ = model(
                pb,
                pm,
                tgt_base=None,
                box_dz=0.05 if model.use_wreath_trm else 0.0,
                clk_dz=0.05 if model.use_wreath_trm else 0.0,
                fp8_eval=fp8,
                snap_swap=snap_swap,
                snap_clk=snap_clk,
            )

            for p_idx in range(n_passes):
                pred_b = b_logs[p_idx].argmax(-1).cpu().tolist()
                pred_m = m_logs[p_idx].argmax(-1).cpu().tolist()
                for r_i, orig_doc in enumerate(sub):
                    gen_blk = tok.decode(pred_b[r_i], pred_m[r_i])
                    p_ans = extract_answer_span(gen_blk)
                    t_ans = extract_answer_span(orig_doc)
                    is_ex = int(p_ans == t_ans and len(t_ans) > 0)
                    pass_exact[p_idx] += is_ex

                    if p_idx == n_passes - 1:
                        pp = p_ans.split(",")
                        tp = t_ans.split(",")
                        for k_i, tv in enumerate(tp):
                            slot_tot += 1
                            if k_i < len(pp) and pp[k_i] == tv:
                                slot_ok += 1
                        for pref in dom_stats:
                            if orig_doc.startswith(pref):
                                dom_stats[pref][0] += is_ex
                                dom_stats[pref][1] += 1
                        if len(samples) < 3 and orig_doc.startswith(("py:", "story:", "induce:")):
                            samples.append((orig_doc, gen_blk, t_ans, p_ans))

    N = max(1, len(docs))
    pass_pcts = [(c / N) * 100.0 for c in pass_exact]
    slot_pct = (slot_ok / max(1, slot_tot)) * 100.0
    dom_pct = {k: (v[0] / max(1, v[1])) * 100.0 for k, v in dom_stats.items() if v[1] > 0}
    return slot_pct, pass_pcts, dom_pct, samples


def inspect_autonomous_gears(model: BlockDiffusionWreathLM, tok: SuperCoBPETokenizer, device: str):
    probe_doc = (
        "py: set a= 2 ; set b= 5 ; set c= 1 ; set clk= 4 ; "
        "note ok. ; tick clk +2 ; swap a,b ; if clk even then swap a,b -> <think> x </think> <ans> 0 </ans>"
    )
    pb, pm, _, _ = collate_block_diffusion([probe_doc], tok)
    model.eval()
    with torch.no_grad():
        _ = model(pb.to(device), pm.to(device), box_dz=0.05, clk_dz=0.05, snap_swap=True, snap_clk=True)
        G_seq = model.last_G_swap[0].cpu()
        th_deg = torch.rad2deg(model.last_theta[0].cpu())

    b_list = pb[0].tolist()
    m_list = pm[0].tolist()
    eye_d = torch.eye(model.d_swap)
    print("\n--- AUTONOMOUS CIRCUIT DISCOVERY (H=8 Block-Diffusion Wreath LM) ---")
    for pos, (bid, mid) in enumerate(zip(b_list, m_list)):
        if bid == 0:
            continue
        piece = ("" if mid == 0 else tok.id_to_mod[mid]) + tok.id_to_base[bid]
        if any(kw in piece for kw in ["swap", "tick", "+2", "note", "even"]):
            G_snap = snap_to_exact_signed_perm(G_seq[pos : pos + 1])[0]
            swapped = not torch.equal(G_snap, eye_d)
            max_ang = th_deg[pos].abs().max().item()
            print(
                f"  Supertoken [{pos:2d}] {repr(piece):18s} -> Swapper Active: {str(swapped):5s} | Peak Dial Angle: {max_ang:5.1f}°"
            )


def run_stage2_v2():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 94)
    print(f"STAGE 2 (v2): True Block Diffusion + NCM + TRM + Multi-Head (H=8) Wreath LM [Device: {device.upper()}]")
    print("=" * 94)

    train_docs, val_docs, long_docs = build_reasoning_splits(seed=42, n_train=4800, n_val=180, n_long_test=180)

    tok = SuperCoBPETokenizer(subword_merges=85, superword_merges=65)
    stats = tok.train(train_docs)
    mask_id = tok.base_to_id["<MASK>"]
    ans_id = tok.base_to_id["<ans>"]
    ans_end_id = tok.base_to_id["</ans>"]
    print(
        f"[SuperBPE + CoBPE] Base Vocab: {stats['base_vocab_size']} | Modifiers: {stats['modifier_vocab_size']} | "
        f"Compression vs BPE: -{stats['compression_vs_bpe_pct']:.1f}% fewer tokens!\n"
    )

    batch_size = 64
    batches = []
    for idx in range(0, len(train_docs), batch_size):
        pb, pm, tb, tm = collate_block_diffusion(train_docs[idx : idx + batch_size], tok)
        batches.append((pb.to(device), pm.to(device), tb.to(device), tm.to(device)))

    trained_ours = None
    for label, use_wreath in [
        ("1. Standard Multi-Head (H=8) DeltaNet Block-Diffusion LM", False),
        ("2. Proposed NCM + TRM + Multi-Head (H=8) Wreath Block-Diffusion LM", True),
    ]:
        torch.manual_seed(2026)
        model = BlockDiffusionWreathLM(
            base_vocab=stats["base_vocab_size"],
            mod_vocab=stats["modifier_vocab_size"],
            mask_base_id=mask_id,
            d_model=128,
            n_heads=8,
            d_swap=8,
            d_v=32,
            n_dials=4,
            chunk_c=4,
            block_len=BLOCK_LEN,
            use_wreath_trm=use_wreath,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        n_epochs = 32
        total_steps = n_epochs * len(batches)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=2e-4)

        recent_clk_acc = 0.10
        recent_var_acc = 0.10
        clk_trig: int | None = None
        box_trig: int | None = None

        t0 = time.time()
        step = 0
        for _ep in range(n_epochs):
            for pb, pm, tb, tm in batches:
                step += 1
                model.train()

                if use_wreath:
                    # EXP #6 CASCADE LESSON:
                    # 1. Clock dead-zone & 200-step magnet trigger at >= 70%!
                    if clk_trig is None and recent_clk_acc >= 0.70:
                        clk_trig = step
                    if clk_trig is None:
                        lam_clk, ns_clk = 0.0, (0.025 if recent_clk_acc < 0.45 else 0.012)
                    else:
                        pc = min(1.0, (step - clk_trig) / 200.0)
                        lam_clk, ns_clk = 0.35 * pc, 0.005 * (1.0 - 0.85 * pc)

                    # 2. Downstream Swap-Boxes magnet held OFF until Clock >= 83% AND Vars >= 62%!
                    if box_trig is None and (clk_trig is not None and recent_clk_acc >= 0.83) and recent_var_acc >= 0.62:
                        box_trig = step
                    if box_trig is None:
                        lam_box, ns_box = 0.0, (0.025 if recent_var_acc < 0.45 else 0.012)
                    else:
                        pb_p = min(1.0, (step - box_trig) / 200.0)
                        lam_box, ns_box = 0.35 * pb_p, 0.005 * (1.0 - 0.85 * pb_p)

                    dz_clk = 0.05 if clk_trig is not None else 0.0
                    dz_box = 0.05 if recent_var_acc >= 0.60 else 0.0
                else:
                    lam_clk = lam_box = ns_clk = ns_box = dz_clk = dz_box = 0.0

                b_logs, m_logs, b_mag, c_mag = model(
                    pb, pm, tgt_base=tb, box_noise=ns_box, clk_noise=ns_clk, box_dz=dz_box, clk_dz=dz_clk
                )

                # Supervise trailing <PAD> at 0.30 weight so block boundaries stay razor-sharp!
                tok_w = torch.where(tb != 0, 1.0, 0.30)
                denom = tok_w.sum().clamp(min=1.0)

                def pass_ce(bl, ml):
                    cb = F.cross_entropy(bl.reshape(-1, stats["base_vocab_size"]), tb.reshape(-1), reduction="none")
                    cm = F.cross_entropy(ml.reshape(-1, stats["modifier_vocab_size"]), tm.reshape(-1), reduction="none")
                    return ((cb + 0.5 * cm) * tok_w.reshape(-1)).sum() / denom

                if use_wreath:
                    loss = (
                        0.15 * pass_ce(b_logs[0], m_logs[0])
                        + 0.25 * pass_ce(b_logs[1], m_logs[1])
                        + 0.40 * pass_ce(b_logs[2], m_logs[2])
                        + 1.00 * pass_ce(b_logs[3], m_logs[3])
                        + lam_box * b_mag
                        + lam_clk * c_mag
                    )
                else:
                    loss = 0.3 * pass_ce(b_logs[0], m_logs[0]) + 1.0 * pass_ce(b_logs[1], m_logs[1])

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

                # Track live answer-region variable & clock accuracy for the Cascade Controller!
                if use_wreath and (step % 5 == 0):
                    with torch.no_grad():
                        preds_final = b_logs[-1].argmax(-1)
                        # Answer tokens live right before `</ans>` (`ans_end_id`): last token before `</ans>` is clk!
                        is_ans_end = tb == ans_end_id
                        has_end = is_ans_end.any(dim=-1)
                        if has_end.any():
                            end_pos = is_ans_end[has_end].float().argmax(dim=-1)
                            clk_pos = (end_pos - 1).clamp(min=0)
                            var_pos = (end_pos - 3).clamp(min=0)
                            rows = torch.arange(has_end.sum(), device=device)
                            tb_sub = tb[has_end]
                            pr_sub = preds_final[has_end]
                            c_ok = (pr_sub[rows, clk_pos] == tb_sub[rows, clk_pos]).float().mean().item()
                            v_ok = (pr_sub[rows, var_pos] == tb_sub[rows, var_pos]).float().mean().item()
                            recent_clk_acc = 0.90 * recent_clk_acc + 0.10 * c_ok
                            recent_var_acc = 0.90 * recent_var_acc + 0.10 * v_ok

        if device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0

        do_snap_s = use_wreath and (lam_box >= 0.20)
        do_snap_c = use_wreath and (lam_clk >= 0.20)
        v_slot, v_passes, v_dom, v_samples = evaluate_block_diffusion(
            model, val_docs, tok, device, fp8=False, snap_swap=do_snap_s, snap_clk=do_snap_c
        )
        l_slot, l_passes, l_dom, _ = evaluate_block_diffusion(
            model, long_docs, tok, device, fp8=True, snap_swap=do_snap_s, snap_clk=do_snap_c
        )

        dom_s = " | ".join([f"{k} {v:5.1f}%" for k, v in v_dom.items()])
        ldom_s = " | ".join([f"{k} {v:5.1f}%" for k, v in l_dom.items()])
        p_traj = " -> ".join([f"P{i}:{p:5.1f}%" for i, p in enumerate(v_passes)])
        lp_traj = " -> ".join([f"P{i}:{p:5.1f}%" for i, p in enumerate(l_passes)])

        print(f"{label} ({n_params:,} params | {dt:.1f}s on {device.upper()})")
        if use_wreath:
            print(f"  Cascade Status -> Clock Acc: {recent_clk_acc*100:.1f}% (lam_clk={lam_clk:.2f}) | Var Acc: {recent_var_acc*100:.1f}% (lam_box={lam_box:.2f})")
        print(f"  In-Domain Parallel Block Diffusion -> Slot Acc: {v_slot:5.1f}% | Exact Answer Across Passes: [{p_traj}]")
        print(f"    By Domain (Final Pass): {dom_s}")
        print(f"  3.5x Long-Horizon 8-Bit FP8 (C=4)  -> Slot Acc: {l_slot:5.1f}% | Exact Answer Across Passes: [{lp_traj}]")
        print(f"    By Domain (FP8 Long)  : {ldom_s}\n")

        if use_wreath:
            trained_ours = model
            print("--- SAMPLE PARALLEL BLOCK-DIFFUSION OUTPUTS (DENOISED FROM 24 <MASK> TOKENS AT ONCE!) ---")
            for s_i, (orig, gen, t_ans, p_ans) in enumerate(v_samples):
                prompt_str = orig.split("<think>", 1)[0].strip()
                print(f"  [Sample {s_i+1}] Prompt : {prompt_str}")
                print(f"             Denoised: {gen}")
                print(f"             Check   : Pred='{p_ans}' vs True='{t_ans}'\n")

    inspect_autonomous_gears(trained_ours, tok, device)


if __name__ == "__main__":
    run_stage2_v2()
