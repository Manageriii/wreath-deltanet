import time
import torch
import torch.nn.functional as F
from exp4_parallel_refinement import parallel_prefix_matmul
from exp1_adaptive_locking import snap_to_exact_signed_perm

device = "mps"
B, L, d_g, d_v = 32, 8192, 6, 24
torch.manual_seed(0)

G = torch.eye(d_g, device=device).view(1, 1, d_g, d_g).repeat(B, L, 1, 1)
k = F.normalize(torch.randn(B, L, d_g, device=device), dim=-1)
v = torch.randn(B, L, d_v, device=device)
beta = torch.rand(B, L, 1, device=device) * 0.1

# Warmup MPS GPU
_ = torch.matmul(G[:, :64], G[:, :64])
torch.mps.synchronize()

# 1. Sequential single-file loop on GPU (8,192 steps)
t0 = time.perf_counter()
Uc = torch.eye(d_g, device=device).unsqueeze(0).expand(B, d_g, d_g)
S = torch.zeros(B, d_v, d_g, device=device)
for t in range(L):
    Uc = torch.bmm(G[:, t], Uc)
    kt = torch.einsum("bij,bi->bj", Uc, k[:, t])
    old = torch.einsum("bvg,bg->bv", S, kt)
    S = S + torch.einsum("bv,bg->bvg", (v[:, t] - old) * beta[:, t], kt)
torch.mps.synchronize()
seq_ms = (time.perf_counter() - t0) * 1000.0
print(f"GPU Sequential Single-File Loop (L=8,192): {seq_ms:7.1f} ms")

# 2. Work-Efficient Chunk-Parallel Engine on GPU across chunk sizes C
for C in [8, 64, 512, 8192]:
    torch.mps.synchronize()
    t0 = time.perf_counter()
    N_c = L // C
    Gc = G.view(B, N_c, C, d_g, d_g)
    U_intra = parallel_prefix_matmul(Gc)
    totals = U_intra[:, :, -1]
    if N_c > 1:
        U_inter = parallel_prefix_matmul(totals)
        eye = torch.eye(d_g, device=device).view(1, 1, d_g, d_g).expand(B, 1, d_g, d_g)
        prev = torch.cat([eye, U_inter[:, :-1]], dim=1)
        U_full = torch.matmul(U_intra, prev.unsqueeze(2)).view(B, L, d_g, d_g)
    else:
        U_full = U_intra.view(B, L, d_g, d_g)

    # Chunk-parallel Delta scan (intra-chunk log2 C + inter-chunk log2 N_c = O(N log C) work!)
    ke = torch.einsum("btij,bti->btj", U_full, k).view(B, N_c, C, d_g)
    vc = v.view(B, N_c, C, d_v)
    bc = beta.view(B, N_c, C, 1)
    I = torch.eye(d_g, device=device).view(1, 1, 1, d_g, d_g)
    A_intra = I - bc.unsqueeze(-1) * torch.einsum("bnci,bncj->bncij", ke, ke)
    B_intra = torch.einsum("bncv,bncg->bncvg", vc * bc, ke)

    st = 1
    while st < C:
        Al, Bl = A_intra[:, :, :-st], B_intra[:, :, :-st]
        Ar, Br = A_intra[:, :, st:], B_intra[:, :, st:]
        A_intra = torch.cat([A_intra[:, :, :st], torch.matmul(Al, Ar)], dim=2)
        B_intra = torch.cat([B_intra[:, :, :st], torch.matmul(Bl, Ar) + Br], dim=2)
        st *= 2

    if N_c > 1:
        Ac, Bc = A_intra[:, :, -1], B_intra[:, :, -1]
        st = 1
        while st < N_c:
            Al, Bl = Ac[:, :-st], Bc[:, :-st]
            Ar, Br = Ac[:, st:], Bc[:, st:]
            Ac = torch.cat([Ac[:, :st], torch.matmul(Al, Ar)], dim=1)
            Bc = torch.cat([Bc[:, :st], torch.matmul(Bl, Ar) + Br], dim=1)
            st *= 2
        S_final = Bc[:, -1]
    else:
        S_final = B_intra[:, 0, -1]

    torch.mps.synchronize()
    par_ms = (time.perf_counter() - t0) * 1000.0
    print(f"GPU Parallel Chunk Engine (C={C:4d}, {N_c:4d} chunks): {par_ms:7.1f} ms ({seq_ms/par_ms:5.1f}x FASTER!)")
