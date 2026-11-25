"""
Stage 2 Tokenizer: Two-Stage SuperBPE + Compositional CoBPE
===========================================================
Implements both tokenization breakthroughs requested for Stage 2:
  1. SuperBPE (Liu et al., arXiv:2503.13423):
     - Stage A: Standard subword BPE strictly inside whitespace/punctuation boundaries.
     - Stage B ("Space Travel"): Lifts the whitespace barrier to merge frequent multi-word
       reasoning idioms ("if even", "swap ", " -> ", "<think>", "mod 12") into superwords!
  2. CoBPE (Reif, Kaplan, & Schwartz, 2026):
     - Decomposes phrases compositionally into `(base_token_id, modifier_id)`:
       reusable surface/operator modifiers (`if `, `then `, `= `, `+ `, `swap `, space prefix)
       compose additively in embedding space on input (`E_base + E_mod`) and factorize on output!
"""

from collections import Counter


SURFACE_MODIFIERS = [
    "<NONE>",
    " ",       # Leading space modifier
    "if ",     # Conditional operator modifier
    "then ",   # Consequence operator modifier
    "swap ",   # Permutation operator modifier
    "set ",    # Assignment operator modifier
    "tick ",   # Clock operator modifier
    "rule ",   # Concept/rule marker modifier
    "= ",      # Equality modifier
    "-> ",     # Deduction arrow modifier
]


class SuperCoBPETokenizer:
    def __init__(self, subword_merges: int = 90, superword_merges: int = 70):
        self.subword_merges = subword_merges
        self.superword_merges = superword_merges
        self.mod_to_id = {m: i for i, m in enumerate(SURFACE_MODIFIERS)}
        self.id_to_mod = {i: m for i, m in enumerate(SURFACE_MODIFIERS)}

        self.special_tokens = ["<PAD>", "<MASK>", "<BOS>", "<EOS>", "<think>", "</think>", "<ans>", "</ans>"]
        self.base_to_id: dict[str, int] = {}
        self.id_to_base: dict[int, str] = {}
        self.stage1_merges: list[tuple[str, str]] = []
        self.stage2_merges: list[tuple[str, str]] = []

    def _split_pretokens_whitespace(self, text: str) -> list[list[str]]:
        """Pretokenizes keeping specials intact and keeping leading spaces attached to words."""
        chunks: list[list[str]] = []
        i = 0
        while i < len(text):
            matched_sp = None
            for sp in self.special_tokens:
                if text.startswith(sp, i):
                    matched_sp = sp
                    break
            if matched_sp is not None:
                chunks.append([matched_sp])
                i += len(matched_sp)
                continue

            start = i
            if text[i] == " ":
                i += 1
            while i < len(text) and text[i] != " " and not any(text.startswith(sp, i) for sp in self.special_tokens):
                i += 1
            word = text[start:i]
            if word:
                chunks.append(list(word))
        return chunks

    @staticmethod
    def _apply_merge_to_seq(seq: list[str], pair: tuple[str, str]) -> list[str]:
        out: list[str] = []
        i = 0
        a, b = pair
        ab = a + b
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                out.append(ab)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def train(self, texts: list[str]) -> dict[str, float]:
        # Extract CoBPE modifier prefixes first during encoding, or train SuperBPE over raw text
        # so Stage 1 learns subwords and Stage 2 bridges whitespace into superwords!
        raw_char_count = sum(len(t) for t in texts)
        word_seqs = [self._split_pretokens_whitespace(t) for t in texts]

        # ---- STAGE 1: Standard BPE (strictly inside word boundaries) ----
        for _ in range(self.subword_merges):
            counts: Counter[tuple[str, str]] = Counter()
            for doc in word_seqs:
                for w in doc:
                    for j in range(len(w) - 1):
                        counts[(w[j], w[j + 1])] += 1
            if not counts:
                break
            best_pair, freq = counts.most_common(1)[0]
            if freq < 4:
                break
            self.stage1_merges.append(best_pair)
            word_seqs = [[self._apply_merge_to_seq(w, best_pair) for w in doc] for doc in word_seqs]

        bpe_stage1_len = sum(sum(len(w) for w in doc) for doc in word_seqs)

        # ---- STAGE 2: SuperBPE Curriculum (remove whitespace barrier -> superwords!) ----
        flat_docs = [[tok for w in doc for tok in w] for doc in word_seqs]
        special_set = set(self.special_tokens)
        for _ in range(self.superword_merges):
            counts = Counter()
            for doc in flat_docs:
                for j in range(len(doc) - 1):
                    if doc[j] in special_set or doc[j + 1] in special_set:
                        continue
                    counts[(doc[j], doc[j + 1])] += 1
            if not counts:
                break
            best_pair, freq = counts.most_common(1)[0]
            if freq < 6:
                break
            self.stage2_merges.append(best_pair)
            flat_docs = [self._apply_merge_to_seq(doc, best_pair) for doc in flat_docs]

        superbpe_len = sum(len(doc) for doc in flat_docs)

        # ---- STAGE 3: Build CoBPE Compositional Base Vocabulary ----
        vocab_items = list(self.special_tokens)
        seen = set(vocab_items)
        # Ensure all printable ASCII characters exist as fallback bases
        for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=()[]<>,.:;?! \n":
            if ch not in seen:
                vocab_items.append(ch)
                seen.add(ch)

        for doc in flat_docs:
            for supertok in doc:
                base, _ = self._decompose_cobpe(supertok)
                if base not in seen:
                    vocab_items.append(base)
                    seen.add(base)

        self.base_to_id = {b: idx for idx, b in enumerate(vocab_items)}
        self.id_to_base = {idx: b for idx, b in enumerate(vocab_items)}

        return {
            "raw_chars": raw_char_count,
            "bpe_tokens": bpe_stage1_len,
            "super_cobpe_tokens": superbpe_len,
            "compression_vs_bpe_pct": (1.0 - superbpe_len / max(1, bpe_stage1_len)) * 100.0,
            "base_vocab_size": len(self.base_to_id),
            "modifier_vocab_size": len(self.mod_to_id),
        }

    def _decompose_cobpe(self, supertok: str) -> tuple[str, int]:
        """Splits a SuperBPE token into `(lexical_base, surface_modifier_id)`."""
        if supertok in self.special_tokens:
            return supertok, 0
        # Check longest matching operator/surface modifier prefix first (excluding `<NONE>`)
        best_mod = ""
        best_id = 0
        stripped = supertok.lstrip(" ")
        leading_sp = supertok[: len(supertok) - len(stripped)]

        for mod_str in SURFACE_MODIFIERS[2:]:
            if stripped.startswith(mod_str) and len(stripped) > len(mod_str):
                if len(mod_str) > len(best_mod):
                    best_mod = mod_str
                    best_id = self.mod_to_id[mod_str]

        if best_id != 0:
            base = stripped[len(best_mod) :]
            return base, best_id

        if leading_sp == " " and len(stripped) > 0:
            return stripped, self.mod_to_id[" "]
        return supertok, 0

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        """Returns `(base_ids, mod_ids)` for CoBPE compositional input & factorized output."""
        words = self._split_pretokens_whitespace(text)
        for pair in self.stage1_merges:
            words = [self._apply_merge_to_seq(w, pair) for w in words]
        flat = [tok for w in words for tok in w]
        for pair in self.stage2_merges:
            flat = self._apply_merge_to_seq(flat, pair)

        base_ids = [self.base_to_id["<BOS>"]]
        mod_ids = [0]
        for tok in flat:
            base, mid = self._decompose_cobpe(tok)
            if base in self.base_to_id:
                base_ids.append(self.base_to_id[base])
                mod_ids.append(mid)
            else:
                # Character fallback so zero UNKs ever occur
                for ch in tok:
                    base_ids.append(self.base_to_id.get(ch, 0))
                    mod_ids.append(0)
        base_ids.append(self.base_to_id["<EOS>"])
        mod_ids.append(0)
        return base_ids, mod_ids

    def decode(self, base_ids: list[int], mod_ids: list[int]) -> str:
        pieces = []
        for bid, mid in zip(base_ids, mod_ids):
            b_str = self.id_to_base.get(bid, "")
            if b_str in ("<PAD>", "<MASK>", "<BOS>", "<EOS>"):
                continue
            m_str = "" if mid == 0 else self.id_to_mod.get(mid, "")
            if mid >= 2:
                pieces.append(" " + m_str + b_str)
            else:
                pieces.append(m_str + b_str)
        return "".join(pieces)
