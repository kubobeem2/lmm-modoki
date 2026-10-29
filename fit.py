#!/usr/bin/env python3
"""lmmもどき デモモデルを作る。勾配学習なしの閉形式フィット。

- transformer 本体はランダム初期化して固定 (特徴抽出器)。
- 最終層だけリッジ回帰で閉形式に解く。エポックなし、数秒で終わる。
- できた .modoki はテキストで、そのまま modoki.py で動く。

使い方: python3 fit.py corpus.txt demo.modoki
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from modoki import encode  # noqa: E402


def fmt(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return " ".join(f"{x:.6g}" for x in a)
    return "\n".join(" ".join(f"{x:.6g}" for x in row) for row in a)


def main(corpus_path, out_path):
    lines = [ln.strip() for ln in open(corpus_path, encoding="utf-8") if ln.strip()]
    chars = sorted(set("".join(lines)))
    vocab = ["<bos>", "<eos>", "<pad>"] + chars
    idx = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)

    d, n_layer, n_head, d_ff, max_len = 48, 2, 3, 96, 64
    rng = np.random.default_rng(0)

    def w(shape):
        return rng.normal(0, 1.0 / np.sqrt(shape[-1]), size=shape)

    m = {"model": {"d_model": d, "n_layer": n_layer, "n_head": n_head,
                    "d_ff": d_ff, "max_len": max_len},
         "vocab": vocab,
         "embed": rng.normal(0, 0.5, size=(V, d)),
         "pos": rng.normal(0, 0.5, size=(max_len, d)),
         "layers": [], "ln_f_w": np.ones(d), "ln_f_b": np.zeros(d),
         "lm_head_w": np.zeros((V, d)), "lm_head_b": np.zeros(V)}
    for _ in range(n_layer):
        m["layers"].append({
            "ln1_w": np.ones(d), "ln1_b": np.zeros(d),
            "attn_q_w": w((d, d)), "attn_q_b": np.zeros(d),
            "attn_k_w": w((d, d)), "attn_k_b": np.zeros(d),
            "attn_v_w": w((d, d)), "attn_v_b": np.zeros(d),
            "attn_o_w": w((d, d)), "attn_o_b": np.zeros(d),
            "ln2_w": np.ones(d), "ln2_b": np.zeros(d),
            "mlp1_w": w((d_ff, d)), "mlp1_b": np.zeros(d_ff),
            "mlp2_w": w((d, d_ff)), "mlp2_b": np.zeros(d),
        })

    # 教師データ: 各位置の隠れ状態 -> 次の文字
    Hs, Ys = [], []
    for line in lines:
        ids = [idx["<bos>"]] + [idx[c] for c in line] + [idx["<eos>"]]
        ids = ids[:max_len]
        H = encode(m, ids)
        for t in range(len(ids) - 1):
            Hs.append(H[t])
            Ys.append(ids[t + 1])
    H = np.array(Hs)
    Y = np.array(Ys)
    print(f"学習サンプル数: {len(Y)}, 語彙数: {V}")

    # リッジ回帰の閉形式: W = (H'H + lam I)^-1 H'Y_onehot
    onehot = np.zeros((len(Y), V))
    onehot[np.arange(len(Y)), Y] = 1.0
    lam = 1.0
    A = H.T @ H + lam * np.eye(d)
    W = np.linalg.solve(A, H.T @ onehot)  # (d, V)
    m["lm_head_w"] = W.T
    counts = np.bincount(Y, minlength=V).astype(float)
    m["lm_head_b"] = np.log((counts + 0.1) / (counts.sum() + 0.1 * V))

    pred = H @ W
    acc = (pred.argmax(axis=1) == Y).mean()
    print(f"訓練次文字正解率: {acc:.3f}")

    out = ["# lmmもどき 1", "[model]",
           f"d_model = {d}", f"n_layer = {n_layer}", f"n_head = {n_head}",
           f"d_ff = {d_ff}", f"max_len = {max_len}", "",
           "[vocab]"] + [json.dumps(t, ensure_ascii=False) for t in vocab] + ["",
           "[embed]", fmt(m["embed"]), "",
           "[pos]", fmt(m["pos"])]
    for i, L in enumerate(m["layers"]):
        out.append("")
        out.append(f"[layer {i}]")
        for name in ("ln1_w", "ln1_b", "attn_q_w", "attn_q_b", "attn_k_w",
                     "attn_k_b", "attn_v_w", "attn_v_b", "attn_o_w", "attn_o_b",
                     "ln2_w", "ln2_b", "mlp1_w", "mlp1_b", "mlp2_w", "mlp2_b"):
            out.append(f"{name}:")
            out.append(fmt(L[name]))
    out += ["", "[head]", "ln_f_w:", fmt(m["ln_f_w"]), "ln_f_b:", fmt(m["ln_f_b"]),
            "lm_head_w:", fmt(m["lm_head_w"]), "lm_head_b:", fmt(m["lm_head_b"]), ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"書き出し: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
