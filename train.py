#!/usr/bin/env python3
"""lmmもどき デモモデルを勾配学習で鍛える。バッチ化した手書き逆伝播 + Adam。

使い方: python3 train.py corpus.txt demo.modoki [--steps 600]
できた .modoki はテキストで、そのまま modoki.py で動く。
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fit import fmt  # noqa: E402


def ln_fwd(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xh = (x - mu) / np.sqrt(var + eps)
    return xh * w + b, (xh, var, w)


def ln_bwd(dy, cache, eps=1e-5):
    xh, var, w = cache
    dyw = dy * w
    dx = (dyw - dyw.mean(axis=-1, keepdims=True)
          - xh * (dyw * xh).mean(axis=-1, keepdims=True)) / np.sqrt(var + eps)
    return dx, (dy * xh).sum(axis=(0, 1)), dy.sum(axis=(0, 1))


def gelu(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def gelu_bwd(dg, u):
    c = np.sqrt(2.0 / np.pi)
    a = 0.044715
    z = c * (u + a * u**3)
    th = np.tanh(z)
    return dg * (0.5 * (1.0 + th) + 0.5 * u * (1 - th**2) * c * (1 + 3 * a * u**2))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class Net:
    def __init__(self, V, d=48, n_layer=2, n_head=3, d_ff=96, max_len=64, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = dict(d=d, n_layer=n_layer, n_head=n_head, d_ff=d_ff, max_len=max_len)
        self.V = V

        def w(shape):
            return rng.normal(0, 1.0 / np.sqrt(shape[-1]), size=shape)

        self.P = {}
        P = self.P
        P["embed"] = rng.normal(0, 0.5, size=(V, d))
        P["pos"] = rng.normal(0, 0.5, size=(max_len, d))
        for i in range(n_layer):
            p = f"L{i}_"
            P[p + "ln1_w"] = np.ones(d); P[p + "ln1_b"] = np.zeros(d)
            for n in ("q", "k", "v", "o"):
                P[p + f"attn_{n}_w"] = w((d, d)); P[p + f"attn_{n}_b"] = np.zeros(d)
            P[p + "ln2_w"] = np.ones(d); P[p + "ln2_b"] = np.zeros(d)
            P[p + "mlp1_w"] = w((d_ff, d)); P[p + "mlp1_b"] = np.zeros(d_ff)
            P[p + "mlp2_w"] = w((d, d_ff)); P[p + "mlp2_b"] = np.zeros(d)
        P["ln_f_w"] = np.ones(d); P["ln_f_b"] = np.zeros(d)
        P["lm_head_w"] = w((V, d)); P["lm_head_b"] = np.zeros(V)

    def forward(self, ids, lengths):
        """ids: (B,T) int, lengths: (B,) 実長。logits (B,T,V) と cache を返す。"""
        cfg, P = self.cfg, self.P
        d, nh, dh = cfg["d"], cfg["n_head"], cfg["d"] // cfg["n_head"]
        B, T = ids.shape
        x = P["embed"][ids] + P["pos"][:T][None, :, :]
        causal = np.triu(np.ones((T, T)) * -1e9, k=1)[None, None, :, :]
        key_ok = (np.arange(T)[None, :] < lengths[:, None])[:, None, None, :]
        mask = causal + np.where(key_ok, 0.0, -1e9)
        caches = []
        for i in range(cfg["n_layer"]):
            p = f"L{i}_"
            h1, c1 = ln_fwd(x, P[p + "ln1_w"], P[p + "ln1_b"])
            q = (h1 @ P[p + "attn_q_w"].T + P[p + "attn_q_b"]).reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
            k = (h1 @ P[p + "attn_k_w"].T + P[p + "attn_k_b"]).reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
            v = (h1 @ P[p + "attn_v_w"].T + P[p + "attn_v_b"]).reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh) + mask
            att = softmax(s)
            a = att @ v
            a2 = a.transpose(0, 2, 1, 3).reshape(B, T, d)
            x1 = x + a2 @ P[p + "attn_o_w"].T + P[p + "attn_o_b"]
            h2, c2 = ln_fwd(x1, P[p + "ln2_w"], P[p + "ln2_b"])
            u = h2 @ P[p + "mlp1_w"].T + P[p + "mlp1_b"]
            g = gelu(u)
            x2 = x1 + g @ P[p + "mlp2_w"].T + P[p + "mlp2_b"]
            caches.append((x, h1, c1, q, k, v, att, a2, x1, h2, c2, u, g))
            x = x2
        hf, cf = ln_fwd(x, P["ln_f_w"], P["ln_f_b"])
        logits = hf @ P["lm_head_w"].T + P["lm_head_b"]
        return logits, (caches, hf, cf, ids)

    def backward(self, dlogits, cache):
        cfg, P = self.cfg, self.P
        d, nh, dh = cfg["d"], cfg["n_head"], cfg["d"] // cfg["n_head"]
        caches, hf, cf, ids = cache
        B, T, V = dlogits.shape
        G = {k: np.zeros_like(v) for k, v in P.items()}
        G["lm_head_w"] = dlogits.transpose(0, 2, 1).reshape(-1, V).T @ hf.reshape(-1, d) if False else np.einsum("btv,btd->vd", dlogits, hf)
        G["lm_head_b"] = dlogits.sum(axis=(0, 1))
        dhf = dlogits @ P["lm_head_w"]
        dx, dw, db = ln_bwd(dhf, cf)
        G["ln_f_w"] += dw; G["ln_f_b"] += db
        for i in reversed(range(cfg["n_layer"])):
            p = f"L{i}_"
            (x, h1, c1, q, k, v, att, a2, x1, h2, c2, u, g) = caches[i]
            # MLP
            dmlp = dx
            dx1 = dx.copy()
            G[p + "mlp2_w"] += np.einsum("btd,btf->df", dmlp, g)
            G[p + "mlp2_b"] += dmlp.sum(axis=(0, 1))
            dg = dmlp @ P[p + "mlp2_w"]
            du = gelu_bwd(dg, u)
            G[p + "mlp1_w"] += np.einsum("btf,btd->fd", du, h2)
            G[p + "mlp1_b"] += du.sum(axis=(0, 1))
            dh2 = du @ P[p + "mlp1_w"]
            dx1b, dw, db = ln_bwd(dh2, c2)
            G[p + "ln2_w"] += dw; G[p + "ln2_b"] += db
            dx1 += dx1b
            # Attention
            do = dx1
            G[p + "attn_o_w"] += np.einsum("btd,bta->da", do, a2)
            G[p + "attn_o_b"] += do.sum(axis=(0, 1))
            da = do @ P[p + "attn_o_w"]
            dah = da.reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
            dv = att.transpose(0, 1, 3, 2) @ dah
            dp = dah @ v.transpose(0, 1, 3, 2)
            ds = att * (dp - (dp * att).sum(axis=-1, keepdims=True)) / np.sqrt(dh)
            dq = ds @ k
            dk = ds.transpose(0, 1, 3, 2) @ q
            dq2 = dq.transpose(0, 2, 1, 3).reshape(B, T, d)
            dk2 = dk.transpose(0, 2, 1, 3).reshape(B, T, d)
            dv2 = dv.transpose(0, 2, 1, 3).reshape(B, T, d)
            G[p + "attn_q_w"] += np.einsum("btd,bte->de", dq2, h1)
            G[p + "attn_q_b"] += dq2.sum(axis=(0, 1))
            G[p + "attn_k_w"] += np.einsum("btd,bte->de", dk2, h1)
            G[p + "attn_k_b"] += dk2.sum(axis=(0, 1))
            G[p + "attn_v_w"] += np.einsum("btd,bte->de", dv2, h1)
            G[p + "attn_v_b"] += dv2.sum(axis=(0, 1))
            dh1 = dq2 @ P[p + "attn_q_w"] + dk2 @ P[p + "attn_k_w"] + dv2 @ P[p + "attn_v_w"]
            dxb, dw, db = ln_bwd(dh1, c1)
            G[p + "ln1_w"] += dw; G[p + "ln1_b"] += db
            dx = dx1 + dxb
        # embed / pos
        np.add.at(G["embed"], ids, dx)
        G["pos"][:T] += dx.sum(axis=0)
        return G


def main(corpus_path, out_path, steps=600, lr=0.005):
    lines = [ln.strip() for ln in open(corpus_path, encoding="utf-8") if ln.strip()]
    chars = sorted(set("".join(lines)))
    vocab = ["<bos>", "<eos>", "<pad>"] + chars
    idx = {c: i for i, c in enumerate(vocab)}
    V, pad = len(vocab), idx["<pad>"]
    seqs = []
    for line in lines:
        ids = [idx["<bos>"]] + [idx[c] for c in line] + [idx["<eos>"]]
        seqs.append(ids)
    net = Net(V)
    P = net.P
    mt = {k: np.zeros_like(v) for k, v in P.items()}
    vt = {k: np.zeros_like(v) for k, v in P.items()}
    rng = np.random.default_rng(1)
    step = 0
    for epoch in range(10000):
        order = rng.permutation(len(seqs))
        for bi in range(0, len(seqs), 16):
            batch = [seqs[j] for j in order[bi:bi + 16]]
            T = max(len(s) for s in batch)
            B = len(batch)
            ids = np.full((B, T), pad, dtype=int)
            lengths = np.array([len(s) for s in batch])
            for b, s in enumerate(batch):
                ids[b, :len(s)] = s
            x_ids, y_ids = ids[:, :-1], ids[:, 1:]
            y_lengths = lengths - 1
            logits, cache = net.forward(x_ids, y_lengths)
            # 損失: 有効位置のみの交差エントロピー
            valid = (np.arange(x_ids.shape[1])[None, :] < y_lengths[:, None])
            lp = logits - logits.max(axis=-1, keepdims=True)
            lp = lp - np.log(np.exp(lp).sum(axis=-1, keepdims=True))
            nll = -lp[np.arange(B)[:, None], np.arange(x_ids.shape[1])[None, :], y_ids]
            loss = (nll * valid).sum() / valid.sum()
            dlogits = np.exp(lp)
            dlogits[np.arange(B)[:, None], np.arange(x_ids.shape[1])[None, :], y_ids] -= 1.0
            dlogits *= (valid / valid.sum())[:, :, None]
            G = net.backward(dlogits, cache)
            step += 1
            # Adam
            b1, b2, eps = 0.9, 0.999, 1e-8
            for k in P:
                mt[k] = b1 * mt[k] + (1 - b1) * G[k]
                vt[k] = b2 * vt[k] + (1 - b2) * G[k] ** 2
                mh = mt[k] / (1 - b1**step)
                vh = vt[k] / (1 - b2**step)
                P[k] -= lr * mh / (np.sqrt(vh) + eps)
            if step % 100 == 0:
                print(f"step {step} loss {loss:.3f}", flush=True)
            if step >= steps:
                break
        if step >= steps:
            break
    print(f"最終 loss {loss:.3f} (step {step})")

    out = ["# lmmもどき 1", "[model]",
           f"d_model = {net.cfg['d']}", f"n_layer = {net.cfg['n_layer']}",
           f"n_head = {net.cfg['n_head']}", f"d_ff = {net.cfg['d_ff']}",
           f"max_len = {net.cfg['max_len']}", "",
           "[vocab]"] + [json.dumps(t, ensure_ascii=False) for t in vocab] + ["",
           "[embed]", fmt(P["embed"]), "",
           "[pos]", fmt(P["pos"])]
    for i in range(net.cfg["n_layer"]):
        p = f"L{i}_"
        out.append("")
        out.append(f"[layer {i}]")
        for name in ("ln1_w", "ln1_b", "attn_q_w", "attn_q_b", "attn_k_w",
                     "attn_k_b", "attn_v_w", "attn_v_b", "attn_o_w", "attn_o_b",
                     "ln2_w", "ln2_b", "mlp1_w", "mlp1_b", "mlp2_w", "mlp2_b"):
            out.append(f"{name}:")
            out.append(fmt(P[p + name]))
    out += ["", "[head]", "ln_f_w:", fmt(P["ln_f_w"]), "ln_f_b:", fmt(P["ln_f_b"]),
            "lm_head_w:", fmt(P["lm_head_w"]), "lm_head_b:", fmt(P["lm_head_b"]), ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"書き出し: {out_path}")


if __name__ == "__main__":
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    main(sys.argv[1], sys.argv[2], steps=steps)
