#!/usr/bin/env python3
"""lmmもどき 1.0 インタプリタ: テキスト形式のモデルをそのまま読んで動かす。

使い方:
    python3 modoki.py demo.modoki --prompt "ねこは" --tokens 40
    python3 modoki.py demo.modoki --prompt "今日は" --tokens 40 --temp 0.9 --seed 1

依存は numpy のみ。ビルドも変換もしない。ファイルがそのままモデル。
"""
import argparse
import json
import re
import sys

import numpy as np


class FormatError(Exception):
    pass


def _strip_comment(line, in_vocab):
    if in_vocab:
        return line
    if "#" in line:
        return line[: line.index("#")]
    return line


def parse_modoki(path):
    """lmmもどき テキストをパースして dict を返す。未知セクションでは止まる。"""
    model = {}
    vocab = []
    mats = {}  # (section, name) -> np.ndarray
    section = None
    cur_mat = None  # (section, name)
    cur_rows = []

    def flush_mat():
        nonlocal cur_mat, cur_rows
        if cur_mat is not None:
            if not cur_rows:
                raise FormatError(f"行列 {cur_mat} に行がありません")
            mats[cur_mat] = np.array(cur_rows, dtype=np.float64)
            cur_mat, cur_rows = None, []

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = _strip_comment(raw.rstrip("\n"), section == "vocab").strip()
            if not line:
                flush_mat()
                continue
            m = re.match(r"^\[(.+)\]$", line)
            if m:
                flush_mat()
                section = m.group(1).strip()
                known = section in ("model", "vocab", "embed", "pos", "head") or re.match(
                    r"^layer \d+$", section
                )
                if not known:
                    raise FormatError(f"未知のセクション: [{section}]")
                continue
            if section is None:
                raise FormatError(f"セクションの外の行: {line!r}")
            if section == "model":
                if "=" not in line:
                    raise FormatError(f"[model] の行が key = value ではありません: {line!r}")
                k, v = line.split("=", 1)
                model[k.strip()] = int(v.strip())
                continue
            if section == "vocab":
                try:
                    tok = json.loads(line)
                except json.JSONDecodeError as e:
                    raise FormatError(f"[vocab] の行が JSON 文字列ではありません: {line!r}") from e
                if not isinstance(tok, str):
                    raise FormatError(f"[vocab] の行が文字列ではありません: {line!r}")
                vocab.append(tok)
                continue
            # 行列セクション
            if line.endswith(":"):
                flush_mat()
                cur_mat = (section, line[:-1].strip())
                cur_rows = []
                continue
            if cur_mat is None:
                if section in ("embed", "pos"):
                    cur_mat = (section, "w")
                    cur_rows = []
                else:
                    raise FormatError(f"行列名のない数値行: {line!r} (セクション [{section}])")
            try:
                cur_rows.append([float(x) for x in line.split()])
            except ValueError as e:
                raise FormatError(f"数値でない行: {line!r}") from e
    flush_mat()

    for k in ("d_model", "n_layer", "n_head", "d_ff", "max_len"):
        if k not in model:
            raise FormatError(f"[model] に {k} がありません")
    d = model["d_model"]
    if d % model["n_head"] != 0:
        raise FormatError("d_model は n_head で割り切れる必要があります")
    if not vocab:
        raise FormatError("[vocab] が空です")

    def need(sec, name, shape=None):
        key = (sec, name)
        if key not in mats:
            raise FormatError(f"行列がありません: [{sec}] {name}")
        a = mats[key]
        if shape is not None and a.shape != shape:
            # 1 行で書かれたベクトルは (1, n) で読めるので潰す
            if a.size == int(np.prod(shape)):
                a = a.reshape(shape)
            else:
                raise FormatError(f"行列 [{sec}] {name} の形が {a.shape}、期待は {shape}")
        return a

    v, max_len = len(vocab), model["max_len"]
    # embed / pos はセクション自体が1つの行列 (名前なしでもよい)
    def section_matrix(sec, shape):
        for name in ("", "embed", "pos", "w"):
            key = (sec, name)
            if key in mats:
                a = mats[key]
                if a.shape != shape:
                    raise FormatError(f"行列 [{sec}] {name} の形が {a.shape}、期待は {shape}")
                return a
        raise FormatError(f"行列がありません: [{sec}]")

    out = {"model": model, "vocab": vocab}
    out["embed"] = section_matrix("embed", (v, d))
    out["pos"] = section_matrix("pos", (max_len, d))
    layers = []
    for i in range(model["n_layer"]):
        sec = f"layer {i}"
        L = {}
        for name, shape in (
            ("ln1_w", (d,)), ("ln1_b", (d,)),
            ("attn_q_w", (d, d)), ("attn_q_b", (d,)),
            ("attn_k_w", (d, d)), ("attn_k_b", (d,)),
            ("attn_v_w", (d, d)), ("attn_v_b", (d,)),
            ("attn_o_w", (d, d)), ("attn_o_b", (d,)),
            ("ln2_w", (d,)), ("ln2_b", (d,)),
            ("mlp1_w", (model["d_ff"], d)), ("mlp1_b", (model["d_ff"],)),
            ("mlp2_w", (d, model["d_ff"])), ("mlp2_b", (d,)),
        ):
            L[name] = need(sec, name, shape)
        layers.append(L)
    out["layers"] = layers
    out["ln_f_w"] = need("head", "ln_f_w", (d,))
    out["ln_f_b"] = need("head", "ln_f_b", (d,))
    out["lm_head_w"] = need("head", "lm_head_w", (v, d))
    out["lm_head_b"] = need("head", "lm_head_b", (v,))
    return out


def ln(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def encode(m, ids):
    """ids: list[int] -> 全位置の最終隠れ状態 (T, d_model)"""
    cfg = m["model"]
    d, nh, dh = cfg["d_model"], cfg["n_head"], cfg["d_model"] // cfg["n_head"]
    t = len(ids)
    if t > cfg["max_len"]:
        raise FormatError(f"入力が max_len={cfg['max_len']} を超えました ({t})")
    x = m["embed"][ids] + m["pos"][:t]
    mask = np.triu(np.ones((t, t)) * -1e9, k=1)
    for L in m["layers"]:
        h = ln(x, L["ln1_w"], L["ln1_b"])
        q = (h @ L["attn_q_w"].T + L["attn_q_b"]).reshape(t, nh, dh).transpose(1, 0, 2)
        k = (h @ L["attn_k_w"].T + L["attn_k_b"]).reshape(t, nh, dh).transpose(1, 0, 2)
        v = (h @ L["attn_v_w"].T + L["attn_v_b"]).reshape(t, nh, dh).transpose(1, 0, 2)
        att = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(dh) + mask, axis=-1) @ v
        att = att.transpose(1, 0, 2).reshape(t, d)
        x = x + att @ L["attn_o_w"].T + L["attn_o_b"]
        h = ln(x, L["ln2_w"], L["ln2_b"])
        x = x + gelu(h @ L["mlp1_w"].T + L["mlp1_b"]) @ L["mlp2_w"].T + L["mlp2_b"]
    return ln(x, m["ln_f_w"], m["ln_f_b"])


def forward(m, ids):
    """ids: list[int] -> 最終位置の logits (vocab_size,)"""
    h = encode(m, ids)
    return h[-1] @ m["lm_head_w"].T + m["lm_head_b"]


def build_tokenizer(vocab):
    """最長一致の文字トークナイザ。特殊トークン (<...>) は入力対象外。"""
    index = {}
    for i, tok in enumerate(vocab):
        if tok.startswith("<") and tok.endswith(">"):
            continue
        index[tok] = i
    maxlen = max((len(t) for t in index), default=1)
    return index, maxlen


def tokenize(text, index, maxlen):
    ids, unknown, i = [], [], 0
    while i < len(text):
        matched = None
        for l in range(min(maxlen, len(text) - i), 0, -1):
            piece = text[i : i + l]
            if piece in index:
                matched = (index[piece], l)
                break
        if matched is None:
            unknown.append(text[i])
            i += 1
        else:
            ids.append(matched[0])
            i += matched[1]
    return ids, unknown


def generate(m, prompt, tokens, temp, seed, topk):
    index, maxlen = build_tokenizer(m["vocab"])
    ids, unknown = tokenize(prompt, index, maxlen)
    if unknown:
        raise FormatError(f"語彙にない文字があります: {''.join(sorted(set(unknown)))}")
    # 学習時と同じく文頭に <bos> を付ける (語彙にある場合)
    if "<bos>" in m["vocab"]:
        ids = [m["vocab"].index("<bos>")] + ids
    if not ids:
        raise FormatError("プロンプトが空です")
    rng = np.random.default_rng(seed)
    out = list(ids)
    for _ in range(tokens):
        ctx = out[-m["model"]["max_len"] :]
        logits = forward(m, ctx)
        if temp == 0:
            nxt = int(np.argmax(logits))
        else:
            z = logits / temp
            if topk and topk < len(z):
                cut = np.sort(z)[-topk]
                z = np.where(z >= cut, z, -1e9)
            p = softmax(z)
            nxt = int(rng.choice(len(p), p=p))
        out.append(nxt)
        if m["vocab"][nxt] == "<eos>":
            break
    tail = [m["vocab"][i] for i in out[len(ids) :]]
    return "".join(t for t in tail if not (t.startswith("<") and t.endswith(">")))


def main(argv=None):
    ap = argparse.ArgumentParser(description="lmmもどき 1.0 インタプリタ")
    ap.add_argument("model", help="*.modoki ファイル")
    ap.add_argument("--prompt", required=True, help="生成の種になる文字列")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.0, help="0 なら貪欲法")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=0, help="0 なら制限なし")
    args = ap.parse_args(argv)
    try:
        m = parse_modoki(args.model)
        text = generate(m, args.prompt, args.tokens, args.temp, args.seed, args.topk)
    except FormatError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    print(args.prompt + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
