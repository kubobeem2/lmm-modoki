# lmmもどき — ビルド不要のテキストモデル形式

> 名前の由来: 最初は LMMS と名乗っていたが、有名な DAW (Linux MultiMedia Studio)
> と被るので「lmmもどき」に改名した。リポジトリ名は `lmm-modoki`。

py ファイルみたいに扱える、小さな LLM 用の形式。
**テキストを書いたらそれがそのままモデル。コンパイルなし、変換なし。**
`modoki.py` がファイルを直接読んで動かす。

## 動かす

```sh
cd ~/workspace/lmm-script
python3 modoki.py demo.modoki --prompt "ねこは" --tokens 40
python3 modoki.py demo.modoki --prompt "今日は" --tokens 40 --temp 0.8 --seed 7
```

`--temp 0` (既定) は貪欲法。`--temp` を上げるとランダムになる。
`--seed` で再現できる。依存は numpy だけ。

## 自分のモデルを作る

```sh
# 1. corpus.txt に1行1文で例文を書く (ひらがな中心だとよく学ぶ)
# 2. 学習して .modoki を吐く
python3 train.py corpus.txt demo.modoki 600
# 3. そのまま動かす。ビルド工程は存在しない
python3 modoki.py demo.modoki --prompt "ねこは" --tokens 40
```

`train.py` は numpy の手書き逆伝播 + Adam で、2層の tiny transformer を
鍛える。`fit.py` は別ルートで、ランダム特徴 + リッジ回帰の閉形式
(勾配なし・数秒) で作る。どちらも出力は同じ lmmもどき 形式。

## ファイル

- `SPEC.md` — 形式の仕様
- `modoki.py` — インタプリタ (パーサー + 順伝播 + 生成)
- `train.py` — 勾配学習でモデルを作る
- `fit.py` — 閉形式フィットでモデルを作る (お手軽版)
- `corpus.txt` — デモ用の例文 (約120文)
- `demo.modoki` — デモモデル (語彙70文字、2層、約5万パラメータ)

## 形式の要点 (詳しくは SPEC.md)

- UTF-8 テキスト。`[model]` `[vocab]` `[embed]` `[pos]` `[layer N]` `[head]`。
- 行列は `名前:` の行のあと数値を空白区切りで並べるだけ。
- `[vocab]` は1行1トークンの JSON 文字列。行番号が ID。
- 各層は pre-norm transformer (`attn` + `mlp`、GELU、causal attention)。
- 生成時は語彙に `<bos>` があれば文頭に自動で付ける。

## 正直な注記

- デモは120文しか学んでないおもちゃ。知らない話題は出ないし、
  長文になると崩れる。形式の実証用。
- 数値は `%.6g` で丸めてあるので、元の学習結果とビット一致はしない。
  実用サイズならもっと桁を残すこと。
