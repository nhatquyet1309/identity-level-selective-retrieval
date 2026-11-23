#!/usr/bin/env python3
# tierA_smoke.py — prereg_v7 §9.3 bước 1 (smoke bậc A) — chạy trên <GPU_SERVER>
#
#   (a) self-retrieval §4.0  : embed lại từ file đúng đường build index, query
#                              index đầy đủ, cosine ảnh thuần, 4 encoder.
#                              + case S (use_text=True, w_img=0.5) SigLIP2/CLIP-B (chỉ báo).
#   (b) 2 ô kiểm tra tất định §3.1: DINOv2 / CLIP-L, mảng kết quả (path+score) 2 cờ.
#   (d) A(i) SigLIP2-S đối chiếu ref_814.txt (81,4 %).
#   (e) A(i) 6 ô: micro + Wilson95 + macro theo identity + 8 category + strata n_c.
#
# Không sửa test/spec; không cài gì; chỉ đọc index/gallery, ghi ra /tmp/prereg_v7.
#
# Cách chạy:  <PYTHON_ENV>/bin/python /tmp/prereg_v7/tierA_smoke.py --stage a
#             ... --stage b | de | all
import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SERVER_ROOT = Path("<REDACTED_PATH>")
TOOLS = SERVER_ROOT / "tools"
VSTORE = SERVER_ROOT / "data" / "vectorstore"
OUT = Path("/tmp/prereg_v7")
sys.path.insert(0, str(TOOLS))

TOL_FP32 = 1e-5          # §4.0 / §9.1
W_IMG = 0.5              # §2 (không tối ưu)
SEED = 20261022          # §9.1 (chỉ dùng cho bootstrap sau này, không dùng ở đây)
Z95 = 1.959963984540054

# encoder -> (index, meta) — đúng 4 file trong data/vectorstore
ENCODERS = {
    "siglip2": ("visual_siglip.index", "visual_siglip_meta.pkl"),
    "clip": ("visual.index", "visual_meta.pkl"),
    "dinov2": ("visual_dinov2.index", "visual_dinov2_meta.pkl"),
    "clipl14": ("visual_clipl14.index", "visual_clipl14_meta.pkl"),
}
# ô §3.1: encoder -> case  (S = blend text cho siglip2/clip, thuần ảnh cho 2 ô kia)
CELLS = [
    ("siglip2", "S"), ("siglip2", "I"),
    ("clip", "S"), ("clip", "I"),
    ("dinov2", "S"),   # S ≡ I, chạy 1 lần
    ("clipl14", "S"),  # S ≡ I, chạy 1 lần
]
CELL_NAMES = ["siglip2_S", "siglip2_I", "clip_S", "clip_I", "dinov2", "clipl14"]


def log(*a):
    print(*a, flush=True)


def load_meta_index(enc):
    import faiss
    idx_name, meta_name = ENCODERS[enc]
    index = faiss.read_index(str(VSTORE / idx_name))
    with open(VSTORE / meta_name, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def stored_vectors(index):
    import faiss
    buf = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d)
    return np.array(buf, dtype=np.float32, copy=True).reshape(index.ntotal, index.d)


def _norm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _from_pretrained(cls, repo, dtype):
    """giống build_extra_indexes.from_pretrained (transformers 4.57 dùng dtype=)."""
    try:
        return cls.from_pretrained(repo, dtype=dtype)
    except TypeError:
        return cls.from_pretrained(repo, torch_dtype=dtype)


def reembed(enc, paths, batch=16):
    """Embed lại từng ảnh TỪ FILE theo đúng đường lúc build index.

    Nguồn đối chiếu (không suy đoán):
      siglip2 : tools/build_siglip_index.py  — wrap.preprocess -> encode_image,
                từng ảnh một, fp16 (siglip_engine.load_siglip), L2-normalize, fp32.
      clip    : tools/build_visual_index.py  — clip.load("ViT-B/32"), preprocess,
                từng ảnh một, L2-normalize, fp32.
      dinov2  : tools/build_extra_indexes.py — AutoImageProcessor, batch 16, fp16
      clipl14 :   trên cuda; dinov2 = last_hidden_state[:,0]; clipl14 = vision_model
                  pooler_output -> visual_projection; L2-normalize, fp32.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    out = []

    if enc == "siglip2":
        from siglip_engine import load_siglip
        wrap = load_siglip()
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            t = wrap.preprocess(img).unsqueeze(0)
            e = wrap.encode_image(t)
            e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.float().cpu().numpy().astype(np.float32)[0])
            img.close()
            if (i + 1) % 200 == 0:
                log(f"  [siglip2] {i + 1}/{len(paths)}")

    elif enc == "clip":
        import clip
        model, pre = clip.load("ViT-B/32", device=dev)
        model.eval()
        with torch.no_grad():
            for i, p in enumerate(paths):
                img = pre(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
                e = model.encode_image(img)
                e = e / e.norm(dim=-1, keepdim=True)
                out.append(e.float().cpu().numpy().astype(np.float32)[0])
                if (i + 1) % 200 == 0:
                    log(f"  [clip] {i + 1}/{len(paths)}")

    elif enc in ("dinov2", "clipl14"):
        from transformers import AutoImageProcessor
        dtype = torch.float16 if dev == "cuda" else torch.float32
        if enc == "dinov2":
            from transformers import Dinov2Model
            repo = "facebook/dinov2-base"
            model = _from_pretrained(Dinov2Model, repo, dtype).to(dev)
        else:
            from transformers import CLIPModel
            repo = "openai/clip-vit-large-patch14"
            model = _from_pretrained(CLIPModel, repo, dtype).to(dev)
        proc = AutoImageProcessor.from_pretrained(repo)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(paths), batch):
                chunk = paths[start:start + batch]
                imgs = [Image.open(p).convert("RGB") for p in chunk]
                try:
                    px = proc(images=imgs, return_tensors="pt")["pixel_values"]
                    px = px.to(dev, dtype=dtype)
                    if enc == "dinov2":
                        cls_tok = model(pixel_values=px).last_hidden_state[:, 0]
                        e = _norm(cls_tok.float())
                    else:
                        pooled = model.vision_model(pixel_values=px).pooler_output
                        e = _norm(model.visual_projection(pooled).float())
                    out.append(e.cpu().numpy().astype(np.float32))
                finally:
                    for im in imgs:
                        im.close()
                if (start + batch) % 200 < batch:
                    log(f"  [{enc}] {min(start + batch, len(paths))}/{len(paths)}")
    else:
        raise SystemExit(f"encoder la: {enc}")

    mat = np.vstack(out).astype(np.float32)
    log(f"  [{enc}] re-embed {mat.shape} xong sau {time.time() - t0:.1f}s")
    return mat


def get_emb(enc, meta, force=False):
    """Cache embedding re-embed theo encoder (§3.1: tính 1 lần / encoder)."""
    f = OUT / f"x_{enc}.npy"
    if f.exists() and not force:
        mat = np.load(f)
        if mat.shape[0] == len(meta):
            log(f"  [{enc}] dùng cache {f.name} {mat.shape}")
            return mat
    mat = reembed(enc, [m["path"] for m in meta])
    np.save(f, mat)
    return mat


def wilson(k, n, z=Z95):
    if n <= 0:
        return dict(p=None, lo=None, hi=None)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / den
    return dict(p=p, lo=max(0.0, centre - half), hi=min(1.0, centre + half))


# ---------------------------------------------------------------- stage (a)
def stage_a(force=False):
    log("=== (a) self-retrieval §4.0 ===")
    res = {}
    for enc in ENCODERS:
        index, meta = load_meta_index(enc)
        paths = [m["path"] for m in meta]
        Xs = stored_vectors(index)
        Xq = get_emb(enc, meta, force=force)
        # chuẩn hoá lại query (build đã normalize; đây chỉ để chắc chắn)
        Xq = Xq / np.linalg.norm(Xq, axis=1, keepdims=True)

        diag = np.einsum("ij,ij->i", Xq, Xs)          # cosine query re-embed vs vector index
        n = Xq.shape[0]
        t0 = time.time()
        D, I = index.search(Xq.astype(np.float32), n)  # index đầy đủ, không loại gì
        ar = np.arange(n)[:, None]
        self_score = (D * (I == ar)).sum(axis=1)
        others = np.where(I == ar, -np.inf, D).max(axis=1)
        gap = others - self_score
        ok = gap <= TOL_FP32
        tie = np.abs(gap) <= 1e-9
        fail = ~ok
        rank_self = np.argmax(I == ar, axis=1) + 1
        d = dict(
            encoder=enc, dim=int(index.d), ntotal=int(index.ntotal), n=int(n),
            index_type=type(index).__name__,
            tol=TOL_FP32,
            n_ok=int(ok.sum()), n_tie=int(tie.sum()), n_fail=int(fail.sum()),
            n_rank1_unique=int((rank_self == 1).sum()),
            max_gap=float(gap.max()), min_gap=float(gap.min()),
            verdict_tol_1e5="PASS" if bool(ok.all()) else "DUNG",
            verdict_tol_1e3="PASS" if bool((gap <= 1e-3).all()) else "DUNG",
            diag_min=float(diag.min()), diag_mean=float(diag.mean()),
            diag_max_dev=float(np.abs(1.0 - diag).max()),
            sec=round(time.time() - t0, 2),
        )
        if fail.any():
            bad = np.where(fail)[0][:10]
            d["fail_examples"] = [
                dict(path=paths[int(i)], self=float(self_score[i]),
                     best_other=float(others[i]),
                     best_other_path=paths[int(I[int(i)][int(np.argmax(I[int(i)] != i))])])
                for i in bad
            ]
        res[enc] = d
        log(f"  [{enc}] {d['n_ok']}/{d['n']} rank1-unique={d['n_rank1_unique']} "
            f"tie={d['n_tie']} fail={d['n_fail']} max_gap={d['max_gap']:.2e} "
            f"diag[min={d['diag_min']:.8f} maxdev={d['diag_max_dev']:.2e}]")

    # --- case S (chỉ báo, KHÔNG phải điều kiện dừng) --------------------------
    # §4.0 case S = self-retrieval (index ĐẦY ĐỦ, chính ảnh còn trong index).
    # Báo thêm bản LOO (loại chính ảnh) để dùng cho bậc A về sau — ghi rõ nhãn.
    case_s = {}
    for enc in ("siglip2", "clip"):
        index, meta = load_meta_index(enc)
        Xq = get_emb(enc, meta)
        Xq = Xq / np.linalg.norm(Xq, axis=1, keepdims=True)
        Xs = stored_vectors(index)
        ids, Tq, idcol = text_part(enc, meta, Xq)
        Simg = Xq @ Xs.T
        Bl = W_IMG * Simg + (1.0 - W_IMG) * Tq[:, idcol]
        n = len(meta)
        own = np.array([m["variant"] for m in meta])

        # (i) self-retrieval: chính ảnh vẫn trong index
        row = np.arange(n)
        rank_self_img = (Bl > Bl[row, row][:, None]).sum(axis=1) + 1  # ảnh đứng thứ mấy
        top1, best = identity_top1(Bl, meta, ids_override=ids)
        k_id = int((top1 == own).sum())
        # (ii) LOO: loại chính ảnh (KHÔNG phải số của §4.0, chỉ để tham chiếu)
        Bl_loo = Bl.copy()
        Bl_loo[row, row] = -np.inf
        top1l, _ = identity_top1(Bl_loo, meta, ids_override=ids)
        k_loo = int((top1l == own).sum())

        case_s[enc] = dict(
            self_retrieval=dict(k_identity_top1=k_id, n=n,
                                pct=100.0 * k_id / n,
                                n_self_rank1_img=int((rank_self_img == 1).sum()),
                                n_self_outside_top50_img=int((rank_self_img > 50).sum()),
                                max_imgs_above_self=int((rank_self_img - 1).max())),
            loo_reference=dict(k_identity_top1=k_loo, n=n, pct=100.0 * k_loo / n),
            w_img=W_IMG, window="full index (không cắt top-50)")
        log(f"  [caseS {enc}] self-retrieval identity top-1 {k_id}/{n} = "
            f"{100.0 * k_id / n:.2f}% | ảnh tự đứng #1 ở {int((rank_self_img == 1).sum())}/{n} "
            f"| LOO (tham chiếu) {k_loo}/{n} = {100.0 * k_loo / n:.2f}%")
    res["case_s"] = case_s

    (OUT / "selfretrieval.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    return res


def text_part(enc, meta, Xq):
    """Text embedding theo đúng code có sẵn (classify_visual.classify):
    prompt = f"a photo of a {VARIANT_LABELS.get(v, v)}", L2-normalize."""
    from classify_visual import VARIANT_LABELS
    from visual_lab import get_engine
    ids = sorted(set(m["variant"] for m in meta))
    prompts = [f"a photo of a {VARIANT_LABELS.get(v, v)}" for v in ids]
    model, pre, index, meta2, dev = get_engine(enc)
    with torch.no_grad():
        if enc == "siglip2":
            toks = model.tokenize(prompts).to(dev)
            te = model.encode_text(toks)
        else:
            import clip
            toks = clip.tokenize(prompts).to(dev)
            te = model.encode_text(toks)
        te = te / te.norm(dim=-1, keepdim=True)
    Te = te.float().cpu().numpy().astype(np.float32)
    Tq = Xq @ Te.T
    idcol = np.array([ids.index(m["variant"]) for m in meta])
    return ids, Tq, idcol


def identity_top1(S, meta, ids_override=None):
    """top-1 identity: gộp max theo identity, sort (−score, identity_id) §4.5."""
    ids = ids_override or sorted(set(m["variant"] for m in meta))
    n = S.shape[0]
    id_arr = np.array([m["variant"] for m in meta])
    cols = {v: np.where(id_arr == v)[0] for v in ids}
    per_id = np.empty((n, len(ids)), dtype=np.float32)
    for j, v in enumerate(ids):
        per_id[:, j] = S[:, cols[v]].max(axis=1)
    key_ids = np.array(ids, dtype=object)
    order = np.lexsort((np.tile(key_ids, (n, 1)), -per_id), axis=1)
    top1 = key_ids[order[:, 0]]
    best = per_id[np.arange(n), order[:, 0]]
    return top1, best


def window_top1(Simg, Tq, idcol, ids, k=50):
    """Mô phỏng ĐÚNG cửa sổ của classify_visual.classify (index.search(q, 50)
    theo điểm ảnh -> gộp max theo variant -> blend 0,5/0,5) để đối chiếu
    bản "cửa sổ đầy đủ" dùng ở bảng chính. Trả (top1, own_or_None)."""
    n = Simg.shape[0]
    k = min(k, Simg.shape[1])
    part = np.argpartition(-Simg, k - 1, axis=1)[:, :k]
    top1, best_sc = [], []
    for i in range(n):
        agg = {}
        for c in part[i]:
            v = int(idcol[c])
            s = float(Simg[i, c])
            if v not in agg or s > agg[v]:
                agg[v] = s
        ranked = sorted(((-(W_IMG * s + (1.0 - W_IMG) * float(Tq[i, v])), ids[v])
                         for v, s in agg.items()))
        top1.append(ranked[0][1])
        best_sc.append(-ranked[0][0])
    return np.array(top1, dtype=object), np.array(best_sc)


# ---------------------------------------------------------------- stage (b)
def _classify_array(enc, meta, use_text, limit=0):
    from classify_visual import classify
    from visual_lab import get_engine
    model, pre, index, meta2, dev = get_engine(enc)
    paths = [m["path"] for m in meta]
    if limit:
        paths = paths[:limit]
    lines = []
    t0 = time.time()
    for i, p in enumerate(paths):
        r = classify(model, pre, index, meta, p, None, 5, use_text=use_text)
        lines.append(p + "\t" + "\t".join(
            f"{v}|{c}|{float(s).hex()}" for v, c, s in r))
        if (i + 1) % 200 == 0:
            log(f"    [{enc} use_text={use_text}] {i + 1}/{len(paths)} "
                f"({time.time() - t0:.0f}s)")
    blob = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), len(lines)


def stage_b():
    """(b) 2 ô kiểm tra tất định §3.1.

    Đường query thật của production (visual_lab.py:594):
        use_text=(_enc not in ("dinov2","clipl14"))
    → với DINOv2/CLIP-L, CẢ HAI ô S và I đều gọi use_text=False (thuần ảnh, §2).
    Vì vậy "2 cờ" được hiện thực đúng như code production: cùng một lời gọi.
    Để phép so không rỗng, chạy lời gọi đó 2 lần và so hash (tất định), đồng thời
    thử gọi thẳng classify(use_text=True) để ghi lại nó làm gì.
    """
    log("=== (b) 2 ô kiểm tra tất định §3.1 (DINOv2, CLIP-L) ===")
    out = {}
    for enc in ("dinov2", "clipl14"):
        index, meta = load_meta_index(enc)
        h_I, n = _classify_array(enc, meta, use_text=False)
        h_I2, _ = _classify_array(enc, meta, use_text=False)
        # ô S theo đúng biểu thức production: _enc not in ("dinov2","clipl14") -> False
        enc_name = enc
        use_text_S = (enc_name not in ("dinov2", "clipl14"))
        h_S = h_I if not use_text_S else None
        probe = None
        try:
            _classify_array(enc, meta, use_text=True, limit=3)
            probe = "khong loi (co diem text!)"
        except Exception as exc:  # noqa: BLE001
            probe = f"{type(exc).__name__}: {exc}"
        d = dict(encoder=enc, n=n,
                 production_expr="use_text=(_enc not in ('dinov2','clipl14'))",
                 use_text_cell_S=use_text_S, use_text_cell_I=False,
                 sha256_cell_I_run1=h_I, sha256_cell_I_run2=h_I2,
                 sha256_cell_S=h_S,
                 bitexact_I=h_I == h_I2,
                 bitexact_S_vs_I=(h_S == h_I),
                 probe_use_text_true=probe,
                 array_format="path<TAB>variant|category|float.hex() (top-5)")
        out[enc] = d
        log(f"  [{enc}] I={h_I[:16]}… run2={h_I2[:16]}… S={h_S[:16]}… "
            f"bitexact={d['bitexact_I'] and d['bitexact_S_vs_I']} "
            f"| probe use_text=True: {probe}")

    # sanity: cờ phải CÓ tác dụng ở 2 encoder blend (nếu không thì cờ chết)
    sanity = {}
    for enc in ("clip", "siglip2"):
        index, meta = load_meta_index(enc)
        hF, _ = _classify_array(enc, meta, use_text=False, limit=50)
        hT, _ = _classify_array(enc, meta, use_text=True, limit=50)
        sanity[enc] = dict(sha256_use_text_False=hF, sha256_use_text_True=hT,
                           differ=hF != hT, n=50)
        log(f"  [sanity {enc}] 50 ảnh: False={hF[:16]}… True={hT[:16]}… "
            f"differ={hF != hT}")
    out["sanity_flag_active"] = sanity

    for enc in ("dinov2", "clipl14"):
        (OUT / f"bitexact_{enc}.json").write_text(
            json.dumps(out[enc], ensure_ascii=False, indent=2))
    (OUT / "bitexact_sanity.json").write_text(
        json.dumps(sanity, ensure_ascii=False, indent=2))
    return out


# ------------------------------------------------------------ stage (d)+(e)
def build_cells(meta):
    """ma trận điểm từng ô + LOO mask, dùng embedding đã cache (stage a)."""
    ids = sorted(set(m["variant"] for m in meta))
    cells, raw = {}, {}
    for enc in ("siglip2", "clip"):
        index, meta_enc = load_meta_index(enc)
        Xq = get_emb(enc, meta_enc)
        Xq = Xq / np.linalg.norm(Xq, axis=1, keepdims=True)
        Xs = stored_vectors(index)
        Simg = Xq @ Xs.T
        # case I
        I_mat = Simg.copy()
        np.fill_diagonal(I_mat, -np.inf)
        cells[f"{enc}_I"] = I_mat
        # case S
        _, Tq, idcol = text_part(enc, meta_enc, Xq)
        Bl = W_IMG * Simg + (1.0 - W_IMG) * Tq[:, idcol]
        np.fill_diagonal(Bl, -np.inf)
        cells[f"{enc}_S"] = Bl
        raw[f"{enc}_S"] = (Simg, Tq, idcol)
        log(f"  [{enc}] 2 ô xong")
    for enc in ("dinov2", "clipl14"):
        index, meta_enc = load_meta_index(enc)
        Xq = get_emb(enc, meta_enc)
        Xq = Xq / np.linalg.norm(Xq, axis=1, keepdims=True)
        Xs = stored_vectors(index)
        M = Xq @ Xs.T
        np.fill_diagonal(M, -np.inf)
        cells[enc] = M
        log(f"  [{enc}] ô thuần ảnh xong (S≡I)")
    return cells, ids, raw


def summarize(correct, meta, ids):
    """micro + Wilson + macro theo identity + 8 category + strata n_c + sensitivity."""
    own = np.array([m["variant"] for m in meta])
    cat = np.array([m["category"] for m in meta])
    nc = {v: sum(1 for m in meta if m["variant"] == v) for v in ids}
    inD = np.array([nc[v] >= 2 for v in own])
    N = int(inD.sum())
    k = int(correct[inD].sum())

    per_id = []
    for v in ids:
        m = own == v
        if m.sum() < 2:
            continue
        per_id.append(dict(identity=v, n=int(m.sum()),
                           acc=float(correct[m].mean()),
                           k=int(correct[m].sum())))
    macro = float(np.mean([d["acc"] for d in per_id])) if per_id else None

    cats = {}
    for c in sorted(set(cat[inD])):
        m = inD & (cat == c)
        cats[c] = dict(k=int(correct[m].sum()), n=int(m.sum()),
                       pct=100.0 * correct[m].sum() / m.sum())

    strata = {}
    for name, lo, hi in (("<8", 0, 7), ("8-15", 8, 15), ("16-20", 16, 20)):
        mask = np.array([lo <= nc[v] <= hi for v in own]) & inD
        strata[name] = dict(k=int(correct[mask].sum()), n=int(mask.sum()),
                            pct=(100.0 * correct[mask].sum() / mask.sum()
                                 if mask.sum() else None),
                            n_identities=sum(1 for v in ids if lo <= nc[v] <= hi))
    w = wilson(k, N)
    return dict(k=k, N=N, pct=100.0 * k / N, wilson=w,
                macro=macro, n_identities_used=len(per_id),
                categories=cats, strata=strata,
                n_out_of_D=int((~inD).sum()),
                sensitivity_k_all=int(correct.sum()), sensitivity_n_all=len(meta),
                sensitivity_pct_all=100.0 * correct.sum() / len(meta),
                nc_dist=dict(min=int(min(nc.values())), max=int(max(nc.values())),
                             n_ids=len(ids),
                             ids_lt8=sum(1 for v in nc.values() if v < 8),
                             ids_8_15=sum(1 for v in nc.values() if 8 <= v <= 15),
                             ids_16_20=sum(1 for v in nc.values() if 16 <= v <= 20)),
                per_identity=per_id)


def stage_de():
    log("=== (d)+(e) A(i) leave-one-out ===")
    index, meta = load_meta_index("siglip2")  # thứ tự chuẩn
    cells, ids, raw = build_cells(meta)
    own = np.array([m["variant"] for m in meta])
    cat = np.array([m["category"] for m in meta])
    nc = {v: sum(1 for m in meta if m["variant"] == v) for v in ids}

    summary = {}
    top1_by_cell = {}
    for name in CELL_NAMES:
        S = cells[name]
        top1, best = identity_top1(S, meta, ids)
        correct = (top1 == own)
        top1_by_cell[name] = (top1, best, correct)
        s = summarize(correct, meta, ids)
        summary[name] = s
        log(f"  [{name}] {s['k']}/{s['N']} = {s['pct']:.2f}% "
            f"(Wilson {100 * s['wilson']['lo']:.1f}–{100 * s['wilson']['hi']:.1f}) "
            f"macro={100 * s['macro']:.2f}% | all-1667={s['sensitivity_pct_all']:.2f}%")

    # (d) đối chiếu ref_814 ----------------------------------------------------
    # primary = cửa sổ đầy đủ (§4.3 max trên toàn index); sensitivity = mô phỏng
    # đúng cửa sổ top-50 của classify_visual.classify (đường production thật).
    check = {}
    for enc, cell in (("siglip2", "siglip2_S"), ("clip", "clip_S")):
        s = summary[cell]
        Simg, Tq, idcol = raw[cell]
        Simg_loo = Simg.copy()
        np.fill_diagonal(Simg_loo, -np.inf)
        wtop1, _ = window_top1(Simg_loo, Tq, idcol, ids, k=50)
        wk = int((wtop1 == own).sum())
        inD = np.array([nc[v] >= 2 for v in own])
        wN = int(inD.sum())
        dev_pp = s["pct"] - 81.4
        wdev = 100.0 * wk / wN - 81.4
        check[cell] = dict(
            k=s["k"], N=s["N"], pct=s["pct"], dev_pp_vs_81_4=dev_pp,
            pass_05pp=abs(dev_pp) <= 0.5,
            all1667_k=s["sensitivity_k_all"],
            all1667_pct=s["sensitivity_pct_all"],
            window50_k=wk, window50_N=wN, window50_pct=100.0 * wk / wN,
            window50_dev_pp=wdev, window50_pass_05pp=abs(wdev) <= 0.5,
        )
        log(f"  [d] {cell} (cửa sổ đầy đủ): {s['k']}/{s['N']} = {s['pct']:.4f}% "
            f"vs 81,4 → lệch {dev_pp:+.4f}pp "
            f"({'PASS' if abs(dev_pp) <= 0.5 else 'LỆCH'})")
        log(f"      {cell} (mô phỏng top-50 của classify): {wk}/{wN} = "
            f"{100.0 * wk / wN:.4f}% → lệch {wdev:+.4f}pp")

        # kiểm chứng bản mô phỏng top-50 so với lời gọi classify() THẬT
        from classify_visual import classify
        from visual_lab import get_engine
        model, pre, index0, meta0, dev = get_engine(enc)
        idx_probe = list(range(0, len(meta), max(1, len(meta) // 20)))[:20]
        ok = 0
        for i in idx_probe:
            r = classify(model, pre, index0, meta, meta[i]["path"], None, 5,
                         use_text=True)
            if r and r[0][0] == wtop1[i]:
                ok += 1
        check[cell]["window50_vs_classify"] = dict(n=len(idx_probe), agree=ok)
        log(f"      {cell} mô phỏng vs classify() thật: khớp {ok}/{len(idx_probe)}")
    summary["_ref814_check"] = check
    summary["_meta"] = dict(
        n_images=len(meta), n_identities=len(ids),
        categories=sorted(set(cat)), nc_dist=summary["siglip2_S"]["nc_dist"],
        w_img=W_IMG, seed=SEED, tol=TOL_FP32,
        note="embedding query = embed lại từ file (cache stage a), candidate = vector index gốc; "
             "LOO loại chính ảnh query; gộp identity bằng max; sort (-score, identity_id)",
    )
    (OUT / "loo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))

    # CSV để bootstrap sau
    import csv
    with open(OUT / "loo_Ai.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        hdr = ["path", "identity", "category", "n_c", "in_D(i)"]
        for name in CELL_NAMES:
            hdr += [f"{name}_correct", f"{name}_top1", f"{name}_score"]
        w.writerow(hdr)
        for i, m in enumerate(meta):
            row = [m["path"], m["variant"], m["category"], nc[m["variant"]],
                   int(nc[m["variant"]] >= 2)]
            for name in CELL_NAMES:
                top1, best, correct = top1_by_cell[name]
                row += [int(correct[i]), top1[i], f"{float(best[i]):.9f}"]
            w.writerow(row)
    log(f"  ghi {OUT / 'loo_Ai.csv'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["a", "b", "de", "all"])
    ap.add_argument("--force", action="store_true",
                    help="bỏ cache embedding")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"# tierA_smoke.py — {time.strftime('%Y-%m-%dT%H:%M:%S')} — <GPU_SERVER>")
    log(f"# torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    if args.stage in ("a", "all"):
        stage_a(force=args.force)
    if args.stage in ("b", "all"):
        stage_b()
    if args.stage in ("de", "all"):
        stage_de()
    log("# xong")


if __name__ == "__main__":
    main()
