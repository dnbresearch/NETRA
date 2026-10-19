#!/usr/bin/env python3
"""ZPE baseline for Table 2 (Allingham et al., ICML 2023).

Scoring modes (--scoring):
  maxlogit  (default) Original ZPE, Algorithm 2 "normalized max logit":
            each prompt p defines a classifier; logits L = X @ W_p^T (cosine);
            score_p = mean_i max_c ( L[i,c] - bias[c] ),
            bias = (E_test + E_pretrain) / 2, each a per-class mean of the logits
            over unlabeled test images / a pre-training image sample.
            Weights = softmax(scores) over the prompt pool (ZPE Eq. 5).
            Without --pretrain-features only E_test is used; the paper's Table 3
            shows E_test alone is slightly weaker than E_test+E_pretrain, so pass
            the pre-training features when you can (see --build-pretrain-features).
  entropy   The variant used for the earlier results: score_p = -mean entropy of
            softmax(logit_scale * L). Reported in the paper as ZPE* (entropy variant).

Descriptions:
  --descriptions-json  Reuse the SAME descriptions as NETRA's Table 2 row (a JSON
                       {class_name: [desc, ...]}, or a directory holding
                       <dataset>.json). This is what makes the comparison controlled.
  (otherwise)          Generate with the paper's Appendix D prompt via LLMClient.

Prompt pool for "ZPE (flat pool)": the 80 ImageNet templates (each a full
classifier) plus every class-specific description, represented as the template
classifier with class c's row replaced by that description (as before).

Usage:
    python scripts/run_zpe_baseline.py \
        --datasets cifar100 flowers102 dtd oxford_pets food101 caltech101 cifar10 eurosat fgvc_aircraft country211 \
        --clip-model ViT-L/14 --descriptions-json experiments/descriptions/gpt-4o \
        --pretrain-features experiments/zpe/laion20k_vitl14.pt \
        --output-dir experiments/zpe

    # one-off: encode a folder of ~20k unlabeled images (e.g. a LAION-400M sample)
    python scripts/run_zpe_baseline.py --build-pretrain-features /path/to/images \
        --clip-model ViT-L/14 --pretrain-features experiments/zpe/laion20k_vitl14.pt

    # Table 3 (large scale): ImageNet-1K val, ImageNet-V2, DomainNet domains as ImageFolder trees
    python scripts/run_zpe_baseline.py --datasets \
        --imagefolder imagenet=/data/imagenet/val --classnames imagenet=data/imagenet_classnames.json \
        --imagefolder imagenet_v2=/data/imagenetv2-matched-frequency --classnames imagenet_v2=data/imagenet_classnames.json \
        --imagefolder domainnet_real=/data/domainnet/real --imagefolder domainnet_clipart=/data/domainnet/clipart \
        --imagefolder domainnet_sketch=/data/domainnet/sketch --imagefolder domainnet_infograph=/data/domainnet/infograph \
        --imagefolder domainnet_quickdraw=/data/domainnet/quickdraw \
        --descriptions-json experiments/descriptions/gpt-4o --output-dir experiments/zpe_largescale

    # reproduce the earlier entropy-variant numbers
    python scripts/run_zpe_baseline.py --scoring entropy ...

Outputs: <output-dir>/zpe_results.json, zpe_table2.csv (one row per method),
and per-dataset prompt weights.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from visprompt.baselines import IMAGENET_TEMPLATES
from visprompt.utils.llm import CostTracker, LLMClient

logger = logging.getLogger(__name__)

# Paper, Appendix D (identical to the prompt used for NETRA's descriptions).
DESCRIPTION_PROMPT = (
    "For each category below, provide 10-15 short visual descriptions suitable for "
    "CLIP-based image classification. Focus on distinctive visual attributes: color, "
    "shape, texture, size, parts, habitat, and setting.\n\n"
    "Categories: {categories}\n\n"
    'Return ONLY valid JSON: {{"category": ["desc1", ...]}}'
)


# ── Dataset loading (unchanged from the previous script) ─────────────────────

DATASET_CONFIGS = {
    "cifar10": {"module": "torchvision.datasets", "class": "CIFAR10", "split_kwarg": {"train": False}, "download": True},
    "cifar100": {"module": "torchvision.datasets", "class": "CIFAR100", "split_kwarg": {"train": False}, "download": True},
    "flowers102": {"module": "torchvision.datasets", "class": "Flowers102", "split_kwarg": {"split": "test"}, "download": True},
    "dtd": {"module": "torchvision.datasets", "class": "DTD", "split_kwarg": {"split": "test"}, "download": True},
    "food101": {"module": "torchvision.datasets", "class": "Food101", "split_kwarg": {"split": "test"}, "download": True},
    "oxford_pets": {"module": "torchvision.datasets", "class": "OxfordIIITPet", "split_kwarg": {"split": "test"}, "download": True},
    "caltech101": {"module": "torchvision.datasets", "class": "Caltech101", "split_kwarg": {}, "download": True},
    "eurosat": {"module": "torchvision.datasets", "class": "EuroSAT", "split_kwarg": {}, "download": True},
    "fgvc_aircraft": {"module": "torchvision.datasets", "class": "FGVCAircraft", "split_kwarg": {"split": "test"}, "download": True},
    "country211": {"module": "torchvision.datasets", "class": "Country211", "split_kwarg": {"split": "test"}, "download": True},
}

HF_FALLBACKS = {
    "dtd": "descriptive-textures/dtd",
    "oxford_pets": "timm/oxford-iiit-pet",
    "eurosat": "timm/eurosat-rgb",
}


def load_dataset_hf(dataset_name, split="test"):
    from datasets import load_dataset as hf_load
    from PIL import Image as PILImage

    hf_name = HF_FALLBACKS[dataset_name]
    print(f"    Loading {dataset_name} from HuggingFace ({hf_name})...")
    ds = hf_load(hf_name, split=split, trust_remote_code=True)
    label_feature = ds.features.get("label", ds.features.get("labels"))
    if hasattr(label_feature, "names"):
        class_names = label_feature.names
    else:
        class_names = [str(i) for i in range(label_feature.num_classes)]
    class_names = [str(c).replace("_", " ") for c in class_names]

    class HFWrapper:
        def __init__(self, hf_ds, cn):
            self.ds = hf_ds
            self.classes = cn

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[int(idx)]
            img = item.get("image", item.get("img"))
            label = item.get("label", item.get("labels"))
            if not isinstance(img, PILImage.Image):
                img = PILImage.open(img)
            return img.convert("RGB"), label

    return HFWrapper(ds, class_names), class_names


def load_imagefolder(path, classnames_file=None):
    """Any ImageFolder-style tree: ImageNet val (wnid folders), ImageNet-V2
    (folders 0..999), DomainNet domains (345 class folders). Class order is the
    sorted folder order (numeric if all names are integers); --classnames gives
    the display names in that order (JSON list, or one name per line)."""
    import torchvision

    ds = torchvision.datasets.ImageFolder(path)
    folders = ds.classes
    if all(f.lstrip("-").isdigit() for f in folders):   # ImageNet-V2 style
        order = sorted(range(len(folders)), key=lambda i: int(folders[i]))
        remap = {old: new for new, old in enumerate(order)}
        ds.samples = [(p, remap[t]) for p, t in ds.samples]
        ds.targets = [t for _, t in ds.samples]
        folders = [folders[i] for i in order]
    if classnames_file:
        txt = Path(classnames_file).read_text()
        names = json.loads(txt) if txt.lstrip().startswith("[") else [l.strip() for l in txt.splitlines() if l.strip()]
        if isinstance(names, dict):
            names = [names[f] for f in folders]
        assert len(names) == len(folders), f"{classnames_file}: {len(names)} names for {len(folders)} folders"
    else:
        names = [f.replace("_", " ") for f in folders]
    return ds, [str(n) for n in names]


def load_dataset_and_classnames(dataset_name, data_dir="./data", imagefolders=None, classnames=None):
    import importlib

    if imagefolders and dataset_name in imagefolders:
        return load_imagefolder(imagefolders[dataset_name], (classnames or {}).get(dataset_name))
    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config["module"])
    cls = getattr(mod, config["class"])
    kwargs = {"root": data_dir, **config["split_kwarg"]}
    if config.get("download"):
        kwargs["download"] = True
    try:
        dataset = cls(**kwargs)
    except Exception as e:
        try:
            print(f"    Download failed ({type(e).__name__}), trying without download...")
            kwargs["download"] = False
            dataset = cls(**kwargs)
        except Exception as e2:
            if dataset_name in HF_FALLBACKS:
                print("    Torchvision failed, using HuggingFace fallback...")
                split = config["split_kwarg"].get("split", "test")
                return load_dataset_hf(dataset_name, split)
            raise e2

    if hasattr(dataset, "classes") and dataset.classes:
        class_names = dataset.classes
    elif hasattr(dataset, "categories") and dataset.categories:
        class_names = dataset.categories
    elif hasattr(dataset, "_labels"):
        class_names = sorted(set(dataset._labels))
    elif hasattr(dataset, "targets"):
        class_names = [str(i) for i in range(max(dataset.targets) + 1)]
    elif hasattr(dataset, "y"):
        class_names = [str(i) for i in range(max(dataset.y) + 1)]
    else:
        labels = [dataset[i][1] for i in range(min(len(dataset), 1000))]
        class_names = [str(i) for i in range(max(labels) + 1)]
    if isinstance(class_names[0], (list, tuple)):
        class_names = [c[0] if isinstance(c, (list, tuple)) else c for c in class_names]
    class_names = [str(c).replace("_", " ") for c in class_names]
    return dataset, class_names


# ── Descriptions ─────────────────────────────────────────────────────────────

def load_descriptions_json(path, dataset_name, class_names):
    """Load {class: [desc,...]} from a JSON file or a directory of <dataset>.json."""
    p = Path(path)
    if p.is_dir():
        cands = [p / f"{dataset_name}.json", p / dataset_name / "descriptions.json"]
        cands += sorted(p.glob(f"*{dataset_name}*.json"))
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            raise FileNotFoundError(f"No descriptions JSON for {dataset_name} under {path}")
    with open(p) as f:
        data = json.load(f)
    # tolerate {"descriptions": {...}} or {dataset: {...}} wrappers
    for key in ("descriptions", dataset_name):
        if isinstance(data, dict) and key in data and isinstance(data[key], dict):
            data = data[key]
    lower = {k.lower(): v for k, v in data.items()}
    out, missing = {}, []
    for cn in class_names:
        descs = data.get(cn, lower.get(cn.lower()))
        if isinstance(descs, list) and descs:
            out[cn] = [str(d) for d in descs]
        else:
            missing.append(cn)
    if missing:
        logger.warning(f"{dataset_name}: {len(missing)} classes without descriptions in {p} "
                       f"(e.g. {missing[:3]}); they will use templates only")
    print(f"  Loaded descriptions for {len(out)}/{len(class_names)} classes from {p}")
    return out


def generate_descriptions(class_names, llm_model, llm_provider, cache_dir):
    """Generate descriptions with the paper's Appendix D prompt (cached)."""
    cache_path = Path(cache_dir) / f"descriptions_{llm_model.replace('/', '_')}.json"
    cached = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
    missing = [c for c in class_names if c not in cached]
    if not missing:
        return cached

    client = LLMClient(model=llm_model, provider=llm_provider,
                       temperature=0.7, cost_tracker=CostTracker())
    batch_size = 10
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        prompt = DESCRIPTION_PROMPT.format(categories=", ".join(batch))
        try:
            result = client.call_json(prompt=prompt,
                                      system="Generate visual descriptions for CLIP zero-shot classification.",
                                      agent_name="zpe_descs")
            for cn in batch:
                descs = result.get(cn, result.get(cn.lower(), []))
                cached[cn] = [str(d) for d in descs] if isinstance(descs, list) and descs \
                    else [f"a photo of a {cn}"]
        except Exception as e:
            logger.warning(f"Batch failed: {e}")
            for cn in batch:
                cached.setdefault(cn, [f"a photo of a {cn}"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cached, f, indent=2)
    return cached


# ── Encoding ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_all_prompts(model, tokenizer, prompts, device, batch_size=128):
    feats = []
    for s in range(0, len(prompts), batch_size):
        tokens = tokenizer(prompts[s:s + batch_size]).to(device)
        f = model.encode_text(tokens)
        feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_images(model, preprocess, dataset, device, batch_size=256, max_images=None):
    from PIL import Image

    feats, labels = [], []
    n = min(len(dataset), max_images) if max_images else len(dataset)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        imgs = []
        for idx in range(s, e):
            img, label = dataset[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            imgs.append(preprocess(img.convert("RGB")))
            labels.append(label)
        f = model.encode_image(torch.stack(imgs).to(device))
        feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
        if (s // batch_size) % 10 == 0:
            print(f"    Encoded {e}/{n} images...")
    return torch.cat(feats, dim=0), np.array(labels)


@torch.no_grad()
def build_pretrain_features(image_dir, model, preprocess, device, out_path, max_images=20000):
    """Encode a folder of unlabeled images (ZPE uses the first 20k LAION-400M images)."""
    from PIL import Image

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in exts)[:max_images]
    if not paths:
        raise FileNotFoundError(f"No images under {image_dir}")
    feats = []
    for s in range(0, len(paths), 256):
        imgs = []
        for p in paths[s:s + 256]:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
            except Exception as e:
                logger.warning(f"skip {p}: {e}")
        if imgs:
            f = model.encode_image(torch.stack(imgs).to(device))
            feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
        print(f"    {min(s + 256, len(paths))}/{len(paths)}")
    feats = torch.cat(feats, dim=0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": feats, "n": feats.shape[0], "source": str(image_dir)}, out_path)
    print(f"  Saved {feats.shape[0]} pre-training image features to {out_path}")
    return feats


# ── Scoring (the only part that differs between ZPE and the entropy variant) ──

def score_logits_maxlogit(logits, pre_logits=None):
    """ZPE Alg. 2: mean_i max_c (L[i,c] - bias[c]); bias = mean of per-class means."""
    bias = logits.mean(dim=0)
    if pre_logits is not None:
        bias = 0.5 * (bias + pre_logits.mean(dim=0))
    return (logits - bias).max(dim=1).values.mean().item()


def score_logits_entropy(logits, logit_scale):
    """Earlier variant (ZPE*): negative mean entropy of softmax(logit_scale * L)."""
    probs = torch.softmax(logit_scale * logits, dim=1)
    ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    return -ent.mean().item()


class PromptScorer:
    def __init__(self, scoring, image_features, pretrain_features=None, logit_scale=100.0):
        self.scoring = scoring
        self.X = image_features
        self.Xp = pretrain_features
        self.logit_scale = logit_scale

    def score_classifier(self, W):
        """W: (C, dim) class embeddings (rows normalized). Returns scalar score."""
        L = self.X @ W.T
        if self.scoring == "maxlogit":
            Lp = self.Xp @ W.T if self.Xp is not None else None
            return score_logits_maxlogit(L, Lp)
        return score_logits_entropy(L, self.logit_scale)

    def _base_stats(self, base_L, base_Lp):
        """Per-row statistics of the base classifier, cached so that swapping one
        class column costs O(N) instead of O(N*C). Exact for both scorings."""
        key = (id(base_L), id(base_Lp))
        if getattr(self, "_cache_key", None) == key:
            return self._cache
        if self.scoring == "maxlogit":
            bias = base_L.mean(dim=0)
            if base_Lp is not None:
                bias = 0.5 * (bias + base_Lp.mean(dim=0))
            Ln = base_L - bias
            top2 = Ln.topk(2, dim=1)
            self._cache = {"top_val": top2.values, "top_idx": top2.indices}
        else:
            z = (self.logit_scale * base_L).double()
            top2 = z.topk(2, dim=1)
            a, a2 = top2.values[:, :1], top2.values[:, 1:2]       # row max, second max
            idx_max = top2.indices[:, 0]
            e = torch.exp(z - a)
            # sums excluding the argmax column, shifted by the second max (used when the
            # swapped class is the argmax; subtracting its term from Z would cancel badly)
            e2 = torch.exp(z - a2)
            e2.scatter_(1, idx_max[:, None], 0.0)
            self._cache = {"z": z, "a": a, "a2": a2, "idx_max": idx_max, "e": e,
                           "Z": e.sum(dim=1), "T": (e * z).sum(dim=1),
                           "Z2": e2.sum(dim=1), "T2": (e2 * z).sum(dim=1)}
        self._cache_key = key
        return self._cache

    def score_description_swaps(self, base_W, base_L, base_Lp, c_idx, desc_feats):
        """Scores of classifiers equal to base_W with row c_idx replaced by each description."""
        st = self._base_stats(base_L, base_Lp)
        col = self.X @ desc_feats.T                                   # (N, n_descs)
        if self.scoring == "maxlogit":
            new_bias = col.mean(dim=0)
            if base_Lp is not None and self.Xp is not None:
                new_bias = 0.5 * (new_bias + (self.Xp @ desc_feats.T).mean(dim=0))
            coln = col - new_bias
            # row-wise max over the other classes
            m_excl = torch.where(st["top_idx"][:, 0] == c_idx, st["top_val"][:, 1], st["top_val"][:, 0])
            return torch.maximum(m_excl[:, None], coln).mean(dim=0)
        # entropy: remove column c from the per-row sums, add the new column
        z, e = st["z"], st["e"]
        e_c, z_c = e[:, c_idx], z[:, c_idx]
        is_max = st["idx_max"] == c_idx
        a_ex = torch.where(is_max[:, None], st["a2"], st["a"])
        Z_ex = torch.where(is_max, st["Z2"], st["Z"] - e_c)
        T_ex = torch.where(is_max, st["T2"], st["T"] - e_c * z_c)
        zn = (self.logit_scale * col).double()
        en = torch.exp(zn - a_ex)
        Zn, Tn = Z_ex[:, None] + en, T_ex[:, None] + en * zn
        ent = torch.log(Zn) + a_ex - Tn / Zn
        return (-ent.mean(dim=0)).float()


def softmax_weights(scores):
    return torch.softmax(torch.as_tensor(scores, dtype=torch.float32), dim=0)


def evaluate(image_features, labels, class_embeddings):
    preds = (image_features @ class_embeddings.T).argmax(dim=-1).numpy()
    return float((preds == labels).mean())


def normalize_rows(W):
    return W / W.norm(dim=-1, keepdim=True)


# ── One dataset ──────────────────────────────────────────────────────────────

def run_dataset(dataset_name, args, model, preprocess, tokenizer, pretrain_features):
    print(f"\n{'=' * 60}\n  {dataset_name.upper()}\n{'=' * 60}")
    dataset, class_names = load_dataset_and_classnames(dataset_name, args.data_dir, args.imagefolders, args.classnames)
    C = len(class_names)
    print(f"  {len(dataset)} images, {C} classes")

    print("  Encoding images...")
    X, labels = encode_images(model, preprocess, dataset, args.device, max_images=args.max_images)

    templates = IMAGENET_TEMPLATES
    M = len(templates)

    if args.descriptions_json:
        descriptions = load_descriptions_json(args.descriptions_json, dataset_name, class_names)
    else:
        descriptions = generate_descriptions(class_names, args.llm, args.llm_provider,
                                             Path(args.output_dir) / dataset_name)

    print("  Encoding templates (one classifier per template)...")
    T = torch.stack([encode_all_prompts(model, tokenizer, [t.format(c) for c in class_names], args.device)
                     for t in templates])                       # (M, C, dim)
    print("  Encoding descriptions...")
    D = {}
    for c_idx, cn in enumerate(class_names):
        descs = descriptions.get(cn, [])
        if descs:
            D[c_idx] = encode_all_prompts(model, tokenizer, descs, args.device)   # (n_c, dim)
    n_desc_total = sum(v.shape[0] for v in D.values())

    logit_scale = float(model.logit_scale.exp().item()) if hasattr(model, "logit_scale") else 100.0
    scorer = PromptScorer(args.scoring, X, pretrain_features, logit_scale)
    results = {"n_images": int(X.shape[0]), "n_classes": C, "n_descriptions": n_desc_total,
               "scoring": args.scoring,
               "bias": ("E_test+E_pretrain" if pretrain_features is not None else "E_test only")
               if args.scoring == "maxlogit" else "n/a"}

    # 80 templates (uniform)
    W_tmpl = normalize_rows(T.mean(dim=0))
    results["80_templates"] = evaluate(X, labels, W_tmpl)
    print(f"  80 templates (uniform):          {results['80_templates'] * 100:6.2f}%")

    # CuPL+e: uniform mean over all templates + descriptions of each class
    W = T.mean(dim=0) * M
    for c_idx, df in D.items():
        W[c_idx] = (W[c_idx] + df.sum(dim=0)) / (M + df.shape[0])
    for c_idx in range(C):
        if c_idx not in D:
            W[c_idx] = W[c_idx] / M
    results["cupl_e"] = evaluate(X, labels, normalize_rows(W))
    print(f"  CuPL+e (uniform all prompts):    {results['cupl_e'] * 100:6.2f}%")

    # ZPE, templates only
    tmpl_scores = torch.tensor([scorer.score_classifier(T[t]) for t in range(M)])
    w_tmpl = softmax_weights(tmpl_scores)
    W_zpe_tmpl = normalize_rows((w_tmpl[:, None, None] * T).sum(dim=0))
    results["zpe_templates_only"] = evaluate(X, labels, W_zpe_tmpl)
    eff = float(1.0 / (w_tmpl ** 2).sum())
    print(f"  ZPE (templates only):            {results['zpe_templates_only'] * 100:6.2f}%   "
          f"[effective #templates {eff:.1f}/{M}, top: '{templates[int(tmpl_scores.argmax())]}']")
    results["zpe_template_weights"] = {templates[t]: float(w_tmpl[t]) for t in range(M)}

    # ZPE, flat pool (templates + description-swap classifiers), one softmax over all
    base_L = X @ W_tmpl.T
    base_Lp = pretrain_features @ W_tmpl.T if pretrain_features is not None else None
    desc_scores = {c: scorer.score_description_swaps(W_tmpl, base_L, base_Lp, c, df) for c, df in D.items()}
    all_scores = torch.cat([tmpl_scores] + [desc_scores[c] for c in sorted(desc_scores)])
    w_all = softmax_weights(all_scores)
    w_t, w_d = w_all[:M], w_all[M:]
    # closed form of sum_p w_p * classifier_p (template classifiers + swap classifiers)
    W = (w_t[:, None, None] * T).sum(dim=0) + w_d.sum() * W_tmpl
    off = 0
    for c in sorted(desc_scores):
        n = D[c].shape[0]
        wc = w_d[off:off + n]
        W[c] += (wc[:, None] * (D[c] - W_tmpl[c])).sum(dim=0)
        off += n
    results["zpe"] = evaluate(X, labels, normalize_rows(W))
    results["zpe_description_weight_mass"] = float(w_d.sum())
    print(f"  ZPE (flat pool)  -> Table 2:     {results['zpe'] * 100:6.2f}%   "
          f"[total weight on descriptions {float(w_d.sum()):.3f}]")

    # NETRA 55/45 (group-normalized), same descriptions
    alpha, beta = args.alpha, 1.0 - args.alpha
    W = T.mean(dim=0).clone()
    for c, df in D.items():
        W[c] = alpha * W[c] + beta * df.mean(dim=0)
    results[f"netra_{int(alpha * 100)}_{int(beta * 100)}"] = evaluate(X, labels, normalize_rows(W))
    print(f"  NETRA {int(alpha * 100)}/{int(beta * 100)}:                    "
          f"{results[f'netra_{int(alpha * 100)}_{int(beta * 100)}'] * 100:6.2f}%")

    # ZPE within groups + NETRA between groups
    W = W_zpe_tmpl.clone()
    for c, df in D.items():
        wc = softmax_weights(desc_scores[c])
        W[c] = alpha * W_zpe_tmpl[c] + beta * normalize_rows((wc[:, None] * df).sum(dim=0, keepdim=True))[0]
    results["zpe_netra"] = evaluate(X, labels, normalize_rows(W))
    print(f"  ZPE within + NETRA between:      {results['zpe_netra'] * 100:6.2f}%")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ZPE baseline (Table 2)")
    ap.add_argument("--datasets", nargs="*",
                    default=["cifar10", "cifar100", "flowers102", "dtd", "food101",
                             "oxford_pets", "caltech101", "fgvc_aircraft", "eurosat", "country211"])
    ap.add_argument("--clip-model", default="ViT-L/14")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scoring", choices=["maxlogit", "entropy"], default="maxlogit",
                    help="maxlogit = original ZPE (Alg. 2); entropy = earlier ZPE* variant")
    ap.add_argument("--pretrain-features", default=None,
                    help=".pt with CLIP image features of ~20k unlabeled pre-training-like images "
                         "(E_pretrain in ZPE). Omit to use E_test only.")
    ap.add_argument("--build-pretrain-features", default=None, metavar="IMAGE_DIR",
                    help="Encode images in IMAGE_DIR into --pretrain-features and exit")
    ap.add_argument("--descriptions-json", default=None,
                    help="Reuse NETRA's descriptions: JSON {class: [desc,...]} or a dir of <dataset>.json")
    ap.add_argument("--llm", default="gpt-4o")
    ap.add_argument("--llm-provider", default="openai")
    ap.add_argument("--alpha", type=float, default=0.55, help="NETRA template weight (default 55/45)")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--imagefolder", action="append", default=[], metavar="NAME=PATH",
                    help="extra dataset from an ImageFolder tree, e.g. imagenet=/data/imagenet/val, "
                         "imagenet_v2=/data/imagenetv2-matched-frequency, domainnet_real=/data/domainnet/real")
    ap.add_argument("--classnames", action="append", default=[], metavar="NAME=FILE",
                    help="class names for an --imagefolder dataset (JSON list or one per line, in folder order)")
    ap.add_argument("--output-dir", default="experiments/zpe")
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.imagefolders = dict(kv.split("=", 1) for kv in args.imagefolder)
    args.classnames = dict(kv.split("=", 1) for kv in args.classnames)
    for name in args.imagefolders:
        if name not in args.datasets:
            args.datasets.append(name)
    if "gemini" in args.llm.lower():
        args.llm_provider = "google"
    elif "claude" in args.llm.lower():
        args.llm_provider = "anthropic"

    import open_clip
    model_name = args.clip_model.replace("/", "-")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai", device=args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    if args.build_pretrain_features:
        if not args.pretrain_features:
            ap.error("--build-pretrain-features needs --pretrain-features OUT.pt")
        build_pretrain_features(args.build_pretrain_features, model, preprocess, args.device, args.pretrain_features)
        return

    pretrain_features = None
    if args.pretrain_features:
        blob = torch.load(args.pretrain_features, map_location="cpu")
        pretrain_features = blob["features"] if isinstance(blob, dict) else blob
        pretrain_features = normalize_rows(pretrain_features.float())
        print(f"  Using {pretrain_features.shape[0]} pre-training image features for E_pretrain")
    elif args.scoring == "maxlogit":
        print("  NOTE: no --pretrain-features; ZPE bias uses E_test only (paper: slightly weaker than E_test+E_pretrain)")

    print(f"\n{'=' * 60}\n  ZPE BASELINE  scoring={args.scoring}  CLIP={args.clip_model}\n"
          f"  descriptions: {args.descriptions_json or ('generated with ' + args.llm)}\n{'=' * 60}")

    all_results = {}
    for ds in args.datasets:
        try:
            all_results[ds] = run_dataset(ds, args, model, preprocess, tokenizer, pretrain_features)
        except Exception as e:
            logger.error(f"Failed on {ds}: {e}")
            import traceback
            traceback.print_exc()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    netra_key = f"netra_{int(args.alpha * 100)}_{int((1 - args.alpha) * 100)}"
    methods = ["80_templates", "cupl_e", "zpe_templates_only", "zpe", netra_key, "zpe_netra"]

    print(f"\n{'=' * 70}\n  SUMMARY (top-1 %)\n{'=' * 70}")
    print(f"  {'dataset':<14}" + "".join(f"{m:>18}" for m in methods))
    for ds, r in all_results.items():
        print(f"  {ds:<14}" + "".join(f"{r[m] * 100:>18.2f}" for m in methods))

    tag = args.scoring + ("" if pretrain_features is not None or args.scoring != "maxlogit" else "_Etest")
    with open(out_dir / f"zpe_results_{tag}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out_dir / f"zpe_table2_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + list(all_results))
        for m in methods:
            w.writerow([m] + [f"{all_results[ds][m] * 100:.2f}" for ds in all_results])
    print(f"\nSaved {out_dir / f'zpe_results_{tag}.json'} and {out_dir / f'zpe_table2_{tag}.csv'}")
    print("Table 2 'ZPE' row = the 'zpe' line of the CSV.")


if __name__ == "__main__":
    main()
