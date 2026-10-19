# NETRA: Normalized Ensembling of Textual Representations for Zero-Shot Visual Adaptation

Divyansh Bhatia (eBay Inc.) and N. Mehala (Vidyashilp University)

Accepted at the **11th International Conference on Computer Vision and Image Processing (CVIP 2026)**, NIT Calicut, 9–12 December 2026. Proceedings to appear in Springer CCIS.

NETRA is a training-free way to build better CLIP zero-shot classifiers. It fuses the standard hand-crafted template ensemble with LLM-generated class descriptions, weighting the two *groups* rather than the individual prompts:

```
t_c = (α/M) · Σ_m f̂_T(t_m^c)  +  (β/N) · Σ_n f̂_T(d_n^c),      β = 1 − α
```

Averaging all prompts uniformly, as CuPL+e does, gives the N descriptions a fixed share of N/(M+N) of the weight (about 11% for 80 templates and 10 descriptions) whatever their quality. Normalizing within each group removes that cap. One LLM call per dataset, no training, no target images, and no cost at test time beyond standard CLIP inference.

## Main results (CLIP ViT-L/14, GPT-4o descriptions)

With a single fixed weighting of α/β = 55/45 and no tuning:

* improves over the 80-template ensemble on **7 of 10** classification datasets and on **all 11** retrieval datasets;
* best average on 7 large-scale datasets (ImageNet-1K, ImageNet-V2, five DomainNet domains; 580K images), with +0.47% on ImageNet-1K at 85/15 (McNemar p = 3×10⁻⁹);
* +1.6% to +3.6% on static-frame classification of UCF-101, HMDB-51 and Kinetics-400;
* gains of +3.0 ± 0.8% (Flowers102) and +3.5 ± 0.5% (DTD) over templates across five independent description-generation trials;
* 43 to 165 times cheaper than classifying every image with a multimodal model (TLAC-style), and far more accurate on low-resolution images (+15.8% on CIFAR-100).

Results were produced with 7 backbones (OpenAI CLIP ViT-B/32, B/16, L/14; EVA-CLIP; MetaCLIP B/32, L/14; SigLIP B/16) and 10 LLMs (GPT-4o, GPT-5.2, Claude Sonnet 4, Claude Opus 4.5, Gemini 2.5 Flash and Flash Lite, LLaMA-3.3-70B, DeepSeek-V3, Mistral-Small-24B, Qwen2.5-7B). The paper reports which weighting each table uses; per-dataset optimal weightings are selected on the test set and are marked as upper bounds.

## Installation

```bash
git clone https://github.com/dnbresearch/NETRA.git
cd NETRA
pip install -e ".[clip]"          # torch, torchvision, open-clip-torch
pip install anthropic google-genai together   # only the providers you will use
```

Python 3.10 or newer and one GPU with at least 8 GB of memory are enough; all experiments in the paper ran on a single A100 (40 GB).

Set the API key for the LLM you want to generate descriptions with:

```bash
export OPENAI_API_KEY=...        # GPT-4o, GPT-5.2
export ANTHROPIC_API_KEY=...     # Claude
export GOOGLE_API_KEY=...        # Gemini
export TOGETHER_API_KEY=...      # LLaMA, DeepSeek, Mistral, Qwen
```

Descriptions are generated once per dataset and cached under `experiments/`, so a run that reuses a cache makes no API calls. Generating descriptions for all datasets costs under $15 with GPT-4o and under $0.50 with Gemini 2.5 Flash Lite.

Datasets are downloaded through torchvision on first use into `./data` (override with `--data-dir`). ImageNet-1K, ImageNet-V2, DomainNet, the video datasets and MS-COCO must be downloaded separately; the scripts below state the expected layout in their docstrings.

## Quick start

Evaluate NETRA and the baselines on one dataset:

```bash
python scripts/run_baselines.py --dataset dtd --clip-model ViT-L/14 --llm gpt-4o
```

Sweep the weighting α/β on one dataset (Fig. 3 in the paper):

```bash
python scripts/run_weight_ablation.py --dataset dtd --clip-model ViT-L/14 --llm gpt-4o
```

Run the 10-dataset suite with baselines, the weight sweep, and repeated trials:

```bash
python scripts/run_multidataset.py \
    --datasets cifar10 cifar100 flowers102 dtd food101 oxford_pets caltech101 fgvc_aircraft eurosat country211 \
    --clip-model ViT-L/14 --llm gpt-4o
```

## Reproducing the paper

| Paper | Script | Notes |
|---|---|---|
| Table 2, classification on 10 datasets | `scripts/run_baselines.py` (one dataset per call) or `scripts/run_multidataset.py` | Includes CuPL+e, DCLIP, Frolic and CLIP-Enhance from `visprompt/baselines` |
| Table 2, ZPE\* row | `scripts/run_zpe_baseline.py --scoring entropy` | Entropy-scored variant of ZPE; `--scoring maxlogit` (default) runs the published ZPE scoring |
| Table 3, ImageNet-1K / ImageNet-V2 / DomainNet | `scripts/run_imagenet.py --dataset {imagenet,imagenet-v2,domainnet} --domain ...` | |
| Table 4, retrieval | `scripts/run_retrieval_baselines.py --dataset ... --run-ablation`; MS-COCO: `scripts/run_coco_retrieval.py` | |
| Table 5, qualitative examples | per-class outputs of `scripts/run_baselines.py` | |
| Fig. 2, few-shot linear probes | `scripts/run_fewshot_comparison.py --datasets oxford_pets cifar100 dtd flowers102` | |
| Sect. 4.5, action video datasets | `scripts/run_action_recognition.py` (UCF-101), `scripts/run_video_benchmarks.py --dataset {hmdb51,kinetics400}` | Middle frame of each video |
| Tables 6 and 9, TLAC-style baseline | `scripts/run_tlac_baseline.py --dataset ... --mode tlac --lmm-model {gpt-4o,gemini-2.5-flash-lite}`; summarize with `scripts/summarize_tlac.py` | `--mode tlac2` runs the two-stage protocol of the TLAC paper |
| Fig. 4, LLM × backbone | `scripts/run_cross_ablation.py --datasets ... --backbones ... --llms ...` | |
| Fig. 5, per-class analysis | `scripts/run_action_perclass.py`, per-class outputs of `run_baselines.py` | |
| Sect. 5.4, variance over trials | `scripts/run_variance.py --dataset ... --n-trials 5` | |
| Sect. 5.5, fixed default and leave-one-out | `scripts/run_fixed_default_trials.py` | |
| Table 7, template-count sensitivity | `scripts/run_theory_verification.py` | |
| Table 8, LLM ablation | `scripts/run_cross_ablation.py --include-opensource --include-gemini` | Open-source models are served through Together AI |
| Sect. 4.2, McNemar test on ImageNet | `scripts/run_mcnemar_imagenet.py` | |
| Appendix A, timing | `scripts/run_timing.py` | |

Every script writes JSON results under `experiments/` and accepts `--help`.

Two details that matter when comparing numbers with the paper:

* The 10 small datasets are evaluated on fixed test subsets whose sizes are given in Table 6 (the `--val-size` defaults in `scripts/run.py`; subsampling uses seed 42).
* CLIP ViT-L/14 is loaded with OpenCLIP's `ViT-L-14` configuration and the `openai` weights. OpenCLIP warns that this configuration uses GELU where the OpenAI weights were trained with QuickGELU; absolute accuracies are therefore slightly below those in the CLIP paper (72.86% against 75.3% on ImageNet-1K with 80 templates). All methods in the paper share this setting.

## Repository layout

```
visprompt/
  task_spec.py            dataset and class-name specification (TaskSpec)
  baselines/              template ensembles, CuPL, DCLIP, WaffleCLIP, CLIP-Enhance, Frolic, ZPE
  tasks/                  classification, retrieval, and (experimental) segmentation/detection runners
  utils/llm.py            LLM client for OpenAI, Anthropic, Google and Together, with cost tracking
  utils/metrics.py        accuracy, mAP, per-class metrics
  _experimental/          earlier multi-agent prompt-search prototype; not used in the paper
scripts/
  run.py                  builds the TaskSpec and runner used by every script
  run_*.py                one script per experiment (see the table above)
configs/                  default and task configurations
```

## Citation

```bibtex
@inproceedings{bhatia2026netra,
  title     = {{NETRA}: Normalized Ensembling of Textual Representations for Zero-Shot Visual Adaptation},
  author    = {Bhatia, Divyansh and Mehala, N.},
  booktitle = {Computer Vision and Image Processing (CVIP 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026}
}
```

## License

MIT. See `LICENSE`.
