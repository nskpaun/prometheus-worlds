# Upscaler shortlist (for audit)

Researched 2026-11-17. Nothing was downloaded, installed or run. Every fact below was read from the
primary page (the repository, the model card, or the GitHub / Hugging Face / PyPI metadata API) on that day,
unless it is marked **estimate** or **not verified**.

## The short answer

| Rank | Model | Why | Fits 2 minutes on this Mac? |
|---|---|---|---|
| 1 | **Real-ESRGAN** (official weights, MLX port) | Small, BSD-3, safetensors, runs on MLX with no PyTorch, measured speed on Apple Silicon | Yes (estimate: 10 s to 2 min for 40 MP, depending on variant) |
| 2 | **SeedVR2 3B** through mflux | The only one that really invents fine texture; Apache-2.0; safetensors; **mflux 0.24.0 in our paint environment already contains it, so no new Python package** | No for everything; yes for about 10-15 MP of chosen surfaces (estimate) |
| 3 | **2xPublic_realplksr_dysample_layernorm_real** (Philip Hofmann) through Spandrel | A newer small 2x network, Apache-2.0, safetensors, native 2x (which is the 4x-more-pixels we need) | Probably yes, but no Mac speed is published anywhere |

The honest trade: rank 1 is safe and fast but mostly sharpens; rank 2 adds believable detail but is 3 to 5
times over the time budget if applied to every surface. A sensible plan is rank 1 everywhere and rank 2 only
on the few surfaces nearest the viewer.

## What we need, in numbers

- 4x more pixels per surface means **2x per side**. A 4x model gives 16x the pixels, which is more than
  asked; its output can be halved, or a native 2x model used.
- Budget: about 2 minutes for 20-40 megapixels (MP) of output. That is **3 to 6 seconds per megapixel**, or
  **12 to 25 seconds per 2048x2048 output**.
- This Mac is an **Apple M4 Pro with 64 GB**. All published Apple Silicon timings found are from faster
  chips (M5 Max, M2 Ultra). Where a number for this Mac is given below it is an **estimate**: the published
  time multiplied by about 2. Measure before trusting it.

## Sources of measured Apple Silicon speed

Only two primary sources report real timings on Macs.

**Real-ESRGAN on MLX**, from `docs/REPORT.md` in xocialize/realesrgan-mlx (M5 Max, fp16, GPU, mean of 3 runs):

| Model | Input to output | Tile | Time | Peak memory |
|---|---|---|---|---|
| RealESRGAN_x4plus | 256 to 1024 | none | 1.36 s | 4.6 GB |
| RealESRGAN_x2plus | 256 to 512 | none | 0.47 s | 1.5 GB |
| realesr-general-x4v3 | 256 to 1024 | none | 0.12 s | 1.8 GB |
| RealESRGAN_x4plus | 1024 to 4096 | none | 17.1 s | 15.9 GB |
| RealESRGAN_x4plus | 1024 to 4096 | 256 | 26.0 s | 5.3 GB |
| RealESRGAN_x4plus | 1024 to 4096 | 128 | 39.5 s | 2.1 GB |

**SeedVR2 on MLX**, from the lvladikov/SeedVR2-1.4B model card (M2 Ultra 64 GB, model load included):

| Job | 1.4B distilled | Official 3B | Official 7B |
|---|---|---|---|
| 512 to 2048 | 20.2-22.5 s, 4.6 GB | 25.6 s, 8.3 GB | 33-38 s, 14-16 GB |
| 512 to 4096 | 54-55 s, 7.5 GB | 72.5 s | 257-307 s (ran out of memory headroom) |
| 2048 to 8192 (67 MP) | 204 s | 288 s | not given |

These SeedVR2 numbers come from one person's card, not from ByteDance or mflux. Treat them as a guide.

## Serious candidates

### A. Real-ESRGAN (official weights)

| | |
|---|---|
| Author | Xintao Wang and others (Tencent ARC) |
| Code | https://github.com/xinntao/Real-ESRGAN, BSD-3-Clause. Last push 2024-08; unmaintained but stable |
| Original weights (pickle `.pth`) | `RealESRGAN_x4plus.pth` 67.0 MB: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth<br>`RealESRGAN_x2plus.pth` 67.1 MB: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth<br>`realesr-general-x4v3.pth` 4.9 MB (plus `realesr-general-wdn-x4v3.pth`): https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth |
| MLX weights (safetensors, fp16) | https://huggingface.co/mlx-community/Real-ESRGAN-x4plus (33.5 MB, rev `16c284e716ee`)<br>https://huggingface.co/mlx-community/Real-ESRGAN-x2plus (33.5 MB, rev `c42bece4e3bf`)<br>https://huggingface.co/mlx-community/Real-ESRGAN-general-x4v3 (2 files of 2.4 MB, rev `e9a382fa779f`) |
| Weights licence | BSD-3-Clause tag on the MLX cards, "inherited from upstream". The upstream repo has one BSD-3 licence and no separate weights licence. Training data was DIV2K, Flickr2K and OST (research datasets); that is the usual grey area for every model on this page |
| Parameters | x4plus and x2plus: 16.7 M (RRDBNet, 23 blocks). general-x4v3: 1.21 M (SRVGGNetCompact) |
| MLX runner | https://github.com/xocialize/realesrgan-mlx ("MVS Collective"), commit `52c0fc1` of 2026-06-15, 7 commits, 7 stars. Needs only `mlx`, `numpy`, `pillow`, `huggingface_hub`, all already in `runtime/paint.lock` |
| Speed for a 2048x2048 output | M5 Max, derived from the table above: x4plus about 4-6 s, general-x4v3 about 0.5 s. **Estimate for M4 Pro**: x4plus 9-15 s, general-x4v3 1-1.5 s |
| Memory | Tile 256 keeps x4plus near 5 GB. Untiled 4096 output needs 16 GB |
| Texture | x4plus invents a little (grain, foliage, fabric) and cleans edges. general-x4v3 mostly sharpens; upstream says "not too strong deblur and denoise capacity" |
| Known artefacts | Smooth, waxy "plastic" surfaces and over-flattened fine texture; painterly look on noise. The ncnn README warns that tiling "may introduce block inconsistency". It does not invent readable text |
| Other ways to run | PyTorch MPS through Spandrel; ncnn-vulkan binary (2022 build, licence metadata unclear); Core ML package https://huggingface.co/mlboydaisuke/Real-ESRGAN-x4-CoreML (BSD-3, 59 MB, fixed 512 to 2048, about 500 MB RAM, Swift only, no speed given) |

Flags for the audit:
- **`realesrgan-mlx` is not on PyPI** (the name returns 404), although the model cards say `pip install realesrgan-mlx`. It would have to be installed from the git commit or copied in.
- **The runner repository has no LICENSE file.** `pyproject.toml` and the README say BSD-3-Clause, but the licence text itself is missing.
- **It downloads weights by itself on first use** from `mlx-community/Real-ESRGAN-<variant>` with no pinned revision. Our setup pins and checksums everything, so the weights should be fetched by `setup.sh` and passed in as a local path.
- The MLX weights are a conversion by a small account. The card reports parity with PyTorch of 3.8e-6 (fp32) and 1.3e-3 (fp16); this is the author's claim.

### B. SeedVR2 (one-step diffusion)

| | |
|---|---|
| Author | ByteDance Seed. Paper: "SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training" (ICLR 2026) |
| Code | https://github.com/ByteDance-Seed/SeedVR, Apache-2.0. The official code needs CUDA, flash-attn and apex; not usable on a Mac as is |
| Original weights | https://huggingface.co/ByteDance-Seed/SeedVR2-3B, Apache-2.0 tag. **Pickle**: `seedvr2_ema_3b.pth` 13.6 GB, `ema_vae.pth` 1.0 GB |
| Weights mflux uses (safetensors) | https://huggingface.co/numz/SeedVR2_comfyUI, Apache-2.0 tag, rev `09ced7102363`. `seedvr2_ema_3b_fp16.safetensors` 6.78 GB + `ema_vae_fp16.safetensors` 501 MB. (7B: 16.5 GB.) This is a third-party fp16 re-save by the ComfyUI node author, not ByteDance's own file |
| Parameters | 3B: 3,391,476,448. "7B": 8,239,608,232 (both as counted on the 1.4B card; ByteDance gives no figure) |
| How it runs here | **Already present**: `spike/.venv` has mflux 0.24.0 (MIT) with `mflux/models/seedvr2` and the `mflux-upscale-seedvr2` command. Python: `SeedVR2(model_config=ModelConfig.seedvr2_3b()).generate_image(seed, image_path, resolution, softness)`. No text prompt. No new Python dependency |
| Speed for a 2048x2048 output | M2 Ultra: 25.6 s for 3B including load (card above). **Estimate for M4 Pro: 45-50 s.** At large sizes 3B ran at 4.3 s/MP on the M2 Ultra; **estimate 7-9 s/MP here**, so 20 MP is about 2.5-3 minutes and 40 MP about 5-6 minutes |
| Memory | 8.3 GB peak for 3B at 2048 on MLX (card). mflux tiles only the VAE; the transformer sees the whole image, so memory grows with size. mflux offers `--low-ram` and `--mlx-cache-limit-gb` |
| Texture | Yes, it generates plausible fine detail (pores, bark, weave), not just edges |
| Known artefacts | The authors' own words: "tend to overly generate details on inputs with very light degradations", "oversharpened results occasionally (especially on small resolutions, e.g., 480p)", and "not robust to heavy degradations". Clean AI paint is exactly the "very light degradation" case; mflux's `softness` setting (0 to 1, pre-blurs the input) exists for this. Because it is generative, two neighbouring surfaces upscaled separately may get different grain at their shared edge. Hallucinated lettering is possible on signs and labels (not confirmed on a primary page) |

Variants seen:
- **SeedVR2-1.4B** distilled by one person: https://huggingface.co/lvladikov/SeedVR2-1.4B, Apache-2.0 tag, safetensors 2.89 GB + VAE 501 MB, 1,442,608,252 parameters, rev `884a50f7072d`. Faster (3 s/MP on M2 Ultra, 2.1 s/MP with its small decoder) and 4.6 GB peak. Its own card lists: weaker fine detail than 3B at 4x, **tile seams above about 8 MP**, edge overshoot. **mflux 0.24.0 does not know this variant** (its weight table has only 3B and 7B), so it needs custom loading code. Zero downloads recorded. Interesting, not ready.
- MLX-format copies: https://huggingface.co/mlx-community/SeedVR2-3B-mlx (7.9 GB, Apache-2.0) and `-int8` (4.2 GB); they target a Swift package that its card says is archived. Not needed, since mflux converts the numz files itself.

### C. Community models run through Spandrel (PyTorch MPS)

Spandrel (https://github.com/chaiNNer-org/spandrel, MIT, v0.4.2 on PyPI, 2026-02) is the loader that
ComfyUI and chaiNNer use. It reads safetensors for every supported architecture and its README says to
prefer them: pickle is "inherently unsafe". Its requirements are `torch`, `torchvision`, `safetensors`,
`numpy`, `einops`. Our paint environment already has torch 2.15, safetensors and numpy; **new would be
spandrel, torchvision and einops**. Its README mentions only CUDA; MPS use is common but **not verified** on the page.

All models below are by Philip Hofmann ("Phhofm" / "Helaman"), a hobbyist trainer. The repository
https://github.com/Phhofm/models has **no LICENSE file**; each release note states its own licence, and
several spell it "CC-BY-0.4" (a typo for CC-BY-4.0, which OpenModelDB shows).

| Model | Release URL | Licence (per release note) | File | Type | Notes |
|---|---|---|---|---|---|
| **2xPublic_realplksr_dysample_layernorm_real** | https://github.com/Phhofm/models/releases/tag/2xPublic_realplksr_dysample_layernorm_real | Apache-2.0 | safetensors 29.5 MB (about 7.4 M parameters, derived from file size) | RealPLKSR, a plain convolutional network, native 2x | 2025-06. Trained on only 4,297 tiles from public-licence images; the author says the dataset is small. Handles mild blur, noise, JPEG |
| 4xNomosWebPhoto_RealPLKSR | https://github.com/Phhofm/models/releases/tag/4xNomosWebPhoto_RealPLKSR | CC-BY-4.0 | safetensors 29.6 MB | RealPLKSR 4x | 2024-05. Photo restoration; trained on Nomos-v2 (6,000 images, dataset licence not checked) |
| 2xLexicaRRDBNet (and `_Sharp`) | https://github.com/Phhofm/models/releases/tag/2xLexicaRRDBNet | CC-BY-4.0 | safetensors 33.5 MB | ESRGAN 2x | Made for AI-generated images, but trained on 43,856 images scraped from lexica.art. Provenance is a concern for an open source app |
| 4xNomos2_hq_drct-l | https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_drct-l | CC-BY-4.0 | safetensors 242.5 MB | DRCT-L transformer 4x | For clean input. Transformers are slow on MPS (**estimate**: well over 25 s per 2048 output) |
| 4xBHI_dat2_real | https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_real | CC-BY-4.0 | safetensors 139.8 MB | DAT2 transformer 4x | Same speed concern |
| 4xTextures_GTAV_rgt-s_dither | https://github.com/Phhofm/models/releases/tag/4xTextures_GTAV_rgt-s_dither | CC-BY-4.0 | safetensors 135.6 MB | RGT-S transformer 4x | Aimed at game textures, but **trained on textures extracted from GTA V**. Do not ship |

Common points: these sharpen and add mild texture like Real-ESRGAN, generally with less of the plastic
look on the newer architectures (community opinion, **not verified**). No Mac speed is published for any of
them. CC-BY-4.0 requires crediting the author in the app. Whether Spandrel 0.4.2 loads the "dysample +
layernorm" RealPLKSR variant is **not verified** (its README lists "PLKSR and RealPLKSR" only).

## Looked at and set aside

| Model | Repository | Licence | Why not |
|---|---|---|---|
| SUPIR | https://github.com/Fanghua-Yu/SUPIR | README: "strictly for non-commercial purposes" | **Non-commercial.** Also needs SDXL (6.9 GB) plus a 13B LLaVA; far over budget |
| StableSR | https://github.com/IceClear/StableSR | NTU S-Lab License 1.0 | **Non-commercial** (same licence family as InvSR). `.ckpt` pickle. 4 to 200 steps |
| InvSR | https://github.com/zsyOAOA/InvSR | NTU S-Lab License 1.0; LICENSE says "use for non-commercial purpose" | **Non-commercial**, and built on SD-Turbo (Stability's own restricted licence, not re-read). `.pth` pickle |
| HYPIR | https://github.com/XPixelGroup/HYPIR | README: "strictly for non-commercial purposes" | **Non-commercial.** `.pth` LoRA on SD 2.1 |
| OSEDiff | https://github.com/cswry/OSEDiff | Apache-2.0 code | Weights are **`.pkl` pickle** on Google Drive; needs SD 2.1-base (now linked from a third-party mirror, `Manojb/stable-diffusion-2-1-base`) and the RAM tagger. Three extra models |
| TSD-SR | https://github.com/Microtreei/TSD-SR | Apache-2.0 code | Needs Stable Diffusion 3 Medium (Stability community licence, not re-read); weights on Google Drive / OneDrive, no weights licence stated |
| FluxSR | reported at comfyui-wiki.com, repo not checked | **not verified** | LoRA on FLUX.1-dev, whose licence is non-commercial. 12B base model, far over budget |
| AuraSR v2 | https://huggingface.co/fal/AuraSR-v2 | **Unclear**: model card says Apache-2.0, the code repo https://github.com/fal-ai/aura-sr says CC-BY-SA-4.0, the v1 card says "cc" | 600 M parameter GAN, 2.47 GB safetensors (a pickle `model.ckpt` sits beside it). Made for generated images and does invent texture. README: tiles "can result in seams"; the overlapped mode doubles the time. No Mac speed. Worth a second look only if the licence is cleared |
| HAT (Real_HAT_GAN_SRx4) | https://github.com/XPixelGroup/HAT | Apache-2.0 | Weights only as `.pth` pickle on Google Drive / Baidu. 20.8 M parameter transformer, slow on MPS (**estimate**) |
| DAT | https://github.com/zhengchen1999/DAT | Apache-2.0 | Official weights are fidelity-trained (soft output); `.pth`. Community GAN versions listed in C |
| DRCT (Real-DRCT-GAN) | https://github.com/ming053l/DRCT | MIT, no weights licence stated | `.pth` on Google Drive; 14-28 M parameter transformer |
| SwinIR | https://github.com/JingyunLiang/SwinIR | Apache-2.0 | `.pth`; 11.9 M parameter transformer; superseded |
| Swin2SR | https://github.com/mv-lab/swin2sr, https://huggingface.co/caidas/swin2SR-classical-sr-x2-64 | Apache-2.0 | 48.5 MB safetensors for the classical x2/x4 models, but those only sharpen clean bicubic input. The real-world x4 model is pickle `.bin` only. Would add `transformers` usage plus torch |
| SPAN | https://github.com/hongyuanyu/SPAN | Apache-2.0 | Tiny and fast, fidelity-trained; no advantage over realesr-general-x4v3 without a community GAN checkpoint |
| EfRLFN | https://github.com/EvgeneyBogatyrev/EfRLFN, https://huggingface.co/mlx-community/EfRLFN-x4 | MIT | 504 k parameters, 2 MB safetensors, MLX port claims 270x480 to 1080x1920 in about 0.06 s. Built for streaming video; whether it adds any texture is not stated. The `efrlfn_mlx` package is not on PyPI. A possible speed floor, nothing more |
| Thera | https://github.com/prs-eth/thera, https://huggingface.co/mlmPenguin/thera-mlx | Apache-2.0 | Any-scale, fidelity-trained (soft). The MLX repo lists scripts but no weight files in its metadata |
| PiperSR | https://github.com/modelpiper/pipersr | **AGPL-3.0** | Copyleft that would bind the whole app |
| 4x-UltraSharp | https://openmodeldb.info/models/4x-UltraSharp | **CC-BY-NC-SA-4.0** | Non-commercial; `.pth` on Mega |
| Z-Image Turbo "Tile" ControlNet | https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 | Apache-2.0 | Same family as the ControlNet we already use; the Tile file is "trained ... (up to 2048x2048) for super-resolution", 8 steps. It repaints with the painter itself, so it costs painter time per megapixel, which is the budget we are already out of |

## Ranked top 3

1. **Real-ESRGAN on MLX.** Start with `RealESRGAN_x2plus` (native 2x, 33.5 MB) and keep
   `realesr-general-x4v3` (2.4 MB) as the fast fallback. Reasons: clear BSD-3 licence from a known lab,
   safetensors, the smallest files, the only candidate with measured Apple Silicon numbers, no PyTorch at
   run time, and its only requirements are packages we already lock. It is a plain convolutional network, so
   the same input always gives the same output and tiles with padding join cleanly. Weakness: it sharpens more
   than it invents, and can look waxy.
2. **SeedVR2 3B through mflux.** Reasons: the best detail of anything here, Apache-2.0, safetensors,
   prompt-free, and it adds no package because mflux 0.24.0 is already installed. Weaknesses: 7.3 GB more
   to download, several times over the time budget if used everywhere, and the authors themselves warn about
   over-generated detail on clean input.
3. **2xPublic_realplksr_dysample_layernorm_real through Spandrel.** Reasons: Apache-2.0, safetensors,
   native 2x, a newer architecture than Real-ESRGAN at half the size, trained on images chosen for their
   licences. Weaknesses: one hobbyist author, small training set, no published Mac speed, and three new
   packages (spandrel, torchvision, einops). Only worth the audit if rank 1 looks too plastic in a side-by-side.

Before any of them: compare against plain Lanczos resize plus a light sharpen. It is free, and it is the
bar a model has to clear.

## Time budget

Budget is 120 s for 20-40 MP of output. All M4 Pro figures are **estimates** (published time x 2).

| Option | Per 2048x2048 output | 20 MP | 40 MP | Verdict |
|---|---|---|---|---|
| realesr-general-x4v3 (MLX) | 1-1.5 s | 5-8 s | 10-15 s | Far inside budget |
| RealESRGAN x2plus / x4plus (MLX, tile 256) | 9-15 s | 40-70 s | 80-140 s | Inside at 20 MP, at the edge at 40 MP |
| SeedVR2 1.4B (custom loader) | 20-40 s | 80-120 s | 160-240 s | Edge at 20 MP, over at 40 MP |
| SeedVR2 3B (mflux) | 45-50 s | 150-180 s | 300-360 s | Over; about 12-15 MP fits in 2 minutes |

Suggested split: give the upscaler **60-90 s** with Real-ESRGAN on every surface, or **about 20 s** for
Real-ESRGAN (compact model) everywhere plus **about 100 s** of SeedVR2 3B on the two or three surfaces
closest to the starting viewpoint (roughly 12 MP). Loading SeedVR2 (7.3 GB) also competes for memory with
the painter, so it should run after the painter has been released.

## Audit checklist

**1. Real-ESRGAN (MLX)**
- Read the runner at commit `52c0fc1`: it is small (three packages: `realesrgan_mlx`, `.archs`, `.utils`). Confirm it does nothing but load safetensors, run the network and write the image. Find and disable the automatic Hugging Face download.
- No LICENSE file in the runner repo and not on PyPI: decide whether to ask the author to add one, or to write our own loader (the two network definitions are short) against the official architecture.
- Confirm the MLX safetensors match the official weights: convert `RealESRGAN_x2plus.pth` once in a throwaway environment and compare tensors, or accept the card's parity claim. Pin the Hugging Face revision and SHA-256 in `setup.sh` like the other models.
- Run it on five of our own painted surfaces at 2x: look for waxy flat areas, lost brush grain, and changed colour or brightness (compare the mean colour before and after).
- Upscale two surfaces that share an edge and one that wraps around; check the join. Check tile borders at tile 256.
- Time 20 MP on the M4 Pro and note peak memory while the painter is still loaded.

**2. SeedVR2 3B (mflux)**
- The weights come from `numz/SeedVR2_comfyUI`, a third party's fp16 copy, not from ByteDance. Decide whether that is acceptable, pin rev `09ced7102363` and its checksums. (The benc0 card claims the 7B copy matched ByteDance's original tensor for tensor; that check was not repeated for 3B.)
- Read `mflux/models/seedvr2` in our installed copy: confirm it loads only the two safetensors files and makes no other network call.
- Run it on the same five surfaces with `softness` 0, 0.25 and 0.5. Look for crunchy over-sharpening, invented patterns that were not in the paint, and garbled lettering on anything with signs or labels.
- Upscale two neighbouring surfaces separately with the same seed and inspect the shared edge for a change in grain.
- Check colour drift (mflux describes a colour-correction step; confirm it is on).
- Time one 2048 output and one 4096 output on the M4 Pro, with model load, and note peak memory. Confirm the painter is unloaded first.

**3. 2xPublic RealPLKSR (Spandrel)**
- Read the release note in full, including its dataset list, and confirm the Apache-2.0 statement covers the weights. The repository itself has no licence file; keep a copy of the release page.
- Download only the `.safetensors` file, never the `.pth`. Pin its SHA-256.
- Confirm Spandrel 0.4.2 recognises this RealPLKSR variant, and that it runs on MPS without falling back to the CPU. If not, the ONNX file in the same release is a fallback (it would need onnxruntime instead).
- Review the three new packages (spandrel, torchvision, einops) and add them to `paint.lock` with hashes.
- Same picture tests as rank 1, side by side with Real-ESRGAN x2plus. Keep it only if it is visibly better.
- Time 20 MP on the M4 Pro.

## Could not verify

- Any speed on an M4 Pro. Every Mac timing here is from an M5 Max or M2 Ultra and scaled by guess.
- Whether SeedVR2 hallucinates text; no primary page says so.
- Parameter counts for the community models (derived from file size where given).
- The full text of the NTU S-Lab licence for StableSR (InvSR's copy was checked and says non-commercial), and the licences of SD-Turbo, SD3 Medium and FLUX.1-dev (stated from memory as restricted).
- The FluxSR repository and licence (seen only in a news summary).
- How the 1.4B SeedVR2 card ran its MLX benchmarks; mflux 0.24.0 has no entry for that model.
- Dataset licences behind any of the weights.
