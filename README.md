<h1 align="center">UltraMatch: Transport Path Routing for Ultra-Fast and Memory-Efficient Image Matching</h1>

<h3 align="center">
  <a href="https://scholar.google.com/citations?user=uWhzrG4AAAAJ&hl=zh-CN">Jiajun Le</a><sup>1</sup> &middot;
  <a href="https://scholar.google.com/citations?user=h-9Ub_cAAAAJ&amp;hl=zh-CN">Yifan Lu</a><sup>1</sup> &middot;
  <a href="https://scholar.google.com/citations?user=bxuEALEAAAAJ&amp;hl=zh-CN">Zizhuo Li</a><sup>1</sup> &middot;
  <a href="https://scholar.google.com/citations?user=LaT72N4AAAAJ&amp;hl=zh-CN&amp;oi=sra">Lei Cao</a><sup>1,2</sup> &middot;
  <a href="https://scholar.google.com/citations?user=WNH2_rgAAAAJ&amp;hl=zh-CN">Junjun Jiang</a><sup>3</sup> &middot;
  <a href="https://scholar.google.com/citations?user=73trMQkAAAAJ&amp;hl=zh-CN&amp;oi=sra">Jiayi Ma</a><sup>1,4,*</sup>
</h3>

<p align="center">
  <sup>1</sup>Electronic Information School, Wuhan University<br>
  <sup>2</sup>Xiaomi Corporation<br>
  <sup>3</sup>School of Computer Science and Technology, Harbin Institute of Technology<br>
  <sup>4</sup>School of Robotics, Wuhan University
</p>

<p align="center">
  <sup>*</sup>Corresponding author
</p>

<!-- Add the public arXiv URL around the badge after announcement. -->
<p align="center">
  <img src="assets/arxiv-badge.svg" alt="arXiv: coming soon" height="24">
  <a href="https://github.com/JiajunLe/UltraMatch"><img src="assets/github-badge.svg" alt="Project: GitHub" height="24"></a>
  <a href="#code-release"><img src="assets/code-badge.svg" alt="Code: coming soon" height="24"></a>
  <a href="#citation"><img src="assets/citation-badge.svg" alt="Citation: BibTeX" height="24"></a>
</p>

<p align="center">
  <b>32.26 ms per image pair &nbsp;&middot;&nbsp; 0.44 GiB peak memory &nbsp;&middot;&nbsp; Up to 6K resolution</b><br>
  <sub>Runtime and memory: MegaDepth-1500. Resolution scalability: ETH3D. Inference on one NVIDIA RTX 3090.</sub>
</p>

## News

- **2026-09-29:** The UltraMatch paper has been submitted to arXiv. The public paper link and arXiv citation will be added after announcement.
- **2026-09-29:** The project overview and main experimental results are available below. Code and pretrained weights have not yet been released.

## Overview

**UltraMatch is an efficient, scalable semi-dense image matcher that routes computation to a small set of promising matching paths.** Instead of constructing a full token-to-token matching matrix, a lightweight **Transport Path Router** ranks candidate target blocks for each source block and retains only a small subset. Matching then runs over the selected paths.

<p align="center">
  <img src="assets/overview.jpg" alt="UltraMatch architecture: reparameterized feature extraction, feature interaction, Transport Path Routing, sparse global Dual-Softmax, and a tiny fine matching head." width="100%">
</p>

The framework combines:

- **Transport Path Routing:** select promising block pairs before token-level matching, reducing the matching search space.
- **Sparse global Dual-Softmax:** preserve global competition across the routed candidates while avoiding the full dense matching matrix.
- **Efficient feature extraction and refinement:** use structural reparameterization and a compact fine matching head with shared parameters.

In the reported experiments, UltraMatch is **1.67x faster than SuperPoint+LightGlue** and **4.35x faster than ELoFTR**, with **0.44 GiB** peak inference memory on MegaDepth-1500. On ETH3D, it supports image pairs at **6048 x 4032** resolution on a single RTX 3090. The routing strategy also transfers to existing semi-dense matchers.

## Results

### Accuracy and efficiency

<p align="center">
  <img src="assets/efficiency.png" alt="Latency versus peak GPU memory on MegaDepth-1500; bubble size indicates relative pose AUC at 5 degrees." width="620">
</p>

Results below reproduce **Table 1** of the paper. Pose AUCs are reported at **5 / 10 / 20 degrees**; higher is better. Runtime and peak GPU memory are measured per image pair on **MegaDepth-1500**, using one **NVIDIA RTX 3090**; lower is better.

| Method | Type | ScanNet-1500 AUC (%) | MegaDepth-1500 AUC (%) | Time (ms) | Memory (GiB) |
| :--- | :--- | :---: | :---: | ---: | ---: |
| SuperPoint+SuperGlue | Sparse | 16.2 / 32.8 / 49.7 | 49.7 / 67.1 / 80.6 | 104.83 | 1.02 |
| SuperPoint+LightGlue | Sparse | 14.8 / 30.8 / 47.5 | 49.9 / 67.0 / 80.1 | 53.96 | 1.02 |
| DKM | Dense | 26.6 / 47.1 / 64.2 | 60.4 / 74.9 / 85.1 | 572.98 | 9.66 |
| RoMa | Dense | 28.9 / 50.4 / 68.3 | 62.6 / 76.7 / 86.3 | 755.44 | 6.79 |
| LoFTR | Semi-dense | 16.9 / 33.6 / 50.6 | 52.8 / 69.2 / 81.2 | 376.15 | 12.19 |
| QuadTree | Semi-dense | 19.0 / 37.3 / 53.5 | 54.6 / 70.5 / 82.2 | 411.74 | 12.20 |
| MatchFormer | Semi-dense | 15.8 / 32.0 / 48.0 | 53.3 / 69.7 / 81.8 | 658.22 | 8.58 |
| ELoFTR | Semi-dense | 19.2 / 37.0 / 53.6 | 56.4 / 72.2 / 83.5 | 140.49 | 8.64 |
| JamMa | Semi-dense | 14.5 / 29.8 / 46.2 | 55.4 / 70.8 / 82.1 | 361.42 | 5.49 |
| EDM | Semi-dense | 19.8 / 37.5 / 54.4 | 57.5 / 73.2 / 84.2 | 83.45 | 8.24 |
| SLiM | Semi-dense | 18.0 / 34.7 / 50.4 | 57.9 / 72.8 / 83.5 | 157.63 | 5.14 |
| **UltraMatch** | **Semi-dense** | **21.1 / 39.8 / 56.6** | **57.4 / 72.5 / 83.6** | **32.26** | **0.44** |

Bold highlights our method. UltraMatch achieves the highest ScanNet AUCs among the listed semi-dense methods, while retaining competitive MegaDepth accuracy.

**Measurement details.** Table 1 uses each method's reported inference settings. UltraMatch's 32.26 ms result uses selective BF16. Under full FP32, it achieves the same MegaDepth AUC@5 of 57.4, with **37.85 ms** latency and **0.57 GiB** peak memory. The paper's Appendix F.3 and Table 10 provide the controlled efficiency comparison and inference configurations.

### High-resolution scalability

<p align="center">
  <img src="assets/inference-runtime.png" alt="Inference runtime as input resolution increases on ETH3D." width="48%">
  <img src="assets/inference-memory.png" alt="Peak inference GPU memory as input resolution increases on ETH3D." width="48%">
</p>

On ETH3D, UltraMatch scales to **6K (6048 x 4032)** with **7.82 GiB** peak inference memory on a single RTX 3090. At **1824 x 1216**, it takes **36.43 ms** and **0.63 GiB**, compared with **604.59 ms** and **18.56 GiB** for ELoFTR under the evaluated settings.

### Transferability of Transport Path Routing

The routing strategy can also accelerate existing matchers. The following results are from **Table 3**. Original and routed variants are trained from scratch using the corresponding official training protocols; these experiments are separate from Table 1.

| Method | AUC@5: original / routed | Time (ms): original / routed | Speedup | Memory (GiB): original / routed |
| :--- | :---: | :---: | ---: | :---: |
| ELoFTR | 54.9 / 55.1 | 141.61 / 76.41 | 1.85x | 8.64 / 2.02 |
| JamMa | 55.7 / 56.0 | 362.64 / 187.05 | 1.94x | 5.49 / 1.43 |
| SLiM | 55.4 / 54.6 | 193.30 / 147.30 | 1.31x | 5.14 / 4.06 |
| EDM | 55.4 / 56.1 | 83.58 / 41.56 | 2.01x | 8.24 / 0.60 |

## Code Release

**Code and pretrained weights have not yet been released.** This repository currently contains the project overview and paper results. Release updates, installation instructions, and usage documentation will be posted here when available.

You can **Watch** this repository for updates. For questions about the project, contact [Jiajun Le](mailto:jiajunle01@gmail.com) or the corresponding author, [Jiayi Ma](mailto:jyma2010@gmail.com).

## Citation

If you find UltraMatch useful in your research, please cite our work. The entry below is provisional; the arXiv identifier and paper URL will be added after the preprint is publicly announced.

```bibtex
@misc{le2026ultramatch,
  title  = {{UltraMatch}: Transport Path Routing for Ultra-Fast and Memory-Efficient Image Matching},
  author = {Le, Jiajun and Lu, Yifan and Li, Zizhuo and Cao, Lei and Jiang, Junjun and Ma, Jiayi},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/JiajunLe/UltraMatch}
}
```
