# Category-agnostic Semantic Edge Detection by Measuring Neural Representation Randomness
Zhiyi Pan, Peng Jiang, Qiong Zeng, Ge Li, Changhe Tu

### Abstract

Edge detection plays a fundamental role in computer vision tasks and gains wide applications. In particular, semantic edge detection recently draws more attention due to the high demand for a fine-grained understanding of visual scenes. However, detecting high-level semantic edges hidden in visual scenes is quite challenging. Existing semantic edge detection methods focus on category-aware semantic edges and require elaborate category annotations. Instead, we first propose the **category-agnostic semantic edge detection** task without additional semantic category annotations. To achieve this goal, we propose to utilize only edge position annotations and leverage the information randomness of semantic edges. Specifically, we align semantic edge positions to the ground truth by maximizing randomness on edge regions and minimizing randomness on non-edge regions in the training process. In the inference process, we first obtain neural representations by the trained network, and then generate semantic edges by measuring neural randomness. We evaluate our method by comparisons with alternative methods on two well-known datasets: Cityscapes and SBD. The results demonstrate our superiority over the alternatives, which is more significant under weak annotations. We also provide comprehensive mechanism studies to verify the generalizability, rationality, and validity of our working mechanism.

### Instruction

All the computations are carried out on NVIDIA TITAN RTX GPUs.

##### Environment

```
pip install -r requirements.txt
```

##### Datasets

Semantic Boundaries Dataset (SBD) can be download at [SBD](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz), Cityscapes can be download at [Cityscapes](https://www.cityscapes-dataset.com/).


##### Train for neural representation extraction

Please modify the dataset file path in **train_edge_detection.sh** and run:

```
sh train_edge_detection.sh
```

##### Evaluate by randomness based semantic edge generation

Please modify the model file path and save path in **evaluate.sh** and run: 

```
sh evaluate.sh
```

### Experimental Results

| Dataset    | Method         | OIS       | ODS       | AP        | Checkpoints                                                  |
| ---------- | -------------- | --------- | --------- | --------- | ------------------------------------------------------------ |
| SBD        | Baseline (BCE) | 0.549     | 0.527     | 0.533     | [Checkpoint](https://disk.pku.edu.cn/link/AA6C28F370E4B242EA8305DFF7AF791819) |
| SBD        | MNRR (Ent.)    | **0.609** | **0.587** | **0.674** | [Checkpoint](https://disk.pku.edu.cn/link/AA77BECA73D97A47D4985EF081ADC1791C) |
| SBD        | MNRR (Gini.)   | 0.609     | 0.586     | 0.669     | [Checkpoint](https://disk.pku.edu.cn/link/AA89ED5A16C1414583BDBA74336CB9C5BA) |
| Cityscapes | Baseline (BCE) | 0.619     | 0.616     | 0.630     | [Checkpoint](https://disk.pku.edu.cn/link/AAC3F09820996C4AFC99C4723733910ED3) |
| Cityscapes | MNRR (Ent.)    | **0.675** | **0.670** | 0.751     | [Checkpoint](https://disk.pku.edu.cn/link/AAB09692BD6F0B4AEEBC57F57C0D00C039) |
| Cityscapes | MNRR (Gini.)   | 0.658     | 0.662     | **0.773** | [Checkpoint](https://disk.pku.edu.cn/link/AA606F59B59CFE45CC83262AA9AAF2B56D) |

### Visualization

Visual comparison for semantic edge detection on SBD.

![](/Visualization/SBD.png)

The edge detection sequence on the video clip of Cityscapes.

![](/Visualization/video.gif)

