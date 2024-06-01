# Semantic Edge Detection by Measuring Neural Representation Randomness

Zhiyi Pan, Peng Jiang, Qiong Zeng, Ge Li, Changhe Tu

#### Abstract

Edge detection plays a fundamental role in computer vision tasks and gains wide applications. In particular, semantic edge detection recently draws more attention due to the high demand for a fine-grained understanding of visual scenes. However, detecting high-level semantic edges hidden in visual scenes is quite challenging. Existing semantic edge detection methods focus on category-aware semantic edges and require elaborate category annotations. Instead, we first propose the **category-agnostic semantic edge detection** task without additional semantic category annotations. To achieve this goal, we propose to utilize only edge position annotations and leverage the information randomness of semantic edges. Specifically, we align semantic edge positions to the ground truth by maximizing randomness on edge regions and minimizing randomness on non-edge regions in the training process. In the inference process, we first obtain neural representations by the trained network, and then generate semantic edges by measuring neural randomness. We evaluate our method by comparisons with alternative methods on two well-known datasets: Cityscapes and SBD. The results demonstrate our superiority over the alternatives, which is more significant under weak annotations. We also provide comprehensive mechanism studies to verify the generalizability, rationality, and validity of our working mechanism.

#### Instruction

All the computations are carried out on NVIDIA TITAN RTX GPUs.

##### environment

```
pip install -r requirements.txt
```

##### train for neural representation extraction

Please modify the dataset file path in **train_edge_detection.sh** and run:

```
sh train_edge_detection.sh
```

##### evaluate by randomness based semantic edge generation

Please modify the model file path and save path in **evaluate.sh** and run: 

```
sh evaluate.sh
```

#### Visualization

Visual comparison for semantic edge detection on SBD.

![](\Visualization\SBD.png)

The edge detection sequence on the video clip of Cityscapes.

![](\Visualization\video.gif)

