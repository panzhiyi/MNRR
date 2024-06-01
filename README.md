# Semantic Edge Detection by Measuring Neural Representation Randomness

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
