#/bin/bash
python ./train_edge_detection.py \
 train_edge_detection.sh \
 0,1 \
 50 \
 /path/to/SBD \
 SBD \
 10 \
 4 \
 RW \
 1 \
 10 \
 0.25 \
 Gini \
 edge_groundtruth_file_path \
 8 \
 1e-3 \
 5e-4 \
 0.9 \
 100 \
 1 \
 20 \
 BCE
