python -u quantize_sample.py \
    --fraction 0.1 --dataset ImageNet --data_path /home/data/dq/imagenet-1k/data/ \
    --num_exp 10 --workers 10 -se 0 --selection Submodular --model ViT_Base_16 \
    -sp ./results/bin_imagenet_010 \
    --batch 128 --submodular GraphCut --submodular_greedy NaiveGreedy --pretrained
