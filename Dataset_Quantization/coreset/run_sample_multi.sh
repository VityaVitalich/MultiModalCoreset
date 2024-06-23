EMBED_PATH='/home/cache/data/dq/clevr/train/multimae_embeds.npy'
SAVE_PATH='/home/cache/data/dq/clevr/train/multimae_index_20_cosine.pickle'
RATIO="0.2"
NUM_BINS=20
RANDOM=0
TOTAL_LEN=50000

python alpaca_sample_multi.py \
--embed_path=$EMBED_PATH \
--save_path=$SAVE_PATH \
--ratio=$RATIO \
--k=$NUM_BINS \
--random=$RANDOM \
--total_len=$TOTAL_LEN
