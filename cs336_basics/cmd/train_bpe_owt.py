import time
import logging
import os

import pickle

import cs336_basics  # triggers logging configuration in package __init__
from cs336_basics.bpe import run_train_bpe

logger = logging.getLogger(__name__)

input_path = "data/owt_train.txt"
output_dir = "output/bpe/owt"

begin_time = time.time()
vocab, merges = run_train_bpe(
    input_path=input_path,
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
)
end_time = time.time()

logger.info("BPE training elapsed time: %.2f s", end_time - begin_time)

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)
with open(os.path.join(output_dir, "merges.pkl"), "wb") as f:
    pickle.dump(merges, f)
