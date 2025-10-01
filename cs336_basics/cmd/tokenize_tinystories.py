import time
import logging
import os
import pickle
import numpy as np

import cs336_basics  # triggers logging configuration in package __init__
from cs336_basics.bpe import Tokenizer

logger = logging.getLogger(__name__)

work_dir = "output/bpe/tinystories"
names = ["TinyStoriesV2-GPT4-train", "TinyStoriesV2-GPT4-valid"]

begin_time = time.time()
tokenizer = Tokenizer.from_files(
    vocab_filepath=os.path.join(work_dir, "vocab.pkl"),
    merges_filepath=os.path.join(work_dir, "merges.pkl"),
    special_tokens=["<|endoftext|>"],
)
for name in names:
    logger.info("Tokenizing %s...", name)
    input_path = f"data/{name}.txt"
    output_path = f"data/{name}.npy"
    with open(input_path, "r", encoding="utf-8") as fp:
        logger.info("File size: %.2f bytes", os.path.getsize(input_path))
        ids = list(tokenizer.encode_iterable(fp))
    logger.info("Number of tokens: %d", len(ids))
    arr = np.array(ids, dtype=np.uint16)
    np.save(output_path, arr)
end_time = time.time()

logger.info("Tokenization elapsed time: %.2f s", end_time - begin_time)
