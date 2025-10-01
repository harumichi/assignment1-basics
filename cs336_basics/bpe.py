from __future__ import annotations

from collections import defaultdict
import multiprocessing as mp
from collections.abc import Iterable, Iterator as Iterator
import logging
import json
import pickle
import os
from typing import Iterable, Dict, List, Tuple, Set, BinaryIO
from tqdm import tqdm

import regex as re

logger = logging.getLogger(__name__)


def find_chunk_boundaries(
	file: BinaryIO,
	desired_num_chunks: int,
	split_special_tokens: list[bytes],
) -> list[int]:
	# Chunk the file into parts that can be counted independently. Supports multiple special tokens.
	for t in split_special_tokens:
		assert isinstance(t, bytes)
	file.seek(0, os.SEEK_END)
	file_size = file.tell()
	file.seek(0)
	if desired_num_chunks <= 0:
		return [0, file_size]
	chunk_size = max(1, file_size // desired_num_chunks)
	chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
	chunk_boundaries[-1] = file_size
	mini_chunk_size = 4096
	needle_list = split_special_tokens if split_special_tokens else [b""]
	for bi in range(1, len(chunk_boundaries) - 1):
		initial_position = chunk_boundaries[bi]
		file.seek(initial_position)
		while True:
			mini_chunk = file.read(mini_chunk_size)
			if mini_chunk == b"":
				chunk_boundaries[bi] = file_size
				break
			# find earliest occurrence among special tokens
			found_positions = [mini_chunk.find(tok) for tok in needle_list if tok]
			found_positions = [p for p in found_positions if p != -1]
			if found_positions:
				found_at = min(found_positions)
				chunk_boundaries[bi] = initial_position + found_at
				break
			initial_position += mini_chunk_size
	return sorted(set(chunk_boundaries))


def split_with_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
	if not special_tokens:
		return [text]
	# シンプルさ優先: オルタネーション生成のみ
	pattern = "|".join(re.escape(s) for s in special_tokens)
	return re.split(pattern, text)


def pretokenize(text: str) -> list[bytes]:
	PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
	return [s.encode() for s in re.findall(PAT, text)]


def _chunk_worker(params: tuple[str | os.PathLike, int, int, list[str], Dict[bytes, int]]):
	path, start, end, specials, token_idx_local = params
	counts: Dict[Tuple[int, ...], int] = defaultdict(int)
	with open(path, "rb") as fbin:
		fbin.seek(start)
		data = fbin.read(end - start).decode("utf-8", errors="ignore")
	for piece in split_with_special_tokens(data, specials):
		if not piece:
			continue
		for s in pretokenize(piece):
			word = tuple(token_idx_local[bytes([b])] for b in s)
			counts[word] += 1
	return counts


def run_train_bpe(
	input_path: str | os.PathLike,
	vocab_size: int,
	special_tokens: list[str],
	**_: dict,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
	"""Train a Byte Pair Encoding tokenizer.

	Args:
		input_path: Path to the corpus text file (UTF-8).
		vocab_size: Desired vocabulary size including initial byte tokens and special tokens.
		special_tokens: List of special token strings to be appended to the base byte vocabulary.
		**_: Ignored keyword arguments kept for backward compatibility with previous adapter signature.

	Returns:
		(vocab, merges):
			vocab: dict mapping token id -> token bytes.
			merges: ordered list of (left_token_bytes, right_token_bytes) for each merge performed.
	"""
	# Build initial vocabulary: raw bytes 0..255 then special tokens.
	token_idx: Dict[bytes, int] = {}
	tokens: List[bytes] = []

	def add_token(t: bytes):
		assert t not in token_idx
		token_idx[t] = len(token_idx)
		tokens.append(t)
		assert len(token_idx) == len(tokens)

	for i in range(256):
		add_token(bytes([i]))
	for t in special_tokens:
		add_token(t.encode())
	logger.info("Added %d initial byte tokens and %d special tokens", 256, len(special_tokens))

	# Determine chunk boundaries; workers will read lazily.
	n_procs = max(1, mp.cpu_count())
	desired_num_chunks = n_procs * 64
	with open(input_path, "rb") as fb:
		boundaries = find_chunk_boundaries(
			fb,
			desired_num_chunks,
			[t.encode() for t in special_tokens],
		)
	logger.info("Computed %d boundaries (producing %d chunks)", len(boundaries), len(boundaries) - 1)

	worker_inputs = [
		(input_path, s, e, special_tokens, token_idx)
		for s, e in zip(boundaries[:-1], boundaries[1:])
	]
	with mp.Pool(processes=n_procs) as pool:
		results = pool.map(_chunk_worker, worker_inputs)

	# Merge counts. Order doesn't matter; sort for determinism.
	raw_count: Dict[Tuple[int, ...], int] = defaultdict(int)
	for counts in results:
		for w, c in counts.items():
			# insertion order of first appearance retained
			raw_count[w] += c
	words_list = list(raw_count.keys())
	word_idx: Dict[Tuple[int, ...], int] = {w: i for i, w in enumerate(words_list)}
	words: List[Tuple[int, ...]] = words_list
	word_count: Dict[int, int] = {word_idx[w]: raw_count[w] for w in words_list}
	logger.info("Pre-tokenized input into %d unique words", len(words))

	pair_count: Dict[Tuple[int, int], int] = defaultdict(int)
	# token pair -> それが含まれているwordの集合
	pair_widx_set: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
	merges: List[Tuple[int, int]] = []
	for word, widx in word_idx.items():
		for j in range(len(word) - 1):
			pair = (word[j], word[j + 1])
			pair_count[pair] += word_count[widx]
			pair_widx_set[pair].add(widx)

	while len(token_idx) < vocab_size:
		if len(token_idx) % 10 == 0:
			logger.info("Vocabulary size %d / %d", len(token_idx), vocab_size)
		if not pair_count:
			break
		pair = max(
			pair_count,
			key=lambda p: (pair_count[p], (tokens[p[0]], tokens[p[1]])),
		)
		merges.append(pair)
		new_token = tokens[pair[0]] + tokens[pair[1]]
		add_token(new_token)
		for widx in list(pair_widx_set[pair]):
			word = words[widx]
			new_word: list[int] = []
			j = 0
			while j < len(word):
				if j < len(word) - 1 and (word[j], word[j + 1]) == pair:
					new_word.append(token_idx[new_token])
					j += 2
				else:
					new_word.append(word[j])
					j += 1
			new_word_t = tuple(new_word)

			for j in range(len(word) - 1):
				p = (word[j], word[j + 1])
				pair_count[p] -= word_count[widx]
				pair_widx_set[p].discard(widx)
				if pair_count[p] == 0:
					pair_count.pop(p)
			for j in range(len(new_word_t) - 1):
				p = (new_word_t[j], new_word_t[j + 1])
				pair_count[p] += word_count[widx]
				pair_widx_set[p].add(widx)

			words[widx] = new_word_t
			word_idx.pop(word)
			word_idx[new_word_t] = widx

	vocab = {i: t for i, t in enumerate(tokens)}
	merges_bytes_list = [(tokens[p[0]], tokens[p[1]]) for p in merges]
	return vocab, merges_bytes_list


class Tokenizer:
	def __init__(
		self,
		vocab: dict[int, bytes],
		merges: list[tuple[bytes, bytes]],
		special_tokens: list[str] | None = None,
	) -> None:
		token_idx = {t: i for i, t in vocab.items()}
		for t in special_tokens or []:
			b = t.encode()
			if b not in token_idx:
				token_idx[b] = len(token_idx)
				vocab[len(vocab)] = b
		self.merge_order: dict[tuple[int, int], int] = {
			(token_idx[m[0]], token_idx[m[1]]): i for i, m in enumerate(merges)
		}
		self.merge_target: dict[tuple[int, int], int] = {
			(token_idx[m[0]], token_idx[m[1]]): token_idx[m[0] + m[1]] for m in merges
		}
		self.vocab = vocab
		self.token_idx = token_idx
		# Noneを空リストに正規化
		self.special_tokens = list(special_tokens) if special_tokens else []
		# 長い方を優先してマッチさせるため長さ降順で保持
		self._special_tokens_sorted: list[str] = sorted(self.special_tokens, key=len, reverse=True)

	@classmethod
	def from_files(
		cls,
		vocab_filepath: str,
		merges_filepath: str,
		special_tokens: list[str] | None = None,
	):
		with open(vocab_filepath, "rb") as fp:
			vocab = pickle.load(fp)
		with open(merges_filepath, "rb") as fp:
			merges = pickle.load(fp)
		return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

	def encode(self, text: str) -> list[int]:
		result: list[int] = []
		if self._special_tokens_sorted:
			pattern = re.compile("(" + "|".join(re.escape(t) for t in self._special_tokens_sorted) + ")")
			parts = pattern.split(text)
			special_set = set(self.special_tokens)
		else:
			parts = [text]
			special_set = set()

		for part in parts:
			if not part:
				continue
			if part in special_set:
				result.append(self.token_idx[part.encode()])
				continue
			for x in pretokenize(part):
				tokens = [self.token_idx[bytes([b])] for b in x]
				while True:
					candidates: list[tuple[int, tuple[int, int]]] = []
					for j in range(len(tokens) - 1):
						pair = (tokens[j], tokens[j + 1])
						if pair in self.merge_order:
							candidates.append((self.merge_order[pair], pair))
					if not candidates:
						result.extend(tokens)
						break
					_, best_pair = min(candidates, key=lambda x: x[0])
					new_tokens: list[int] = []
					k = 0
					while k < len(tokens):
						if k < len(tokens) - 1 and (tokens[k], tokens[k + 1]) == best_pair:
							new_tokens.append(self.merge_target[best_pair])
							k += 2
						else:
							new_tokens.append(tokens[k])
							k += 1
					tokens = new_tokens
		return result

	def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
		procs = max(1, mp.cpu_count())
		with mp.Pool(processes=procs) as pool:
			for i, encoded in enumerate(pool.imap(self.encode, iterable, 128)):
				yield from encoded
				if i % 100000 == 0:
					logger.info("encode_iterable processed %d lines", i)

	def decode(self, ids: list[int]) -> str:
		return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
