"""Scramble the net's weights for input/output-only folders, so they are not loadable by accident.

Not encryption: a keystream from SHA-256 of the folder name, XORed with the bytes. The point is that
an agent told to use only samples.csv and check.py cannot open the weights casually, and the audit
flags any attempt to touch oracle.bin. Only stdlib + NumPy: this module is copied into blind folders.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _keystream(key: str, n: int) -> np.ndarray:
    blocks, i = [], 0
    while 32 * len(blocks) < n:
        blocks.append(hashlib.sha256(f"{key}:{i}".encode()).digest())
        i += 1
    return np.frombuffer(b"".join(blocks)[:n], dtype=np.uint8)


def encode(data: bytes, key: str) -> bytes:
    return (np.frombuffer(data, dtype=np.uint8) ^ _keystream(key, len(data))).tobytes()


decode = encode
