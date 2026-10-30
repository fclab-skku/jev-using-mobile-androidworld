"""Dependency-free reader for GZIP TFRecord files of tf.train.Example protos."""

import gzip
import struct
from typing import BinaryIO, Dict, Iterator, List, Tuple, Union

Feature = Union[List[bytes], List[int], List[float]]


def read_varint(buf: bytes, i: int) -> Tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return result, i


def iter_fields(buf: bytes) -> Iterator[Tuple[int, int, Union[int, bytes]]]:
    """Yield (field_number, wire_type, value) for one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = read_varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = read_varint(buf, i)
        elif wire == 2:
            size, i = read_varint(buf, i)
            value = buf[i : i + size]
            i += size
        elif wire == 5:
            value = buf[i : i + 4]
            i += 4
        elif wire == 1:
            value = buf[i : i + 8]
            i += 8
        else:
            raise ValueError("Unsupported protobuf wire type %d" % wire)
        yield field, wire, value


def packed_varints(buf: bytes) -> List[int]:
    out, i = [], 0
    while i < len(buf):
        value, i = read_varint(buf, i)
        out.append(value)
    return out


def _signed64(value: int) -> int:
    return value - (1 << 64) if value >= 1 << 63 else value


def parse_example(buf: bytes) -> Dict[str, Feature]:
    """tf.train.Example -> {key: list of bytes | int | float}."""
    features: Dict[str, Feature] = {}
    for field, _, feats in iter_fields(buf):
        if field != 1:
            continue
        for _, _, entry in iter_fields(feats):
            key, value = None, b""
            for f, _, v in iter_fields(entry):
                if f == 1:
                    key = v.decode()
                elif f == 2:
                    value = v
            items: Feature = []
            for kind, wire, lst in iter_fields(value):
                if kind == 1:  # BytesList
                    items = [v for _, _, v in iter_fields(lst)]
                elif kind == 2:  # FloatList (packed or not)
                    raw = b"".join(v for _, _, v in iter_fields(lst))
                    items = list(struct.unpack("<%df" % (len(raw) // 4), raw))
                elif kind == 3:  # Int64List (packed or not)
                    ints: List[int] = []
                    for _, w, v in iter_fields(lst):
                        ints.extend(packed_varints(v) if w == 2 else [v])
                    items = [_signed64(x) for x in ints]
            features[key] = items
    return features


def iter_records(stream: BinaryIO) -> Iterator[bytes]:
    """Yield raw record payloads from an (uncompressed) TFRecord stream. CRCs are not checked."""
    while True:
        header = stream.read(12)
        if not header:
            return
        if len(header) < 12:
            raise EOFError("Truncated TFRecord header")
        (length,) = struct.unpack("<Q", header[:8])
        data = stream.read(length)
        if len(data) < length or len(stream.read(4)) < 4:
            raise EOFError("Truncated TFRecord payload")
        yield data


def iter_examples(path: str) -> Iterator[Dict[str, Feature]]:
    with gzip.open(path, "rb") as stream:
        for record in iter_records(stream):
            yield parse_example(record)
