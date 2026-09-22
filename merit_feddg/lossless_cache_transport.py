"""Experimental byte-exact native-cache transport; never a model input format."""
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import shlex
import subprocess
import zipfile
import zlib

import numpy as np


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _gzip_header(data):
    if data[:3] != b'\x1f\x8b\x08' or data[3] & 0xE0:
        raise ValueError('Unsupported gzip envelope')
    flags, pos = data[3], 10
    if flags & 4:
        pos += 2 + struct.unpack_from('<H', data, pos)[0]
    for flag in (8, 16):
        if flags & flag:
            pos = data.index(b'\0', pos) + 1
    if flags & 2:
        pos += 2
    return data[:pos]


def _gzip_restore(raw, header, footer):
    compressor = zlib.compressobj(1, zlib.DEFLATED, -15)
    # Native caches use gzip.open(..., 'wt'): TextIOWrapper flushes once
    # before GzipFile.close emits the final deflate block.
    return header + compressor.compress(raw) + compressor.flush(zlib.Z_SYNC_FLUSH) + compressor.flush() + footer


def _masks(value):
    if isinstance(value, dict):
        if value.get('encoding') == 'float32-zlib-base64':
            yield value['data']
        else:
            for child in value.values():
                yield from _masks(child)
    elif isinstance(value, list):
        for child in value:
            yield from _masks(child)


def pack(original):
    """Reject envelopes that cannot be reproduced exactly; caller keeps originals."""
    raw = gzip.decompress(original)
    header, footer = _gzip_header(original), original[-8:]
    if _gzip_restore(raw, header, footer) != original:
        raise ValueError('Original gzip cannot be reproduced with this codec')
    template = raw
    blobs, records = [], []
    for i, data in enumerate(dict.fromkeys(_masks(json.loads(raw)))):
        compressed = base64.b64decode(data, validate=True)
        pixels = zlib.decompress(compressed)
        if len(pixels) % 4 or not pixels:
            raise ValueError('Invalid float32 byte length')
        if zlib.compress(pixels) != compressed:
            raise ValueError('Original mask encoding cannot be reproduced')
        values = np.frombuffer(pixels, dtype='<u4')
        delta = np.empty_like(values)
        delta[0] = values[0]
        np.subtract(values[1:], values[:-1], out=delta[1:])
        shuffled = delta.view(np.uint8).reshape(-1, 4).T.copy().tobytes()
        marker = ('__NATIVE_TRANSPORT_' + _sha(raw) + '_' + str(i) + '__').encode()
        if marker in raw:
            raise ValueError('Transport marker collision')
        encoded = data.encode('ascii')
        occurrences = template.count(encoded)
        if not occurrences:
            raise ValueError('Mask not found verbatim in JSON')
        template = template.replace(encoded, marker)
        blobs.append(zlib.compress(shuffled))
        records.append({'marker': marker.decode(), 'nbytes': len(pixels),
                        'encoded_sha256': _sha(compressed), 'occurrences': occurrences})
    metadata = {'version': 1, 'file_sha256': _sha(original), 'raw_sha256': _sha(raw),
                'header': base64.b64encode(header).decode(),
                'footer': base64.b64encode(footer).decode(), 'masks': records}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('metadata.json', json.dumps(metadata))
        archive.writestr('template', template, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
        for i, blob in enumerate(blobs):
            archive.writestr(str(i), blob)
    packed = stream.getvalue()
    if unpack(packed) != original:
        raise ValueError('Transport round-trip failed')
    return packed


def unpack(packed):
    """Restore in memory; caller must publish only after this function succeeds."""
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        metadata = json.loads(archive.read('metadata.json'))
        if metadata['version'] != 1:
            raise ValueError('Unsupported transport version')
        raw = archive.read('template')
        for i, record in enumerate(metadata['masks']):
            blob = zlib.decompress(archive.read(str(i)))
            if len(blob) != record['nbytes'] or len(blob) % 4:
                raise ValueError('Mask size mismatch')
            delta = np.frombuffer(blob, dtype=np.uint8).reshape(4, -1).T.copy().view('<u4').reshape(-1)
            pixels = np.cumsum(delta, dtype=np.uint32).astype('<u4').tobytes()
            compressed = zlib.compress(pixels)
            if _sha(compressed) != record['encoded_sha256']:
                raise ValueError('Reconstructed mask SHA mismatch')
            marker = record['marker'].encode()
            if raw.count(marker) != record['occurrences']:
                raise ValueError('Mask occurrence mismatch')
            raw = raw.replace(marker, base64.b64encode(compressed))
    if _sha(raw) != metadata['raw_sha256']:
        raise ValueError('Reconstructed JSON SHA mismatch')
    result = _gzip_restore(raw, base64.b64decode(metadata['header']), base64.b64decode(metadata['footer']))
    if _sha(result) != metadata['file_sha256']:
        raise ValueError('Reconstructed gzip SHA mismatch')
    return result


def restore_file(packed, destination):
    """Publish verified original bytes atomically, never overwrite a different file."""
    original = unpack(packed)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,
                                         prefix='.native-restore-', suffix='.tmp',
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Same-filesystem hard link publishes complete bytes without a
            # check-then-replace race against native producers or another retry.
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != original:
                raise ValueError('Existing destination differs; refusing overwrite')
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _sha(original)


def send_files(paths, ssh_command, remote_python, remote_root):
    """Best-effort acceleration; caller MUST run its original checksum rsync next."""
    source = base64.b64encode(Path(__file__).read_bytes()).decode()
    for path in paths:
        path = Path(path)
        # Bound per-file memory; small/unsupported files use ordinary rsync.
        if not 8 * 2**20 <= path.stat().st_size <= 128 * 2**20:
            continue
        try:
            original = path.read_bytes()
            destination = str(Path(remote_root) / path)
            expected = _sha(original)
            check = ('import pathlib,hashlib,sys;p=pathlib.Path(' + repr(destination)
                     + ');sys.exit(0 if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()=='
                     + repr(expected) + ' else 1)')
            existing = subprocess.run(ssh_command + [remote_python + ' -c ' + shlex.quote(check)],
                                      timeout=30, check=False)
            if existing.returncode == 0:
                continue
            if existing.returncode != 1:
                raise RuntimeError('Remote precheck failed')
            packed = pack(original)
            if len(packed) >= len(original) * .85:
                continue
            code = ('import base64,sys;exec(base64.b64decode(' + repr(source)
                    + '));print(restore_file(sys.stdin.buffer.read(),'
                    + repr(destination) + '))')
            result = subprocess.run(ssh_command + [remote_python + ' -c ' + shlex.quote(code)],
                                    input=packed, stdout=subprocess.PIPE,
                                    timeout=240, check=True)
            if result.stdout.decode().strip() != expected:
                raise ValueError('Remote restored SHA acknowledgment mismatch')
            print('LOSSLESS_DELIVERED', path, len(original), len(packed), flush=True)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
            print('LOSSLESS_FALLBACK', path, str(exc), flush=True)
