import base64
import gzip
import io
import json
import zipfile
import zlib

import numpy as np
import pytest

from merit_feddg.lossless_cache_transport import pack, unpack, restore_file


def original_record():
    values = np.linspace(0, 1, 2048, dtype='<f4')
    mask = {'encoding': 'float32-zlib-base64', 'size': [32, 64],
            'data': base64.b64encode(zlib.compress(values.tobytes())).decode()}
    buffer = io.BytesIO()
    with gzip.GzipFile(filename='native.json.tmp', fileobj=buffer, mode='wb', compresslevel=1, mtime=123) as gz:
        with io.TextIOWrapper(gz, encoding='utf-8') as text:
            json.dump({'items': [mask, mask], 'note': '医学'}, text)
    return buffer.getvalue()


def test_exact_gzip_and_repeated_mask_roundtrip():
    original = original_record()
    packed = pack(original)
    assert unpack(packed) == original
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        metadata = json.loads(archive.read('metadata.json'))
    assert len(metadata['masks']) == 1
    assert metadata['masks'][0]['occurrences'] == 2


def test_mismatched_original_hash_rejected():
    packed = pack(original_record())
    forged = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(packed)) as source, zipfile.ZipFile(forged, 'w') as target:
        for name in source.namelist():
            content = source.read(name)
            if name == 'metadata.json':
                metadata = json.loads(content)
                metadata['file_sha256'] = '0' * 64
                content = json.dumps(metadata).encode()
            target.writestr(name, content)
    with pytest.raises(ValueError, match='gzip SHA mismatch'):
        unpack(forged.getvalue())


def test_atomic_restore_and_retry_preserve_existing(tmp_path):
    original = original_record()
    packed = pack(original)
    destination = tmp_path / 'native.json.gz'
    restore_file(packed, destination)
    inode = destination.stat().st_ino
    restore_file(packed, destination)
    assert destination.read_bytes() == original
    assert destination.stat().st_ino == inode
    destination.write_bytes(b'other native record')
    with pytest.raises(ValueError, match='refusing overwrite'):
        restore_file(packed, destination)
    assert destination.read_bytes() == b'other native record'
    assert list(tmp_path.iterdir()) == [destination]


def test_truncated_transport_is_never_published(tmp_path):
    destination = tmp_path / 'native.json.gz'
    with pytest.raises(zipfile.BadZipFile):
        restore_file(pack(original_record())[:-30], destination)
    assert not destination.exists()
    assert not list(tmp_path.iterdir())
