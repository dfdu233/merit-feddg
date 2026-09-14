"""Download one pinned U-Bench U-KAN reproduction checkpoint; never train or install.

Use bounded parallel HTTP ranges and validate the publisher's LFS SHA-256 before
publishing the assembled file. Partial files are retained for resumable retries.
"""
import concurrent.futures
import hashlib
import pathlib
import subprocess

URL = ('https://huggingface.co/FengheTan9/U-Bench/resolve/'
       'e6945d560d2395254c72ce0100cf10638784c578/U_KAN/busi/checkpoint_best.pth')
SIZE = 101790181
SHA = '2c6ae3fc0c99650585776d6cc304144bb334dde267057c69cd52722db3672e22'
ROOT = pathlib.Path(__file__).resolve().parents[1] / 'runs/model-downloads'


def fetch(interval):
    start, end = interval
    path = ROOT / f'ukan-range-{start}-{end}.bin'
    if path.exists() and path.stat().st_size == end-start+1:
        return path
    partial = path.with_suffix('.part')
    headers = path.with_suffix('.headers')
    result = subprocess.run(['curl', '-fL', '--silent', '--show-error',
        '--max-time', '90', '--retry', '2', '--range', f'{start}-{end}',
        URL + f'?download=true&range_retry={start}',
        '-D', str(headers), '-o', str(partial), '-w', '%{http_code}'],
        capture_output=True, text=True, timeout=300, check=True)
    expected = f'content-range: bytes {start}-{end}/{SIZE}'
    if result.stdout != '206' or expected not in headers.read_text().lower().splitlines():
        raise ValueError('server did not return the requested exact range')
    content = partial.read_bytes()
    if len(content) != end-start+1:
        raise ValueError('incomplete download range')
    path.write_bytes(content)
    print(f'downloaded {start}-{end}', flush=True)
    return path


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'ukan-busi-ubench-verified.pth'
    if output.exists():
        if hashlib.sha256(output.read_bytes()).hexdigest() != SHA:
            raise ValueError('existing output checksum mismatch')
        print('already verified', output); return
    # Known earlier direct-download prefix; final SHA validates it as well.
    prefix = ROOT / 'ukan-busi-ubench.pth'
    start = prefix.stat().st_size if prefix.exists() else 0
    if not 0 <= start <= SIZE:
        raise ValueError('invalid retained prefix length')
    intervals = [(i,min(i+4*1024*1024-1,SIZE-1)) for i in range(start,SIZE,4*1024*1024)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(fetch, intervals))
    content = (prefix.read_bytes() if prefix.exists() else b'') + b''.join(p.read_bytes() for p in parts)
    if len(content) != SIZE or hashlib.sha256(content).hexdigest() != SHA:
        raise ValueError('publisher SHA256 mismatch; not a usable checkpoint')
    with output.open('xb') as stream:
        stream.write(content)
    print('VERIFIED', SHA, output)


if __name__ == '__main__':
    main()
