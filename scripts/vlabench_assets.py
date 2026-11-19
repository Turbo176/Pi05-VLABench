"""Read selected official asset ZIP members through bounded HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from urllib.request import ProxyHandler, Request, build_opener
import zipfile
import xml.etree.ElementTree as ET
import zlib

ARCHIVES = {
    'objects': ('1ldEMZua2OzXHJTYTCP0IGVU1aFYBCMu-', 4987960728),
    'scenes': ('1KdReRkibJClBHHD32jz_wTkaBzhEJ9Kw', 1015309482),
}
MAX_TOTAL = 100 * 1024**2


class RangeZip(io.RawIOBase):
    def __init__(self, archive, proxy_fallback):
        self.file_id, self.size = ARCHIVES[archive]
        self.offset = 0
        self.transferred = 0
        self.opener = build_opener(ProxyHandler({}))
        if proxy_fallback:
            try:
                self.fetch(self.size - 1024, self.size, timeout=5, retries=1)
            except OSError as error:
                print(f'Direct Drive request failed ({error}). Trying configured network route.', flush=True)
                self.opener = build_opener()

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=os.SEEK_SET):
        self.offset = {os.SEEK_SET: 0, os.SEEK_CUR: self.offset, os.SEEK_END: self.size}[whence] + offset
        if self.offset < 0:
            raise OSError('Negative ZIP offset')
        return self.offset

    def fetch(self, start, stop, timeout=20, retries=3):
        if not 0 <= start < stop <= self.size or stop-start > MAX_TOTAL:
            raise ValueError(f'Range outside download budget: {start}:{stop}')
        span = f'{start}-{stop-1}'
        url = (f'https://drive.usercontent.google.com/download?id={self.file_id}'
               f'&export=download&confirm=t&pv_range={span}')
        expected = f'bytes {span}/{self.size}'
        for attempt in range(retries):
            try:
                request = Request(url, headers={'Range': f'bytes={span}', 'Accept-Encoding': 'identity'})
                with self.opener.open(request, timeout=timeout) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise OSError(f'Expected HTTP 206 and {expected}, got {response.status} and '
                                      f'{response.headers.get("Content-Range")}')
                    data = response.read(stop-start+1)
                if len(data) != stop-start:
                    raise OSError(f'Truncated range: {span}')
                return data
            except OSError:
                if attempt == retries-1:
                    raise
                time.sleep(0.5 * (attempt+1))

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.offset + size)
        if end <= self.offset:
            return b''
        data = self.fetch(self.offset, end)
        self.offset = end
        self.transferred += len(data)
        return data


def asset_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError(f'Invalid archive member: {name}')
    return path


def asset_batches(archive, entries):
    ends = {entry.header_offset: end for entry, end in zip(
        sorted(archive.infolist(), key=lambda entry: entry.header_offset),
        sorted(entry.header_offset for entry in archive.infolist())[1:] + [archive.start_dir],
    )}
    groups = []
    for entry in sorted(entries, key=lambda entry: entry.header_offset):
        start, stop = entry.header_offset, ends[entry.header_offset]
        if groups and start - groups[-1][1] <= 16 * 1024 and stop - groups[-1][0] <= 4 * 1024**2:
            groups[-1][1] = stop
            groups[-1][2].append(entry)
        else:
            groups.append([start, stop, [entry]])
    return groups


def extract_group(remote, group, assets):
    start, stop, entries = group
    blob = remote.fetch(start, stop)
    results = []
    for entry in entries:
        offset = entry.header_offset - start
        with io.BytesIO(blob) as local:
            local.seek(offset + 26)
            name_len = int.from_bytes(local.read(2), 'little')
            extra_len = int.from_bytes(local.read(2), 'little')
            payload = offset + 30 + name_len + extra_len
        if blob[offset:offset+4] != b'PK\x03\x04' or payload + entry.compress_size > len(blob):
            raise OSError(f'Invalid local ZIP header: {entry.filename}')
        # Open the remote member with ZipExtFile's decompressor and CRC check.
        compressed = io.BytesIO(blob[payload:payload + entry.compress_size])
        with zipfile.ZipExtFile(compressed, 'r', entry, close_fileobj=True) as source:
            data = source.read()
        if len(data) != entry.file_size:
            raise OSError(f'Wrong member size: {entry.filename}')
        target = assets / asset_path(entry.filename)
        if not target.resolve().is_relative_to(assets):
            raise ValueError(f'Member escapes asset directory: {entry.filename}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.partial')
        temp.write_bytes(data)
        temp.replace(target)
        results.append({'path': entry.filename, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', choices=ARCHIVES)
    parser.add_argument('--root', type=Path, default=Path('.local/VLABench/VLABench/assets'))
    parser.add_argument('--proxy-fallback', action='store_true', help='Try configured routes if a direct probe fails')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--match', action='append', default=[], help='Regex for a member to extract')
    parser.add_argument('--mesh-directories', action='store_true', help='Create official object catalog directories used during import')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--xml-deps', help='Include file references from an MJCF already on disk')
    parser.add_argument('--reuse-scene', help='Check another scene directory for byte-identical ZIP members')
    args = parser.parse_args()
    assets = args.root.resolve(strict=True)
    remote = RangeZip(args.archive, args.proxy_fallback)
    with zipfile.ZipFile(remote) as archive:
        members = [entry for entry in archive.infolist() if entry.filename.startswith(('obj/', 'scenes/'))]
        if args.list:
            for entry in members:
                print(f'{entry.compress_size}\t{entry.file_size}\t{entry.filename}')
            return
        patterns = [re.compile(pattern) for pattern in args.match]
        if args.xml_deps:
            xml = asset_path(args.xml_deps)
            references = {str(xml)}
            for element in ET.parse(assets / xml).iter():
                if 'file' in element.attrib:
                    references.add(str(xml.parent / asset_path(element.attrib['file'])))
            missing = references - {entry.filename for entry in members}
            if missing:
                raise ValueError(f'MJCF references absent from archive: {sorted(missing)}')
            patterns.extend(re.compile(f'^{re.escape(ref)}$') for ref in references)
        selected = [entry for entry in members if not entry.is_dir()
                    and any(pattern.search(entry.filename) for pattern in patterns)]
        if not selected or not args.manifest:
            parser.error('Extraction requires matching files and --manifest')
        reused = []
        to_fetch = []
        for entry in selected:
            path = asset_path(entry.filename)
            target = assets / path
            options = [target]
            if args.reuse_scene and path.parts[0] == 'scenes':
                options.append(assets / 'scenes' / asset_path(args.reuse_scene) / PurePosixPath(*path.parts[2:]))
            for candidate in options:
                if candidate.is_file() and candidate.stat().st_size == entry.file_size:
                    data = candidate.read_bytes()
                    if zlib.crc32(data) == entry.CRC:
                        if candidate != target:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(data)
                        reused.append({'path': entry.filename, 'bytes': len(data),
                                       'sha256': hashlib.sha256(data).hexdigest()})
                        break
            else:
                to_fetch.append(entry)
        groups = asset_batches(archive, to_fetch)
        planned = sum(stop-start for start, stop, _ in groups) + remote.transferred
        expanded = sum(entry.file_size for entry in selected)
        print(f'{args.archive}: {len(selected)} files, {len(groups)} ranges, {planned:,} transfer bytes, {expanded:,} extracted bytes', flush=True)
        if planned > MAX_TOTAL or expanded > 3 * MAX_TOTAL:
            raise SystemExit('Selection exceeds the smoke-test budget')
        if args.mesh_directories:
            for entry in members:
                if entry.filename.startswith('obj/meshes/'):
                    path = asset_path(entry.filename)
                    (assets / (path if entry.is_dir() else path.parent)).mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            batches = pool.map(lambda group: extract_group(remote, group, assets), groups)
            files = [item for batch in batches for item in batch]
        files = sorted(files + reused, key=lambda item: item['path'])
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({'archive': args.archive, 'drive_id': remote.file_id,
                                            'archive_bytes': remote.size, 'transfer_bytes': planned,
                                            'files': files}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
