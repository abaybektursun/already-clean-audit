"""Read one member of an .npz on S3 by HTTP range requests (presigned URL; profile fuelos, us-east-1, read-only).

Only the zip directory and the named member's bytes are fetched. Nothing is written to disk.
"""
import io, json, struct, subprocess, urllib.request, zlib
import numpy as np

BUCKET = "pg-lineage-769063704696"
AWS = ["aws", "--profile", "fuelos", "--region", "us-east-1"]


def presign(key):
    return subprocess.run(AWS + ["s3", "presign", f"s3://{BUCKET}/{key}", "--expires-in", "3600"],
                          check=True, capture_output=True, text=True).stdout.strip()


def get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    assert len(data) == b - a, (len(data), a, b)
    return data


def size(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Range"].split("/")[1])


def directory(url):
    n = size(url)
    tail = get(url, max(0, n - 65536), n)
    base = max(0, n - 65536)
    e = tail.rfind(b"PK\x05\x06")
    assert e >= 0, "no end of central directory"
    cd_size, cd_off = struct.unpack("<II", tail[e + 12:e + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:  # zip64
        l = tail.rfind(b"PK\x06\x07")
        z64 = struct.unpack("<Q", tail[l + 8:l + 16])[0]
        rec = get(url, z64, z64 + 56) if z64 < base else tail[z64 - base:z64 - base + 56]
        cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
    cd = tail[cd_off - base:cd_off - base + cd_size] if cd_off >= base else get(url, cd_off, cd_off + cd_size)
    out, i = {}, 0
    while i < len(cd):
        assert cd[i:i + 4] == b"PK\x01\x02"
        method, = struct.unpack("<H", cd[i + 10:i + 12])
        csize, usize = struct.unpack("<II", cd[i + 20:i + 28])
        nl, xl, cl = struct.unpack("<HHH", cd[i + 28:i + 34])
        off, = struct.unpack("<I", cd[i + 42:i + 46])
        name = cd[i + 46:i + 46 + nl].decode()
        extra = cd[i + 46 + nl:i + 46 + nl + xl]
        j = 0
        while j < len(extra):  # zip64 extra: the fields that are 0xFFFFFFFF, in order
            hid, hl = struct.unpack("<HH", extra[j:j + 4])
            if hid == 1:
                vals = list(struct.unpack("<" + "Q" * (hl // 8), extra[j + 4:j + 4 + hl]))
                if usize == 0xFFFFFFFF:
                    usize = vals.pop(0)
                if csize == 0xFFFFFFFF:
                    csize = vals.pop(0)
                if off == 0xFFFFFFFF:
                    off = vals.pop(0)
            j += 4 + hl
        out[name] = dict(method=method, csize=csize, usize=usize, off=off)
        i += 46 + nl + xl + cl
    return out


def member(key, name, url=None, d=None):
    url = url or presign(key)
    d = d or directory(url)
    m = d[name + ".npy"]
    head = get(url, m["off"], m["off"] + 30)
    assert head[:4] == b"PK\x03\x04"
    nl, xl = struct.unpack("<HH", head[26:30])
    start = m["off"] + 30 + nl + xl
    raw = get(url, start, start + m["csize"])
    if m["method"] == 8:
        raw = zlib.decompress(raw, -15)
    else:
        assert m["method"] == 0
    assert len(raw) == m["usize"]
    return np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)


def members(key, names):
    url = presign(key)
    d = directory(url)
    return {n: member(key, n, url, d) for n in names}
