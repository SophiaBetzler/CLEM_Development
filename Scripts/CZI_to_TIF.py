#!/usr/bin/env python3
"""
Convert Zeiss .czi (Gray8/Gray16, uncompressed or Zstd) to ImageJ TIFF(s)
whose pixel size is written into the resolution tags, so they plug directly
into the CLEM pipeline (read_tiff_pixel_spacing_um + load_ome_tiff).

Dependencies (install on your machine):
    pip install imagecodecs tifffile numpy
(imagecodecs bundles the zstd decoder; alternatively `pip install zstandard`.)

Usage:
    # single file
    python czi_to_tif.py input.czi [output.tif]

    # every .czi in a folder
    python czi_to_tif.py  path/to/folder
    python czi_to_tif.py  path/to/folder --recursive --outdir path/to/tifs --skip-existing
"""
import os
import re
import sys
import glob
import struct
import argparse
import numpy as np
import tifffile

# ---- zstd decoder: prefer imagecodecs, fall back to zstandard --------------
def _zstd_decode(buf):
    try:
        from imagecodecs import zstd_decode
        return zstd_decode(buf)
    except Exception:
        pass
    import zstandard
    return zstandard.ZstdDecompressor().decompress(buf)

_PT = {0: np.uint8, 1: np.uint16, 2: np.float32}     # CZI PixelType -> numpy
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _pixel_sizes_um(raw):
    """(xy_um, z_um) read from the CZI scaling XML; None if absent."""
    s = raw.find(b"<ImageDocument"); e = raw.find(b"</ImageDocument>")
    xml = (raw[s:e] if (s != -1 and e != -1) else raw).decode("utf-8", "replace")
    def dist(axis):
        m = re.search(rf'<Distance Id="{axis}">\s*<Value>([^<]+)</Value>', xml)
        return float(m.group(1)) * 1e6 if m else None      # metres -> um
    return dist("X"), dist("Z")


def _read_subblocks(raw):
    N, off, blocks = len(raw), 0, []
    while off + 32 <= N:
        sid = raw[off:off + 16].split(b"\x00")[0]
        allocated, _used = struct.unpack_from("<qq", raw, off + 16)
        ds = off + 32
        if sid == b"ZISRAWSUBBLOCK":
            meta, _att = struct.unpack_from("<ii", raw, ds)
            dsize, = struct.unpack_from("<q", raw, ds + 8)
            de = ds + 16
            pixel_type, = struct.unpack_from("<i", raw, de + 2)
            compression, = struct.unpack_from("<i", raw, de + 18)
            dim_count, = struct.unpack_from("<i", raw, de + 28)
            dims, dpos = {}, de + 32
            for _ in range(dim_count):
                name = raw[dpos:dpos + 4].split(b"\x00")[0].decode()
                start, size, _startc, stored = struct.unpack_from("<iifi", raw, dpos + 4)
                dims[name] = (start, size, stored)
                dpos += 20
            fixed = max(256, 16 + 32 + 20 * dim_count)
            payload = ds + fixed + meta
            blocks.append(dict(pixel_type=pixel_type, compression=compression,
                               dims=dims, chunk=raw[payload:payload + dsize]))
        if allocated <= 0:
            break
        off = ds + allocated
    return blocks


def _decode_plane(b):
    dtype = _PT.get(b["pixel_type"])
    if dtype is None:
        raise ValueError(f"Unsupported CZI PixelType {b['pixel_type']}")
    _, sx, _ = b["dims"]["X"]
    _, sy, _ = b["dims"]["Y"]
    comp, chunk = b["compression"], b["chunk"]
    if comp == 0:                                   # uncompressed
        data = chunk
    elif comp in (5, 6):                            # Zstd0 / Zstd1
        pos = chunk.find(_ZSTD_MAGIC)               # skip any zstd1 header
        if pos < 0:
            raise ValueError("Zstd frame magic not found in subblock")
        data = _zstd_decode(chunk[pos:])
    else:
        raise ValueError(f"Compression {comp} not supported here "
                         f"(install aicspylibczi for JPEG-XR etc.)")
    return np.frombuffer(data, dtype=dtype)[: sy * sx].reshape(sy, sx)


def convert(czi_path, tif_path=None):
    """Convert one .czi to an ImageJ .tif. Returns the output path."""
    raw = open(czi_path, "rb").read()
    xy_um, z_um = _pixel_sizes_um(raw)
    blocks = _read_subblocks(raw)
    if not blocks:
        raise ValueError("No image subblocks found in CZI")

    nC = max(b["dims"].get("C", (0, 1, 1))[0] for b in blocks) + 1
    nZ = max(b["dims"].get("Z", (0, 1, 1))[0] for b in blocks) + 1
    sample = blocks[0]
    _, sx, _ = sample["dims"]["X"]; _, sy, _ = sample["dims"]["Y"]
    dtype = _PT[sample["pixel_type"]]

    arr = np.zeros((nZ, nC, sy, sx), dtype=dtype)   # ImageJ order: Z, C, Y, X
    for b in blocks:
        c = b["dims"].get("C", (0,))[0]
        z = b["dims"].get("Z", (0,))[0]
        arr[z, c] = _decode_plane(b)

    if tif_path is None:
        tif_path = os.path.splitext(czi_path)[0] + ".tif"

    meta = {"axes": "ZCYX"}
    resolution = None
    if xy_um:
        meta["unit"] = "micron"
        resolution = (1.0 / xy_um, 1.0 / xy_um)
    if z_um:
        meta["spacing"] = z_um

    tifffile.imwrite(tif_path, arr, imagej=True, resolution=resolution, metadata=meta)
    print(f"[OK] {os.path.basename(czi_path)} -> {os.path.basename(tif_path)}  "
          f"(Z,C,Y,X)={arr.shape} {dtype.__name__} xy={xy_um} um/px z={z_um} um")
    return tif_path


def find_czis(folder, recursive=False):
    pattern = "**/*.czi" if recursive else "*.czi"
    hits = glob.glob(os.path.join(folder, pattern), recursive=recursive)
    seen, out = set(), []
    for p in sorted(hits):
        if os.path.basename(p).startswith("._"):
            continue
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key); out.append(p)
    return out


def convert_folder(folder, outdir=None, recursive=False, skip_existing=False):
    """Convert every .czi under `folder`. Returns (ok_list, fail_list)."""
    czis = find_czis(folder, recursive=recursive)
    if not czis:
        print(f"[INFO] No .czi files found in {folder}"
              f"{' (recursive)' if recursive else ''}.")
        return [], []
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] Found {len(czis)} .czi file(s).")
    ok, fail = [], []
    for i, czi in enumerate(czis, 1):
        stem = os.path.splitext(os.path.basename(czi))[0]
        tif = os.path.join(outdir, stem + ".tif") if outdir else \
              os.path.splitext(czi)[0] + ".tif"
        if skip_existing and os.path.exists(tif):
            print(f"[{i}/{len(czis)}] skip (exists): {os.path.basename(tif)}")
            ok.append(tif)
            continue
        try:
            print(f"[{i}/{len(czis)}] converting {os.path.basename(czi)} ...")
            convert(czi, tif)
            ok.append(tif)
        except Exception as e:
            print(f"[FAIL] {os.path.basename(czi)}: {type(e).__name__}: {e}")
            fail.append((czi, str(e)))

    print(f"\n[SUMMARY] {len(ok)} succeeded, {len(fail)} failed.")
    for czi, err in fail:
        print(f"   - {os.path.basename(czi)}: {err}")
    return ok, fail


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert .czi to ImageJ .tif (file or folder).")
    ap.add_argument("path", help="a .czi file OR a folder containing .czi files")
    ap.add_argument("output", nargs="?", default=None,
                    help="output .tif (single-file mode only)")
    ap.add_argument("--recursive", "-r", action="store_true",
                    help="recurse into subfolders (folder mode)")
    ap.add_argument("--outdir", "-o", default=None,
                    help="write .tif files here instead of next to each .czi")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip files whose .tif already exists")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        _, fail = convert_folder(args.path, outdir=args.outdir,
                                 recursive=args.recursive,
                                 skip_existing=args.skip_existing)
        sys.exit(1 if fail else 0)
    else:
        convert(args.path, args.output)