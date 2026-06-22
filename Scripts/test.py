p = "/Users/sophia.betzler/Desktop/12-chief-dog_montage_20260616-07-47-11.mrc.mdoc"
with open(p, "rb") as fh:               # binary, so nothing is hidden
    raw = fh.read(300)
print(raw[:4])                          # look for a BOM: b'\xff\xfe', b'\xef\xbb\xbf', etc.
print(repr(raw[:120]))