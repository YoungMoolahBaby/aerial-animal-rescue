"""Privacy on ingest: remove location data from images.

Footage can show private property, so we never keep precise coordinates.
Every image entering the dataset passes through `copy_without_gps`, which
removes the EXIF GPS block and XMP (DJI drones write GPS into XMP as well),
plus EXIF fields that identify the device or its owner (serial numbers). Other
EXIF (camera model, timestamp, orientation) is kept.

JPEGs (including DJI's MPO files) are rewritten segment by segment, so the
compressed image data is copied byte for byte: no re-encoding, no quality loss.
Only the metadata blocks needed to display the image are kept (JFIF, EXIF
without GPS, ICC profile, Adobe colour info); proprietary blocks such as DJI's
debug data, signatures and R-JPEG thermal data are dropped, as are MPO preview
images (they carry their own EXIF). Other formats are re-saved with Pillow.
Keep the untouched originals in data/raw/ (never published) if you need them.
"""

import shutil
from pathlib import Path

from PIL import Image

GPS_IFD_TAG = 0x8825
EXIF_IFD_TAG = 0x8769
# Exif sub-IFD tags that identify a device or person: ImageUniqueID,
# CameraOwnerName, BodySerialNumber, LensSerialNumber
_IDENTIFYING_TAGS = (0xA420, 0xA430, 0xA431, 0xA435)
_XMP_KEYS = ("xmp", "XML:com.adobe.xmp")
_GPS_XMP_MARKERS = (b"gpslatitude", b"gpslongitude", b"exif:gps")
# APPn / comment blocks we keep: (marker, payload prefix). Everything else is dropped.
_KEEP_APP = ((0xE0, b"JFIF\0"), (0xE1, b"Exif\0\0"), (0xE2, b"ICC_PROFILE\0"), (0xEE, b"Adobe"))


def _clean_exif(exif: Image.Exif) -> Image.Exif:
    """Drop the GPS block and identifying tags, in place."""
    exif.pop(GPS_IFD_TAG, None)
    sub_ifd = exif.get_ifd(EXIF_IFD_TAG)
    for tag in _IDENTIFYING_TAGS:
        sub_ifd.pop(tag, None)
    return exif


def _strip_jpeg(data: bytes) -> bytes:
    """Rewrite a JPEG/MPO keeping only `_KEEP_APP` metadata (EXIF without GPS).

    The compressed scan data is copied as is.

    Stops at the first image's EOI, which drops any MPO preview images after it.
    """
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    out, i = [b"\xff\xd8"], 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError(f"bad JPEG marker at byte {i}")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker == 0xD9:  # EOI: end of the first image
            out.append(b"\xff\xd9")
            return b"".join(out)
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:  # markers without a length
            out.append(data[i : i + 2])
            i += 2
            continue
        end = i + 2 + int.from_bytes(data[i + 2 : i + 4], "big")
        segment, payload = data[i:end], data[i + 4 : end]
        i = end
        if 0xE0 <= marker <= 0xEF or marker == 0xFE:  # APPn or comment
            if not any(marker == m and payload.startswith(p) for m, p in _KEEP_APP):
                continue  # XMP (DJI writes GPS there), MPF, maker blocks, comments
            if marker == 0xE1:
                exif = Image.Exif()
                exif.load(payload)
                new = _clean_exif(exif).tobytes()
                segment = b"\xff\xe1" + (len(new) + 2).to_bytes(2, "big") + new
        out.append(segment)
        if marker == 0xDA:  # start of scan: copy entropy-coded data up to the next marker
            j = i
            while True:
                j = data.index(b"\xff", j)
                nxt = data[j + 1]
                if nxt == 0x00 or 0xD0 <= nxt <= 0xD7:  # stuffed byte or restart marker
                    j += 2
                    continue
                break
            out.append(data[i:j])
            i = j
    raise ValueError("JPEG ended without EOI")


def copy_without_gps(src: str | Path, dst: str | Path) -> None:
    """Copy an image from `src` to `dst`, stripping GPS EXIF and XMP."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        exif = im.getexif()
        has_xmp = any(k in im.info for k in _XMP_KEYS)
        identifying = any(t in exif.get_ifd(EXIF_IFD_TAG) for t in _IDENTIFYING_TAGS)
        clean = GPS_IFD_TAG not in exif and not has_xmp and not identifying
        if clean and src.suffix.lower() == dst.suffix.lower():
            shutil.copyfile(src, dst)  # nothing to strip: keep bytes identical
            return
        if im.format in {"JPEG", "MPO"} and dst.suffix.lower() in {".jpg", ".jpeg"}:
            dst.write_bytes(_strip_jpeg(src.read_bytes()))
            return
        for k in _XMP_KEYS:
            im.info.pop(k, None)
        im.save(dst, exif=_clean_exif(exif).tobytes())


def has_gps(path: str | Path) -> bool:
    """True if the image still carries GPS in EXIF or XMP."""
    with Image.open(path) as im:
        if im.getexif().get_ifd(GPS_IFD_TAG):
            return True
        for k in _XMP_KEYS:
            xmp = im.info.get(k)
            if xmp:
                data = (xmp if isinstance(xmp, bytes) else str(xmp).encode()).lower()
                if any(m in data for m in _GPS_XMP_MARKERS):
                    return True
    return False
