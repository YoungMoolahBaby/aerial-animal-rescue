"""Privacy on ingest: remove location data from images.

Footage can show private property, so we never keep precise coordinates.
Every image entering the dataset passes through `copy_without_gps`, which
re-saves it without the EXIF GPS block and without XMP (DJI drones write GPS
into XMP as well). Other EXIF (camera, timestamp, orientation) is kept.

Note: re-saving drops proprietary blocks such as DJI radiometric thermal data
(R-JPEG). Keep the untouched originals in data/raw/ (never published) if you
need temperatures later.
"""

import shutil
from pathlib import Path

from PIL import Image

GPS_IFD_TAG = 0x8825
_XMP_KEYS = ("xmp", "XML:com.adobe.xmp")
_GPS_XMP_MARKERS = (b"gpslatitude", b"gpslongitude", b"exif:gps")


def copy_without_gps(src: str | Path, dst: str | Path) -> None:
    """Copy an image from `src` to `dst`, stripping GPS EXIF and XMP."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        exif = im.getexif()
        has_xmp = any(k in im.info for k in _XMP_KEYS)
        if GPS_IFD_TAG not in exif and not has_xmp and src.suffix.lower() == dst.suffix.lower():
            shutil.copyfile(src, dst)  # nothing to strip: keep bytes identical
            return
        exif.pop(GPS_IFD_TAG, None)
        for k in _XMP_KEYS:
            im.info.pop(k, None)
        kwargs = {"exif": exif.tobytes()}
        if im.format == "JPEG" and dst.suffix.lower() in {".jpg", ".jpeg"}:
            kwargs["quality"] = "keep"  # reuse the original quantization: minimal loss
        im.save(dst, **kwargs)


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
