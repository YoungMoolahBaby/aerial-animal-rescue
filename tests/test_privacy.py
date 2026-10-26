from PIL import Image

from aerial_animal_rescue.privacy import GPS_IFD_TAG, copy_without_gps, has_gps
from tests.conftest import make_image

XMP = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
    b'drone-dji:GpsLatitude="52.1" drone-dji:GpsLongitude="5.2"/></rdf:RDF></x:xmpmeta>'
)


def make_gps_jpeg(path):
    make_image(path.with_suffix(".png"))
    with Image.open(path.with_suffix(".png")) as im:
        exif = im.getexif()
        exif[0x010F] = "DJI"  # Make: should survive
        gps = exif.get_ifd(GPS_IFD_TAG)
        gps[1] = "N"
        gps[2] = (52.0, 6.0, 0.0)
        gps[3] = "E"
        gps[4] = (5.0, 12.0, 0.0)
        im.convert("RGB").save(path, exif=exif, xmp=XMP)
    return path


def test_fixture_really_has_gps(tmp_path):
    assert has_gps(make_gps_jpeg(tmp_path / "a.jpg"))


def test_gps_exif_and_xmp_are_stripped(tmp_path):
    src = make_gps_jpeg(tmp_path / "a.jpg")
    dst = tmp_path / "out" / "a.jpg"
    copy_without_gps(src, dst)
    assert not has_gps(dst)
    data = dst.read_bytes().lower()
    assert b"gpslatitude" not in data
    with Image.open(dst) as im:
        assert im.getexif().get(0x010F) == "DJI"  # non-location EXIF is kept
        assert im.size == (64, 48)


def test_clean_image_is_copied_unchanged(tmp_path):
    src = make_image(tmp_path / "clean.png")
    dst = tmp_path / "out" / "clean.png"
    copy_without_gps(src, dst)
    assert dst.read_bytes() == src.read_bytes()
