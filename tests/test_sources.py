import pytest

from aerial_animal_rescue.sources import download_file


def test_download_file_rejects_short_download(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 100)
    dest = tmp_path / "out" / "src.bin"
    with pytest.raises(OSError, match="incomplete download"):
        download_file(src.as_uri(), dest, size=1000)  # server promised more than it sent
    assert not dest.exists()
    download_file(src.as_uri(), dest, size=100)
    assert dest.read_bytes() == b"x" * 100
