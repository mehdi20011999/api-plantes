import io
import pytest
from PIL import Image


def image_bytes(fmt="PNG", size=(320, 240), color=(40, 160, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def png():
    return image_bytes("PNG")


@pytest.fixture
def jpeg():
    return image_bytes("JPEG")
