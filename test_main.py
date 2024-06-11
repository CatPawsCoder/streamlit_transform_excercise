import io
from PIL import Image
import pytest
from main import preprocess_image, process_and_display


@pytest.fixture
def sample_image():
    # Create a sample image for testing
    image = Image.new("RGB", (512, 512), color="white")
    return image


def test_preprocess_image(sample_image):
    # Test preprocess_image function
    preprocessed_image = preprocess_image(sample_image)
    assert preprocessed_image.size == (512, 512)
