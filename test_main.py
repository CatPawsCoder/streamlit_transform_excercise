# test_main.py

from PIL import Image
import pytest
from main import preprocess_image, process_and_display


@pytest.fixture
def sample_image():
    # Create a sample image for testing
    image = Image.new("RGB", (512, 512), color="white")
    return image


@pytest.fixture
def small_image():
    # Create a smaller image for testing
    image = Image.new("RGB", (256, 256), color="blue")
    return image


@pytest.fixture
def large_image():
    # Create a larger image for testing
    image = Image.new("RGB", (1024, 1024), color="green")
    return image


def test_preprocess_image(sample_image):
    # Test preprocess_image function
    preprocessed_image = preprocess_image(sample_image)
    assert preprocessed_image.size == (512, 512)


def test_preprocess_small_image(small_image):
    # Test preprocess_image function with a smaller image
    preprocessed_image = preprocess_image(small_image)
    assert preprocessed_image.size == (512, 512)


def test_preprocess_large_image(large_image):
    # Test preprocess_image function with a larger image
    preprocessed_image = preprocess_image(large_image)
    assert preprocessed_image.size == (512, 512)


def test_process_and_display(sample_image):
    # Test process_and_display function with a sample image
    preprocessed_image = preprocess_image(sample_image)
    # This test will check if process_and_display runs without errors
    try:
        process_and_display(preprocessed_image)
    except Exception as e:
        pytest.fail(f"process_and_display raised an exception: {e}")
