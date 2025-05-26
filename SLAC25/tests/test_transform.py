import random
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2
from SLAC25.transform import TransformV1, tfConfig

@pytest.fixture
def dummy_image():
    # Create a simple white image of fixed size
    return Image.new("RGB", (256, 256), "white")

def test_default_tf_config():
    # When no configuration is provided, default should be tfConfig.
    transformer = TransformV1()
    assert transformer.tf_config == tfConfig

def test_preprocessing_pipeline(dummy_image):
    # Test that the preprocessing pipeline produces a tensor of shape (C, 512, 512)
    transformer = TransformV1()
    processed = transformer.preprocessing(dummy_image)
    assert isinstance(processed, torch.Tensor)
    # Assuming a color image, expect 3 channels with height and width 512.
    assert processed.shape[0] == 3
    assert processed.shape[1:] == (512, 512)
    assert processed.dtype == torch.float32

def test_record_transform_disabled(dummy_image):
    # When recordTransform is False, the log should remain disabled (None).
    transformer = TransformV1(recordTransform=False)
    # Call one transform method (e.g., rotation)
    transformer._random_rotation(dummy_image, "test")
    log = transformer._getLog("test")
    assert log is None

def test_random_rotation_transformation(dummy_image, monkeypatch):
    # Force random.random to return a value that triggers a rotation (< 0.5)
    monkeypatch.setattr(random, "random", lambda: 0.1)
    # Force a fixed angle by patching random.uniform (returns the midpoint)
    monkeypatch.setattr(random, "uniform", lambda a, b: (a + b) / 2)
    
    transformer = TransformV1(recordTransform=True)
    # Apply random rotation; expecting the log to record the rotation.
    transformer._random_rotation(dummy_image, "rotation_log")
    log = transformer._getLog("rotation_log")
    assert log is not None
    assert "Rotating by" in log

def test_random_rotation_no_transformation(dummy_image, monkeypatch):
    # Force random.random to return a value that does not trigger rotation (>= 0.5)
    monkeypatch.setattr(random, "random", lambda: 0.9)
    
    transformer = TransformV1(recordTransform=True)
    transformer._random_rotation(dummy_image, "no_rotation_log")
    log = transformer._getLog("no_rotation_log")
    # Since the transformation did not trigger, no log should be recorded.
    assert log is None

def test_random_gaussian_blur_transformation(dummy_image, monkeypatch):
    # Force gaussian blur to be applied: random.random() < 0.5
    monkeypatch.setattr(random, "random", lambda: 0.1)
    
    transformer = TransformV1(recordTransform=True)
    transformer._random_gaussian_blur(dummy_image, "blur_log")
    log = transformer._getLog("blur_log")
    assert log is not None
    assert "Gaussian Blur" in log

def test_random_horizontal_flip_transformation(dummy_image, monkeypatch):
    # Force horizontal flip to be applied.
    monkeypatch.setattr(random, "random", lambda: 0.1)
    
    transformer = TransformV1(recordTransform=True)
    transformer._random_horizontal_flip(dummy_image, "hflip_log")
    log = transformer._getLog("hflip_log")
    assert log is not None
    assert "Horizontal Flip" in log

def test_random_vertical_flip_transformation(dummy_image, monkeypatch):
    # Force vertical flip to be applied.
    # Note: The vertical flip method checks the same "horizontal_flip" flag in tf_config.
    monkeypatch.setattr(random, "random", lambda: 0.1)
    
    transformer = TransformV1(recordTransform=True)
    transformer._random_vertical_flip(dummy_image, "vflip_log")
    log = transformer._getLog("vflip_log")
    assert log is not None
    assert "Vertical Flip" in log

def test_random_sharpening_transformation(dummy_image, monkeypatch):
    # Force sharpening transformation by making random.random() return a value below 0.5.
    monkeypatch.setattr(random, "random", lambda: 0.1)

    from torchvision.transforms import v2
    monkeypatch.setattr(v2, "RandomAdjustSharpness", lambda *args, **kwargs: (lambda img: img))
    
    transformer = TransformV1(recordTransform=True)
    transformer._random_sharpening(dummy_image, "sharpen_log")
    log = transformer._getLog("sharpen_log")
    assert log is not None
    assert "Sharpening" in log