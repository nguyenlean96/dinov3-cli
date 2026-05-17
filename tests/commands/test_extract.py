import time
import pytest
from typer.testing import CliRunner
import numpy as np
from PIL import Image

from dinov3_cli.cli import app

runner = CliRunner()

@pytest.fixture(autouse=True)
def pause_between_tests() -> None:
    time.sleep(2)

def test_extract_command_help():
    """Test that the help message displays correctly."""
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    assert "Extract dense features from images" in result.output
    assert "--pool" in result.output
    assert "--model" in result.output

def test_extract_command_mocked(mocker, tmp_path):
    """Test the extract command with mocked models to avoid downloading weights."""
    
    # 1. Setup mock image
    test_img_path = tmp_path / "test_image.jpg"
    img = Image.new('RGB', (224, 224), color='red')
    img.save(test_img_path)
    
    # 2. Mock ModelLoader and FeatureExtractor
    mock_model = mocker.MagicMock()
    mock_processor = mocker.MagicMock()
    
    mocker.patch(
        "dinov3_cli.commands.extract.ModelLoader.load",
        return_value=(mock_model, mock_processor)
    )
    
    # Mock FeatureExtractor to return a dummy numpy array
    dummy_features = np.array([[[0.1, 0.2, 0.3]]]) # Shape (1, 1, 3)
    mock_extractor_instance = mocker.MagicMock()
    mock_extractor_instance.extract.return_value = dummy_features
    
    mocker.patch(
        "dinov3_cli.commands.extract.FeatureExtractor",
        return_value=mock_extractor_instance
    )
    
    # 3. Test saving to .npy output
    out_npy = tmp_path / "out.npy"
    result = runner.invoke(app, ["extract", str(test_img_path), "-o", str(out_npy)])
    
    assert result.exit_code == 0
    assert out_npy.exists()
    
    loaded_features = np.load(out_npy)
    assert np.allclose(loaded_features, dummy_features)

    # 4. Verify FeatureExtractor was called with correct arguments
    mock_extractor_instance.extract.assert_called_once()
    _, kwargs = mock_extractor_instance.extract.call_args
    assert kwargs.get("pool") is False # default
