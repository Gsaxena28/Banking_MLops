from src.train import build_pipeline


def test_pipeline_contains_preprocessor_and_classifier():
    pipeline = build_pipeline()
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
