from anomaly_detection.config import load_config,resolve_path

def test_config_from_fixture(tmp_path):
    p=tmp_path/"test.yaml"
    p.write_text("category: screw\nseed: 42\n")
    cfg=load_config(p)
    assert cfg["seed"]==42
    assert resolve_path(cfg,"/tmp/example").as_posix()=="/tmp/example"
