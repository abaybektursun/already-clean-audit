from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "infra/aws_parallax/full_science_build_user_data.sh"


def test_full_science_builder_is_syntax_valid_and_immutable():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text()
    assert "sha256:e30143429c80fe75b1bc5aa2fc4e3c076c93a98aca6bad12605c9e32b2e08de0" in text
    assert "ea8c94aa04e2723be8bd5d22abb2b1979e0f58be" in text
    assert 'test "$(find . -type f | wc -l)" -eq 271' in text
    assert 'test "$(wc -l < "$BUILD/source-keys.txt")" -eq 271' in text
    assert '"s3://${BUCKET}/${PREFIX}/source-keys.txt"' in text
    assert '"s3://${BUCKET}/${PREFIX}/source/" "$BUILD/source/" --recursive' in text
    assert "list-objects" not in text
    assert 'cmp "$BUILD/expected-source-paths.txt" "$BUILD/observed-source-paths.txt"' in text
    assert "source-sha256sums.txt" in text


def test_full_science_builder_gates_push_on_all_cpu_checks():
    text = SCRIPT.read_text()
    push = text.index('docker push "$IMAGE"')
    for required in (
        "python3 -m pip check",
        "python3 -m infra.vast.parallax.cli dag-status",
        "python3 -m infra.vast.parallax.image_smoke",
        "source-sha256sums.txt",
    ):
        assert text.index(required) < push
    assert "shutdown -h +45" in text
    assert "WAITING_FOR_EXTERNAL_REAPER_RELEASE" in text
    assert "RESULTS_DURABLE" in text
