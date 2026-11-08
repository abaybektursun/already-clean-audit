from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "infra/aws_parallax/full_science_resolve_user_data.sh"


def test_full_science_resolver_executes_disposable_container_before_inspection():
    text = SCRIPT.read_text(encoding="utf-8")
    create = text.index("docker create --name parallax-full-science-rehearsal")
    start = text.index("docker start -a parallax-full-science-rehearsal")
    inspect = text.index("docker inspect --size parallax-full-science-rehearsal")
    assert create < start < inspect
    assert "docker push" not in text


def test_full_science_resolver_is_shell_syntax_valid_and_lock_bound():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "04f40dd87d925df19a1d2e8797e2c52147a714d620768f64a6d4ec80aa627e9b" in text
    assert "GREEN_FULL_SCIENCE_CPU_REHEARSAL" in text
    assert "shutdown -h +35" in text
    assert "trap finish EXIT" in text
