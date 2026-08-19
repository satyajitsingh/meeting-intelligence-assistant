"""Architectural guard: the domain layer stays free of web and network deps.

Run in a subprocess because the test session itself imports FastAPI via the app
fixtures. A subprocess also catches *transitive* imports -- a domain module
pulling in something that in turn imports a web framework.
"""

import subprocess
import sys

FORBIDDEN = ("fastapi", "starlette", "uvicorn", "httpx", "httpx2", "requests", "anthropic")

PROBE = """
import sys

import app.domain.chunker
import app.domain.errors
import app.domain.models
import app.domain.parser
import app.domain.timestamps

forbidden = {forbidden!r}
loaded = sorted({{m for m in sys.modules if m.split(".")[0] in forbidden}})
print(",".join(loaded))
"""


def test_domain_imports_no_web_or_network_packages():
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(forbidden=FORBIDDEN)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "", (
        f"domain layer pulled in forbidden modules: {result.stdout.strip()}"
    )


def test_domain_modules_do_not_reference_fastapi_in_source():
    """Cheap textual backstop, so the intent is visible even if imports move."""
    from pathlib import Path

    import app.domain

    domain_dir = Path(app.domain.__file__).parent
    for source in sorted(domain_dir.glob("*.py")):
        text = source.read_text()
        assert "fastapi" not in text.lower(), f"{source.name} references FastAPI"
