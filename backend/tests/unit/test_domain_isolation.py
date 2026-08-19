"""Architectural guard: inner layers stay free of web-framework dependencies.

Two complementary checks per layer:

* a subprocess import probe, which catches *transitive* dependencies -- a module
  importing something that in turn imports a web framework;
* an AST scan of the source, which states the rule declaratively. It inspects
  import statements only, so prose mentioning a framework in a docstring is not
  a violation.

The subprocess is required because the test session itself imports FastAPI via
the app fixtures, so an in-process ``sys.modules`` check would always fail.
"""

import ast
import subprocess
import sys
from pathlib import Path

# The domain layer is pure: no web framework, no network client, no LLM SDK.
DOMAIN_FORBIDDEN = ("fastapi", "starlette", "uvicorn", "httpx", "httpx2", "requests", "anthropic")

# Adapters may legitimately use an HTTP client to reach a provider; what they
# must never do is depend on the web framework serving our own API.
ADAPTER_FORBIDDEN = ("fastapi", "starlette", "uvicorn")

PROBE = """
import sys

{imports}

forbidden = {forbidden!r}
loaded = sorted({{m for m in sys.modules if m.split(".")[0] in forbidden}})
print(",".join(loaded))
"""

DOMAIN_IMPORTS = "\n".join(
    f"import app.domain.{name}" for name in ("chunker", "errors", "models", "parser", "timestamps")
)

ADAPTER_IMPORTS = "\n".join(
    f"import app.adapters.{name}"
    for name in (
        "embeddings.base",
        "embeddings.fake",
        "embeddings.local",
        "vectorstore.base",
        "vectorstore.memory",
    )
)


def imported_roots(source: Path) -> set[str]:
    """Top-level package names imported by a module, from its AST."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def assert_no_forbidden_imports(probe: str, forbidden: tuple[str, ...], layer: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = result.stdout.strip()
    assert loaded == "", f"{layer} layer pulled in forbidden modules: {loaded}"


def test_domain_imports_no_web_or_network_packages():
    assert_no_forbidden_imports(
        PROBE.format(imports=DOMAIN_IMPORTS, forbidden=DOMAIN_FORBIDDEN),
        DOMAIN_FORBIDDEN,
        "domain",
    )


def test_adapters_import_no_web_framework():
    assert_no_forbidden_imports(
        PROBE.format(imports=ADAPTER_IMPORTS, forbidden=ADAPTER_FORBIDDEN),
        ADAPTER_FORBIDDEN,
        "adapter",
    )


def test_domain_sources_declare_no_forbidden_imports():
    import app.domain

    for source in sorted(Path(app.domain.__file__).parent.glob("*.py")):
        offending = imported_roots(source) & set(DOMAIN_FORBIDDEN)
        assert not offending, f"{source.name} imports {sorted(offending)}"


def test_adapter_sources_declare_no_web_framework_imports():
    import app.adapters

    for source in sorted(Path(app.adapters.__file__).parent.rglob("*.py")):
        offending = imported_roots(source) & set(ADAPTER_FORBIDDEN)
        assert not offending, f"{source.name} imports {sorted(offending)}"
