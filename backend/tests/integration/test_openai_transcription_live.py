"""Live OpenAI transcription check.

Excluded from the default run: it needs OPENAI_API_KEY and makes a real,
billable request. Run explicitly with::

    OPENAI_API_KEY=sk-... pytest -m integration

No audio fixture is committed -- binary clutter in the repository is not worth
it for a check that cannot run in CI anyway. Point AUDIO_FIXTURE_ENV at any
short recording to exercise it::

    MEETING_AUDIO_FIXTURE=/path/to/clip.m4a OPENAI_API_KEY=sk-... pytest -m integration
"""

import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.adapters.stt.openai import OpenAISpeechToTextProvider

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

AUDIO_FIXTURE_ENV = "MEETING_AUDIO_FIXTURE"


@pytest.fixture
def provider():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set")
    return OpenAISpeechToTextProvider(api_key=SecretStr(api_key))


@pytest.fixture
def audio_file() -> Path:
    configured = os.environ.get(AUDIO_FIXTURE_ENV)
    if not configured:
        pytest.skip(f"{AUDIO_FIXTURE_ENV} is not set; no audio fixture is committed")

    path = Path(configured)
    if not path.is_file():
        pytest.skip(f"{AUDIO_FIXTURE_ENV} does not point at a file: {path}")
    return path


async def test_transcribes_a_real_recording(provider, audio_file):
    result = await provider.transcribe(
        audio=audio_file.read_bytes(),
        filename=audio_file.name,
        content_type=None,
    )

    assert result.text.strip()
    # whisper-1 reports both; the gpt-4o-transcribe family reports neither.
    if provider.model == "whisper-1":
        assert result.language
        assert result.duration_seconds is not None
        assert result.duration_seconds > 0
