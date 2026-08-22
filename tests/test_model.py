"""The optional model channel: request shape, grounding, advisory red-only scoring, errors.

No real model is involved: a scripted stdlib HTTP server plays the OpenAI-compatible endpoint,
so these tests pin the CONTRACT (what is sent, how replies are parsed, which failures raise)
hermetically. The channel must stay opt-in: with no ``ModelConfig`` nothing here may run, which
``test_scorer_without_config_makes_no_call`` pins by running with no server at all.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from pii_kit import (
    UNIVERSAL_PATTERNS,
    ModelAPIError,
    ModelConfig,
    model_image_findings,
    model_image_leak,
    model_leak,
    model_redact,
    model_text_findings,
    score_pii_safety,
)

# 1x1 PNG / minimal JPEG magic prefixes; enough for media-type sniffing, no decoder involved.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 16


class MockModelServer:
    """A scripted OpenAI-compatible endpoint: queue replies, capture what the client sent."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.paths: list[str] = []
        self._replies: list[tuple[int, bytes]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                outer.requests.append(json.loads(self.rfile.read(length)))
                outer.headers.append({k: v for k, v in self.headers.items()})
                outer.paths.append(self.path)
                status, body = outer._replies.pop(0) if outer._replies else (500, b"exhausted")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass  # keep pytest output clean

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def queue_content(self, content: str) -> None:
        """Queue a 200 reply whose assistant message content is ``content``."""
        envelope = {"choices": [{"message": {"role": "assistant", "content": content}}]}
        self._replies.append((200, json.dumps(envelope).encode()))

    def queue_findings(self, *pairs: tuple[str, str]) -> None:
        self.queue_content(json.dumps({"findings": [{"type": t, "text": v} for t, v in pairs]}))

    def queue_raw(self, status: int, body: bytes) -> None:
        self._replies.append((status, body))

    @property
    def config(self) -> ModelConfig:
        host, port = self._server.server_address[:2]
        return ModelConfig(base_url=f"http://{host}:{port}/v1", model="test-model")

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def server() -> Iterator[MockModelServer]:
    srv = MockModelServer()
    yield srv
    srv.close()


class TestRequestShape:
    def test_text_request_is_openai_chat_completions(self, server: MockModelServer) -> None:
        server.queue_findings()
        model_text_findings("Contact John Smith.", server.config)
        assert server.paths == ["/v1/chat/completions"]
        body = server.requests[0]
        assert body["model"] == "test-model"
        assert body["temperature"] == 0.0
        # Default is schema-constrained decoding, portable across vLLM/Ollama/llama.cpp.
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"]["required"] == ["findings"]
        (message,) = body["messages"]  # ONE user turn: Gemma chat templates have no system role
        assert message["role"] == "user"
        assert "Contact John Smith." in message["content"]

    def test_response_format_json_object_mode(self, server: MockModelServer) -> None:
        server.queue_findings()
        cfg = ModelConfig(
            base_url=server.config.base_url, model="test-model", response_format="json_object"
        )
        model_leak("some text", cfg)
        assert server.requests[0]["response_format"] == {"type": "json_object"}

    def test_api_key_becomes_bearer_header_and_absence_sends_none(
        self, server: MockModelServer
    ) -> None:
        server.queue_findings()
        server.queue_findings()
        model_leak("x y z", server.config)
        cfg = ModelConfig(
            base_url=server.config.base_url, model="test-model", api_key="sk-local-123"
        )
        model_leak("x y z", cfg)
        assert "Authorization" not in server.headers[0]
        assert server.headers[1]["Authorization"] == "Bearer sk-local-123"

    def test_response_format_none_omits_the_field(self, server: MockModelServer) -> None:
        server.queue_findings()
        cfg = ModelConfig(
            base_url=server.config.base_url, model="test-model", response_format="none"
        )
        model_leak("some text", cfg)
        assert "response_format" not in server.requests[0]

    def test_image_request_carries_base64_data_uri(self, server: MockModelServer) -> None:
        server.queue_findings(("PERSON_NAME", "Jane"))
        findings = model_image_findings(PNG_BYTES, server.config)
        content = server.requests[0]["messages"][0]["content"]
        assert isinstance(content, list) and content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,iVBOR")
        assert [f.grounded for f in findings] == [False], "image findings are never grounded"

    def test_image_media_type_sniffed_or_explicit(self, server: MockModelServer) -> None:
        server.queue_findings()
        model_image_leak(JPEG_BYTES, server.config)
        assert (
            "data:image/jpeg;base64,"
            in server.requests[0]["messages"][0]["content"][1]["image_url"]["url"]
        )
        server.queue_findings()
        model_image_leak(b"\x00\x01rawbytes", server.config, media_type="image/tiff")
        assert (
            "data:image/tiff;base64,"
            in server.requests[1]["messages"][0]["content"][1]["image_url"]["url"]
        )

    def test_unrecognised_image_bytes_without_media_type_raise(
        self, server: MockModelServer
    ) -> None:
        with pytest.raises(ValueError, match="media_type"):
            model_image_findings(b"\x00\x01not-an-image", server.config)
        assert server.requests == [], "nothing may be sent when the format is unknown"

    def test_empty_inputs_never_call_the_endpoint(self, server: MockModelServer) -> None:
        assert model_text_findings("", server.config) == []
        assert model_image_findings(b"", server.config) == []
        assert server.requests == []


class TestGroundingAndParsing:
    def test_verbatim_finding_grounded_paraphrase_kept_ungrounded(
        self, server: MockModelServer
    ) -> None:
        text = "Applicant John Smith, 12 High St."
        server.queue_findings(("PERSON_NAME", "John Smith"), ("POSTAL_ADDRESS", "12 High Street"))
        found = model_text_findings(text, server.config)
        by_type = {f.info_type: f.grounded for f in found}
        # The paraphrase is DETECTION evidence (red-only direction) but never maskable.
        assert by_type == {"PERSON_NAME": True, "POSTAL_ADDRESS": False}

    def test_fenced_and_prose_wrapped_json_still_parse(self, server: MockModelServer) -> None:
        payload = json.dumps({"findings": [{"type": "PERSON_NAME", "text": "Jane"}]})
        server.queue_content(f"```json\n{payload}\n```")
        server.queue_content(f"Here are the findings: {payload}")
        assert model_leak("Jane was here", server.config)
        assert model_leak("Jane was here", server.config)

    def test_findings_object_salvaged_from_multiple_json_objects(
        self, server: MockModelServer
    ) -> None:
        # A chatty reply with a decoy object first: the findings object must still be found.
        payload = json.dumps({"findings": [{"type": "PERSON_NAME", "text": "Jane"}]})
        server.queue_content(f'I scanned it. Metadata: {{"pages": 1}}. Result: {payload} Done.')
        assert model_leak("Jane was here", server.config)

    def test_truncated_json_still_raises(self, server: MockModelServer) -> None:
        server.queue_content('{"findings": [{"type": "PERSON_NAME", "text": "Ja')
        with pytest.raises(ModelAPIError):
            model_leak("Jane was here", server.config)

    def test_type_names_are_normalised(self, server: MockModelServer) -> None:
        server.queue_findings(("person name", "Jane"), ("", "Jane"))
        types = {f.info_type for f in model_text_findings("Jane", server.config)}
        assert types == {"PERSON_NAME", "PII"}

    def test_malformed_entries_dropped_but_reply_survives(self, server: MockModelServer) -> None:
        server.queue_content(
            json.dumps({"findings": [{"type": "X"}, "junk", {"type": "PERSON_NAME", "text": "Jo"}]})
        )
        found = model_text_findings("Jo was here", server.config)
        assert [(f.info_type, f.text) for f in found] == [("PERSON_NAME", "Jo")]

    def test_non_json_reply_raises_not_passes(self, server: MockModelServer) -> None:
        # A truncated/duff reply must be an ERROR: treating it as "no findings" would let a
        # broken model score a vacuous pass on the advertised scan.
        server.queue_content("I could not find anything of note.")
        with pytest.raises(ModelAPIError):
            model_leak("S1234567D", server.config)

    def test_http_error_and_unreachable_endpoint_raise(self, server: MockModelServer) -> None:
        server.queue_raw(503, b"loading model")
        with pytest.raises(ModelAPIError, match="503") as excinfo:
            model_leak("text", server.config)
        assert "loading model" not in str(excinfo.value), "server bodies stay out of messages"
        dead = ModelConfig(base_url="http://127.0.0.1:9/v1", model="m", timeout=0.5)
        with pytest.raises(ModelAPIError, match="unreachable"):
            model_leak("text", dead)

    def test_error_message_never_echoes_scanned_text(self, server: MockModelServer) -> None:
        # The hostile case: a server (or proxy) that ECHOES the scanned request back in its
        # error body. Status-only messages keep the PII out of the exception regardless.
        secret = "S1234567D-super-secret"
        server.queue_raw(400, f"invalid request; input was: {secret}".encode())
        with pytest.raises(ModelAPIError) as excinfo:
            model_leak(secret, server.config)
        assert secret not in str(excinfo.value)


class TestScorerIntegration:
    def test_model_finding_turns_score_red_when_pack_is_green(
        self, server: MockModelServer
    ) -> None:
        # A person name: invisible to every regex row, so only the model channel can go red.
        server.queue_findings(("PERSON_NAME", "John Smith"))
        surfaces = ["Case notes: John Smith prefers evening calls."]
        assert score_pii_safety(surfaces, UNIVERSAL_PATTERNS) == 1.0, "pack alone is green"
        assert score_pii_safety(surfaces, UNIVERSAL_PATTERNS, model_config=server.config) == 0.0, (
            "the model channel must be able to LOWER the score"
        )

    def test_clean_surfaces_stay_green_with_model(self, server: MockModelServer) -> None:
        server.queue_findings()
        assert (
            score_pii_safety(["nothing here"], UNIVERSAL_PATTERNS, model_config=server.config)
            == 1.0
        )

    def test_model_cannot_turn_a_red_score_green(self, server: MockModelServer) -> None:
        # Deterministic halves already failed: the model must not run at all (short-circuit),
        # so it has no way to influence — let alone lift — the score.
        surfaces = ["raw email a@b.com survived"]
        assert score_pii_safety(surfaces, UNIVERSAL_PATTERNS, model_config=server.config) == 0.0
        assert server.requests == []

    def test_scorer_without_config_makes_no_call(self) -> None:
        # No server exists in this test: any HTTP attempt would error, so passing proves the
        # default flow is untouched.
        assert score_pii_safety(["John Smith"], UNIVERSAL_PATTERNS) == 1.0

    def test_dead_endpoint_raises_through_the_gate(self) -> None:
        dead = ModelConfig(base_url="http://127.0.0.1:9/v1", model="m", timeout=0.5)
        with pytest.raises(ModelAPIError):
            score_pii_safety(["clean text"], UNIVERSAL_PATTERNS, model_config=dead)

    def test_every_surface_is_model_scanned(self, server: MockModelServer) -> None:
        # A shortcut that scanned only the first surface would both miss the leak on the last
        # one and show a single request here.
        server.queue_findings()
        server.queue_findings()
        server.queue_findings(("PERSON_NAME", "Wei Ling"))
        surfaces = ["clean one", "clean two", "note: Wei Ling called"]
        assert score_pii_safety(surfaces, UNIVERSAL_PATTERNS, model_config=server.config) == 0.0
        assert len(server.requests) == 3, "every surface must be scanned"


class TestModelRedact:
    def test_masks_grounded_longest_first(self, server: MockModelServer) -> None:
        text = "Report by John Smith; contact Smith after 5."
        server.queue_findings(("PERSON_NAME", "John Smith"), ("PERSON_NAME", "Smith"))
        out = model_redact(text, server.config)
        assert out == ("Report by [REDACTED:PERSON_NAME]; contact [REDACTED:PERSON_NAME] after 5.")

    def test_ungrounded_findings_are_not_masked(self, server: MockModelServer) -> None:
        server.queue_findings(("PERSON_NAME", "Jon Smith"))  # paraphrase not present verbatim
        text = "Report by John Smith."
        assert model_redact(text, server.config) == text

    def test_composes_after_pack_redaction(self, server: MockModelServer) -> None:
        from pii_kit import redact

        server.queue_findings(("PERSON_NAME", "John Smith"))
        staged = redact("John Smith <a@b.com>", UNIVERSAL_PATTERNS)
        assert model_redact(staged, server.config) == (
            "[REDACTED:PERSON_NAME] <[REDACTED:EMAIL_ADDRESS]>"
        )

    def test_value_inside_a_mask_token_cannot_corrupt_it(self, server: MockModelServer) -> None:
        # "ID" occurs inside "[REDACTED:NATIONAL_ID]"; sequential replacement would corrupt
        # the inserted mask. Span-based masking against the original text cannot.
        server.queue_findings(("NATIONAL_ID", "A1234563"), ("REFERENCE", "ID"))
        out = model_redact("ID A1234563", server.config)
        assert out == "[REDACTED:REFERENCE] [REDACTED:NATIONAL_ID]"

    def test_partially_overlapping_findings_merge_into_one_mask(
        self, server: MockModelServer
    ) -> None:
        # Neither finding contains the other; the union must be masked, leaving no fragment.
        server.queue_findings(("PERSON_NAME", "Anna Maria"), ("PERSON_NAME", "Maria Lopez"))
        assert model_redact("Anna Maria Lopez", server.config) == "[REDACTED:PERSON_NAME]"

    def test_identical_text_under_two_types_masks_deterministically(
        self, server: MockModelServer
    ) -> None:
        # Same value, two labels: the lexically-first type must win every run.
        server.queue_findings(("USERNAME", "Jo"), ("PERSON_NAME", "Jo"))
        assert model_redact("login Jo", server.config) == "login [REDACTED:PERSON_NAME]"


class TestConfigFromEnv:
    def test_absent_env_yields_none(self) -> None:
        assert ModelConfig.from_env({}) is None
        assert ModelConfig.from_env({"PII_PACK_MODEL_BASE_URL": "http://x/v1"}) is None
        assert ModelConfig.from_env({"PII_PACK_MODEL": "gemma"}) is None

    def test_full_env_yields_config(self) -> None:
        cfg = ModelConfig.from_env(
            {
                "PII_PACK_MODEL_BASE_URL": "http://localhost:8000/v1",
                "PII_PACK_MODEL": "google/gemma-3-27b-it",
                "PII_PACK_MODEL_API_KEY": "sk-local",
            }
        )
        assert cfg == ModelConfig(
            base_url="http://localhost:8000/v1", model="google/gemma-3-27b-it", api_key="sk-local"
        )

    def test_blank_api_key_normalises_to_none(self) -> None:
        cfg = ModelConfig.from_env(
            {
                "PII_PACK_MODEL_BASE_URL": "http://x/v1",
                "PII_PACK_MODEL": "m",
                "PII_PACK_MODEL_API_KEY": "",
            }
        )
        assert cfg is not None and cfg.api_key is None
