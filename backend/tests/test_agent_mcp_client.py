import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.agents.mcp_client as mcp_client


def test_get_resource_payload_decodes_text_resource(monkeypatch):
    class FakeContent:
        text = json.dumps({"status": "ok", "value": 123})

    class FakeClient:
        def __init__(self, runtime):
            self.runtime = runtime

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def read_resource(self, uri):
            return [FakeContent()]

    monkeypatch.setattr(mcp_client, "build_runtime_server", lambda db_path=None: object())
    monkeypatch.setattr(mcp_client, "Client", FakeClient)

    payload = mcp_client.get_resource_payload("planner://budget/active-plan")

    assert payload == {"status": "ok", "value": 123}
