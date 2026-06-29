"""/events/stream — Server-Sent Events bridge from the backend event bus to the UI.

Lets the UI watch backend activity in real time — including actions driven by
an external agent over MCP. When mesh decompose / CAD build / context events
fire on the bus, they stream to every connected UI as SSE.

Connect from the renderer:
    const es = new EventSource('http://127.0.0.1:8765/events/stream');
    es.onmessage = e => { const {topic, payload} = JSON.parse(e.data); ... };
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from orynd_core.services.event_bus import bus

router = APIRouter(prefix="/events", tags=["events"])

# Topics the UI cares about (model lifecycle + workspace activity).
UI_TOPICS = [
    "model.ready",       # CAD/mesh model produced → {stl_url, step_url, session_id}
    "decompose.done",    # mesh decompose finished → {file, summary, regions}
    "decompose.running",
    "cad.built",
    "context.chip",
    "drop2.complete",
    "workspace.event",
    # Wire #4: pipeline block events → frontend
    "credits.consumed",           # tool was charged → {session_id, tool, cost, session_total}
    "learning.hint",              # Learning Engine has a suggestion → {session_id, suggestions}
    "action.recorded",            # any tool call logged → {tool, params, session_id}
    # Wire #6: library events → frontend
    "library.article.published",  # deep research saved → {topic, article_id, session_id}
]


@router.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    unsubs = []

    def make_listener(topic: str):
        async def listener(payload: dict) -> None:
            await queue.put({"topic": topic, "payload": payload})
        return listener

    for topic in UI_TOPICS:
        unsubs.append(bus.subscribe(topic, make_listener(topic)))

    async def gen():
        try:
            yield "event: ready\ndata: {}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            for u in unsubs:
                u()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
