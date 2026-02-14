"""
WebSocket endpoint for real-time agent step streaming.

The frontend connects here to see each agent step as it happens:
  - Step started (with tool name)
  - Step completed (with output summary)
  - Step failed (with error)
  - Final report ready
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agent.orchestrator import Orchestrator
from app.models.schemas import CaseSubmission

router = APIRouter()


@router.websocket("/agent")
async def agent_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent pipeline execution.

    Protocol:
      Client sends: JSON with patient case data (CaseSubmission format)
      Server sends: JSON messages for each step update and final report

    Message types:
      - {"type": "step_update", "step": {...}}
      - {"type": "report", "report": {...}}
      - {"type": "error", "message": "..."}
      - {"type": "complete", "case_id": "..."}
    """
    await websocket.accept()

    try:
        # Receive the case submission
        raw = await websocket.receive_text()
        data = json.loads(raw)
        case = CaseSubmission(**data)

        # Send acknowledgment
        await websocket.send_json({
            "type": "ack",
            "message": "Case received. Starting agent pipeline...",
        })

        # Run the orchestrator and stream updates
        orchestrator = Orchestrator()

        async for step in orchestrator.run(case):
            await websocket.send_json({
                "type": "step_update",
                "step": step.model_dump(mode="json"),
            })

        # Send final report
        report = orchestrator.get_result()
        if report:
            await websocket.send_json({
                "type": "report",
                "report": report.model_dump(mode="json"),
            })

        # Send completion
        await websocket.send_json({
            "type": "complete",
            "case_id": orchestrator.state.case_id if orchestrator.state else "unknown",
        })

    except WebSocketDisconnect:
        pass
    except json.JSONDecodeError:
        await websocket.send_json({
            "type": "error",
            "message": "Invalid JSON received",
        })
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
