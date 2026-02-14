"""Quick script to poll a case result."""
import httpx
import asyncio
import json
import sys

async def main():
    case_id = sys.argv[1] if len(sys.argv) > 1 else "55a04557"
    r = await httpx.AsyncClient(timeout=30).get(f"http://localhost:8000/api/cases/{case_id}")
    d = r.json()
    steps = d.get("state", {}).get("steps", [])
    for s in steps:
        err = s.get("error", "")
        print(f"  {s['step_id']:12s} => {s['status']:10s} ({s.get('duration_ms','?')}ms) {err[:80] if err else ''}")
    
    report = d.get("report")
    if report:
        print("\n=== REPORT (truncated) ===")
        print(json.dumps(report, indent=2, default=str)[:3000])
    else:
        print("\nNo report yet.")

asyncio.run(main())
