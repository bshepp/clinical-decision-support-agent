"use client";

import { useCallback, useRef, useState } from "react";

interface Step {
  step_id: string;
  step_name: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  tool_name?: string;
  output_summary?: string;
  duration_ms?: number;
  error?: string;
}

interface CaseSubmission {
  patient_text: string;
  include_drug_check: boolean;
  include_guidelines: boolean;
}

interface UseAgentWebSocketReturn {
  steps: Step[];
  report: any | null;
  isRunning: boolean;
  error: string | null;
  submitCase: (submission: CaseSubmission) => void;
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8002/ws/agent";

export function useAgentWebSocket(): UseAgentWebSocketReturn {
  const [steps, setSteps] = useState<Step[]>([]);
  const [report, setReport] = useState<any | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const submitCase = useCallback((submission: CaseSubmission) => {
    // Reset state
    setSteps([]);
    setReport(null);
    setError(null);
    setIsRunning(true);

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify(submission));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "ack":
          // Pipeline acknowledged
          break;

        case "step_update":
          setSteps((prev) => {
            const existing = prev.findIndex(
              (s) => s.step_id === data.step.step_id
            );
            if (existing >= 0) {
              const updated = [...prev];
              updated[existing] = data.step;
              return updated;
            }
            return [...prev, data.step];
          });
          break;

        case "report":
          setReport(data.report);
          break;

        case "complete":
          setIsRunning(false);
          break;

        case "error":
          setError(data.message);
          setIsRunning(false);
          break;
      }
    };

    ws.onerror = () => {
      setError("WebSocket connection failed. Is the backend running?");
      setIsRunning(false);
    };

    ws.onclose = () => {
      setIsRunning(false);
    };
  }, []);

  return { steps, report, isRunning, error, submitCase };
}
