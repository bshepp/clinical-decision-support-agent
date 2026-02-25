"use client";

import { useEffect, useRef, useState } from "react";

interface Step {
  step_id: string;
  step_name: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  tool_name?: string;
  output_summary?: string;
  duration_ms?: number;
  error?: string;
}

interface AgentPipelineProps {
  steps: Step[];
  isRunning: boolean;
}

const STATUS_CONFIG = {
  pending: {
    icon: "○",
    color: "text-gray-400",
    bg: "bg-gray-50",
    border: "border-gray-200",
  },
  running: {
    icon: "◉",
    color: "text-blue-600",
    bg: "bg-blue-50",
    border: "border-blue-300",
  },
  completed: {
    icon: "✓",
    color: "text-green-600",
    bg: "bg-green-50",
    border: "border-green-300",
  },
  failed: {
    icon: "✗",
    color: "text-red-600",
    bg: "bg-red-50",
    border: "border-red-300",
  },
  skipped: {
    icon: "–",
    color: "text-gray-400",
    bg: "bg-gray-50",
    border: "border-gray-200",
  },
};

const STEP_MESSAGES: Record<string, string[]> = {
  parse: [
    "Extracting demographics, medications, labs...",
    "Identifying chief complaint and history...",
    "Structuring patient data...",
  ],
  reason: [
    "Analyzing symptoms and history...",
    "Generating differential diagnosis...",
    "Evaluating clinical evidence...",
  ],
  drugs: [
    "Querying OpenFDA drug database...",
    "Checking RxNorm interactions...",
    "Evaluating medication safety...",
  ],
  guidelines: [
    "Searching clinical guideline database...",
    "Retrieving relevant excerpts...",
    "Matching guidelines to diagnosis...",
  ],
  conflicts: [
    "Cross-referencing care plan with guidelines...",
    "Detecting care gaps...",
    "Checking for contraindications...",
  ],
  synthesize: [
    "Compiling analysis results...",
    "Generating clinical report...",
    "Finalizing recommendations...",
  ],
};

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

export function AgentPipeline({ steps, isRunning }: AgentPipelineProps) {
  const [tick, setTick] = useState(0);
  const startTimes = useRef<Record<string, number>>({});

  // Track when steps start running
  useEffect(() => {
    const now = Date.now();
    steps.forEach((step) => {
      if (step.status === "running" && !startTimes.current[step.step_id]) {
        startTimes.current[step.step_id] = now;
      }
      if (step.status !== "running" && startTimes.current[step.step_id]) {
        delete startTimes.current[step.step_id];
      }
    });
  }, [steps]);

  // Tick every second when any step is running
  useEffect(() => {
    const hasRunning = steps.some((s) => s.status === "running");
    if (!hasRunning) return;

    const interval = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(interval);
  }, [steps.map((s) => s.status).join(",")]);

  const getElapsed = (stepId: string): number => {
    const start = startTimes.current[stepId];
    if (!start) return 0;
    return Math.floor((Date.now() - start) / 1000);
  };

  const getActivityMessage = (stepId: string, elapsed: number): string => {
    const messages = STEP_MESSAGES[stepId];
    if (!messages) return "Processing...";
    // Cycle through messages every 8 seconds
    const idx = Math.floor(elapsed / 8) % messages.length;
    return messages[idx];
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">
        Agent Pipeline
      </h3>

      <div className="space-y-1">
        {steps.map((step, index) => {
          const config = STATUS_CONFIG[step.status];
          const elapsed = step.status === "running" ? getElapsed(step.step_id) : 0;
          return (
            <div key={step.step_id}>
              {/* Connector line */}
              {index > 0 && (
                <div className="ml-3 h-4 w-px bg-gray-200" />
              )}

              {/* Step card */}
              <div
                className={`flex items-start gap-3 p-3 rounded-lg border ${config.bg} ${config.border} transition-all duration-300`}
              >
                {/* Status icon */}
                <div className="flex-shrink-0 mt-0.5">
                  {step.status === "running" ? (
                    <div className="w-5 h-5 relative">
                      <div className="absolute inset-0 rounded-full border-2 border-blue-200" />
                      <div className="absolute inset-0 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
                    </div>
                  ) : (
                    <span className={`text-lg font-bold ${config.color}`}>
                      {config.icon}
                    </span>
                  )}
                </div>

                {/* Step info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-800">
                      {step.step_name}
                    </span>
                    {step.status === "running" && elapsed > 0 && (
                      <span className="text-xs font-mono text-blue-500 tabular-nums">
                        {formatElapsed(elapsed)}
                      </span>
                    )}
                    {step.duration_ms != null && step.status !== "running" && (
                      <span className="text-xs text-gray-400">
                        {(step.duration_ms / 1000).toFixed(1)}s
                      </span>
                    )}
                  </div>

                  {step.status === "running" && (
                    <p className="text-xs text-blue-500 mt-1 animate-pulse">
                      {getActivityMessage(step.step_id, elapsed)}
                    </p>
                  )}

                  {step.tool_name && step.status !== "running" && (
                    <span className="text-xs text-gray-400 font-mono">
                      {step.tool_name}
                    </span>
                  )}

                  {step.output_summary && (
                    <p className="text-xs text-gray-600 mt-1">
                      {step.output_summary}
                    </p>
                  )}

                  {step.error && (
                    <p className="text-xs text-red-600 mt-1">{step.error}</p>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {steps.length === 0 && (
        <p className="text-sm text-gray-400 text-center py-8">
          Pipeline will appear here once a case is submitted
        </p>
      )}
    </div>
  );
}
