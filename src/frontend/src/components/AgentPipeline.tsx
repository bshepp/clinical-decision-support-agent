"use client";

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

export function AgentPipeline({ steps, isRunning }: AgentPipelineProps) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">
        Agent Pipeline
      </h3>

      <div className="space-y-1">
        {steps.map((step, index) => {
          const config = STATUS_CONFIG[step.status];
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
                <span
                  className={`text-lg font-bold ${config.color} ${
                    step.status === "running" ? "animate-pulse-dot" : ""
                  }`}
                >
                  {config.icon}
                </span>

                {/* Step info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-800">
                      {step.step_name}
                    </span>
                    {step.duration_ms != null && (
                      <span className="text-xs text-gray-400">
                        {step.duration_ms}ms
                      </span>
                    )}
                  </div>

                  {step.tool_name && (
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
