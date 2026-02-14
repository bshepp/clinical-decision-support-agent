"use client";

import { useState } from "react";
import { PatientInput } from "@/components/PatientInput";
import { AgentPipeline } from "@/components/AgentPipeline";
import { CDSReport } from "@/components/CDSReport";
import { useAgentWebSocket } from "@/hooks/useAgentWebSocket";

export default function Home() {
  const { steps, report, isRunning, error, submitCase } = useAgentWebSocket();
  const [hasSubmitted, setHasSubmitted] = useState(false);

  const handleSubmit = (patientText: string) => {
    setHasSubmitted(true);
    submitCase({
      patient_text: patientText,
      include_drug_check: true,
      include_guidelines: true,
    });
  };

  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              🏥 CDS Agent
            </h1>
            <p className="text-sm text-gray-500">
              Clinical Decision Support powered by MedGemma
            </p>
          </div>
          <span className="text-xs bg-blue-100 text-blue-700 px-3 py-1 rounded-full font-medium">
            HAI-DEF · Agentic Workflow
          </span>
        </div>
      </header>

      {/* Main content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {!hasSubmitted ? (
          /* Input view */
          <div className="max-w-3xl mx-auto">
            <div className="mb-8 text-center">
              <h2 className="text-xl font-semibold text-gray-800 mb-2">
                Submit a Patient Case
              </h2>
              <p className="text-gray-500">
                Enter a patient case description. The agent pipeline will
                parse, reason, check interactions, retrieve guidelines, and
                synthesize a clinical decision support report.
              </p>
            </div>
            <PatientInput onSubmit={handleSubmit} isLoading={isRunning} />
          </div>
        ) : (
          /* Pipeline + Results view */
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Agent Pipeline (left) */}
            <div className="lg:col-span-1">
              <AgentPipeline steps={steps} isRunning={isRunning} />
              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                  {error}
                </div>
              )}
            </div>

            {/* CDS Report (right) */}
            <div className="lg:col-span-2">
              {report ? (
                <CDSReport report={report} />
              ) : isRunning ? (
                <div className="flex items-center justify-center h-64 text-gray-400">
                  <div className="text-center">
                    <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
                    <p>Agent pipeline running...</p>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}
      </div>

      {/* Disclaimer footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-amber-50 border-t border-amber-200 px-6 py-2">
        <p className="text-center text-xs text-amber-700">
          ⚠️ AI-generated clinical decision support — for demonstration purposes
          only. Does not replace professional medical judgment.
        </p>
      </footer>
    </main>
  );
}
