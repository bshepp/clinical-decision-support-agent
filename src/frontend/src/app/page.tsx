"use client";

import { useState, useCallback } from "react";
import { PatientInput } from "@/components/PatientInput";
import { AgentPipeline } from "@/components/AgentPipeline";
import { CDSReport } from "@/components/CDSReport";
import { FeedbackWidget } from "@/components/FeedbackWidget";
import { useAgentWebSocket } from "@/hooks/useAgentWebSocket";
import { reportToMarkdown } from "@/lib/reportToMarkdown";

export default function Home() {
  const { steps, report, isRunning, isWarmingUp, warmUpMessage, error, submitCase, reset } = useAgentWebSocket();
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [lastPatientText, setLastPatientText] = useState("");

  const handleSubmit = (patientText: string) => {
    setHasSubmitted(true);
    setLastPatientText(patientText);
    submitCase({
      patient_text: patientText,
      include_drug_check: true,
      include_guidelines: true,
    });
  };

  const handleNewCase = useCallback(() => {
    reset();
    setHasSubmitted(false);
    setLastPatientText("");
  }, [reset]);

  const handleRetry = useCallback(() => {
    if (lastPatientText) {
      handleSubmit(lastPatientText);
    }
  }, [lastPatientText]);

  const handleDownload = useCallback(() => {
    if (!report) return;
    const md = reportToMarkdown(report);
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cds-report-${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [report]);

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
            {/* Backend availability notice */}
            <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg text-center">
              <p className="text-sm text-blue-800">
                The MedGemma backend is currently paused.{" "}
                <strong>Contact me to spin it up for a live demo.</strong>
              </p>
              <div className="mt-2 flex items-center justify-center gap-4 text-sm">
                <a
                  href="https://github.com/bshepp/clinical-decision-support-agent"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-blue-700 hover:text-blue-900 font-medium underline"
                >
                  GitHub Repo
                </a>
                <span className="text-blue-300">|</span>
                <a
                  href="https://huggingface.co/bshepp"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-blue-700 hover:text-blue-900 font-medium underline"
                >
                  Hugging Face Profile
                </a>
              </div>
            </div>

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

              {/* Error display with retry/reset */}
              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-700 text-sm mb-3">{error}</p>
                  <div className="flex gap-2">
                    <button
                      onClick={handleRetry}
                      className="px-3 py-1.5 text-xs font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                    >
                      Try Again
                    </button>
                    <button
                      onClick={handleNewCase}
                      className="px-3 py-1.5 text-xs font-medium text-gray-700 bg-white border border-gray-300 hover:bg-gray-50 rounded-lg transition-colors"
                    >
                      New Case
                    </button>
                  </div>
                </div>
              )}

              {/* New Case button when pipeline finished (not running, no error) */}
              {!isRunning && !error && steps.length > 0 && (
                <div className="mt-4 flex gap-2">
                  <button
                    onClick={handleNewCase}
                    className="flex-1 px-4 py-2 text-sm font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-lg transition-colors"
                  >
                    New Case
                  </button>
                  {report && (
                    <button
                      onClick={handleDownload}
                      className="px-4 py-2 text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 border border-gray-300 rounded-lg transition-colors"
                      title="Download report as Markdown"
                    >
                      Download .md
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* CDS Report (right) */}
            <div className="lg:col-span-2">
              {report ? (
                <CDSReport report={report} />
              ) : isWarmingUp ? (
                <div className="flex items-center justify-center h-64">
                  <div className="text-center max-w-lg">
                    <div className="relative mx-auto mb-5 w-14 h-14">
                      <div className="absolute inset-0 rounded-full border-4 border-amber-200" />
                      <div className="absolute inset-0 rounded-full border-4 border-amber-500 border-t-transparent animate-spin" />
                      <div className="absolute inset-0 flex items-center justify-center text-xl text-amber-600">
                        &#9881;
                      </div>
                    </div>
                    <p className="font-semibold text-amber-700 text-lg">
                      MedGemma 27B is Loading
                    </p>
                    <p className="text-sm text-amber-600 mt-1">
                      {warmUpMessage || "Waiting for MedGemma endpoint..."}
                    </p>
                    <p className="text-sm text-gray-500 mt-3 leading-relaxed">
                      This is a 27-billion parameter medical AI model running on
                      dedicated GPUs. It scales to zero when inactive to save costs,
                      so it needs <strong>3-5 minutes</strong> to load on first visit.
                    </p>
                    <p className="text-sm text-gray-500 mt-2 leading-relaxed">
                      Please wait — the analysis will start automatically once the model is ready.
                    </p>
                    <p className="text-xs text-gray-400 mt-4 italic">
                      While you wait: this pipeline will parse the patient case, generate
                      a differential diagnosis, check drug interactions against real FDA
                      databases, retrieve clinical guidelines, and detect care gaps.
                    </p>
                  </div>
                </div>
              ) : isRunning ? (
                <div className="flex items-center justify-center h-64 text-gray-400">
                  <div className="text-center">
                    <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
                    <p className="text-gray-600 font-medium">Agent pipeline running...</p>
                    <p className="text-sm text-gray-400 mt-1">
                      Watch the steps on the left — the report will appear here when done.
                    </p>
                    <p className="text-xs text-gray-400 mt-3">
                      Full analysis typically takes 2–4 minutes across 6 steps.
                    </p>
                  </div>
                </div>
              ) : error && steps.length === 0 ? (
                /* Full-screen error when nothing has even started (e.g. WS connection failed) */
                <div className="flex items-center justify-center h-64">
                  <div className="text-center max-w-sm">
                    <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center mx-auto mb-4">
                      <span className="text-red-500 text-xl">!</span>
                    </div>
                    <p className="font-medium text-gray-800 mb-1">Connection Failed</p>
                    <p className="text-sm text-gray-500 mb-4">{error}</p>
                    <div className="flex gap-2 justify-center">
                      <button
                        onClick={handleRetry}
                        className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                      >
                        Try Again
                      </button>
                      <button
                        onClick={handleNewCase}
                        className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 hover:bg-gray-50 rounded-lg transition-colors"
                      >
                        New Case
                      </button>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}
      </div>

      {/* Feedback widget */}
      <FeedbackWidget />

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
