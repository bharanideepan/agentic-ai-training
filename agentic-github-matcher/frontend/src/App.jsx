import { useState } from "react";
import JobDescriptionForm from "./components/JobDescriptionForm";
import ResultsDisplay from "./components/ResultsDisplay";
import ProgressIndicator from "./components/ProgressIndicator";

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [searchStrategy, setSearchStrategy] = useState(null);

  const handleAnalyze = async (jobDescription, maxCandidates) => {
    setIsLoading(true);
    setProgress(0);
    setStatusMessage("");
    setResults(null);
    setError(null);
    setSearchStrategy(null);

    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          job_description: jobDescription,
          max_candidates: maxCandidates || 10,
          output_format: "json",
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const chunk = JSON.parse(line);

            switch (chunk.type) {
              case "status":
                setStatusMessage(chunk.message || "");
                // Update strategy if provided
                if (chunk.data && chunk.data.strategy) {
                  setSearchStrategy(chunk.data.strategy);
                }
                break;

              case "progress":
                setProgress(chunk.progress || 0);
                setStatusMessage(chunk.message || "");
                if (chunk.data) {
                  // Update strategy if provided
                  if (chunk.data.strategy) {
                    setSearchStrategy(chunk.data.strategy);
                  }
                  // Update partial results if available
                  if (chunk.data.skills) {
                    setResults((prev) => ({
                      ...prev,
                      skills: chunk.data.skills,
                      analysis: chunk.data.analysis,
                    }));
                  }
                }
                break;

              case "result":
                setProgress(100);
                setStatusMessage(chunk.message || "Analysis complete!");
                if (chunk.data) {
                  // Update strategy if provided
                  if (chunk.data.strategy) {
                    setSearchStrategy(chunk.data.strategy);
                  }
                  setResults({
                    analysis: chunk.data.analysis,
                    results: chunk.data.results,
                  });
                }
                break;

              case "error":
                setError(chunk.message || "An error occurred");
                setIsLoading(false);
                return;

              default:
                console.log("Unknown chunk type:", chunk.type);
            }
          } catch (e) {
            console.error("Error parsing chunk:", e, line);
          }
        }
      }

      setIsLoading(false);
    } catch (err) {
      setError(err.message || "Failed to analyze job description");
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setIsLoading(false);
    setProgress(0);
    setStatusMessage("");
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🚀 Agentic GitHub Matcher
          </h1>
          <p className="text-gray-600">
            AI-powered job description analyzer and GitHub candidate finder
          </p>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Form */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Job Description
            </h2>
            <JobDescriptionForm
              onAnalyze={handleAnalyze}
              onReset={handleReset}
              disabled={isLoading}
            />
          </div>

          {/* Right Column - Progress & Results */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Results
            </h2>

            {isLoading && (
              <div className="space-y-4">
                <ProgressIndicator
                  progress={progress}
                  message={statusMessage}
                  strategy={searchStrategy}
                />
                {statusMessage && (
                  <div className="bg-blue-50 border-l-4 border-blue-500 p-3 rounded">
                    <p className="text-sm text-blue-800 font-medium">
                      {statusMessage}
                    </p>
                    {searchStrategy && (
                      <p className="text-xs text-blue-600 mt-1">
                        Using:{" "}
                        <span className="font-semibold">{searchStrategy}</span>
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                <p className="text-red-800 font-semibold">Error</p>
                <p className="text-red-600">{error}</p>
              </div>
            )}

            {results && !isLoading && (
              <ResultsDisplay results={results} strategy={searchStrategy} />
            )}

            {!isLoading && !results && !error && (
              <div className="text-center text-gray-500 py-12">
                <p>
                  Enter a job description and click "Analyze" to get started
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
