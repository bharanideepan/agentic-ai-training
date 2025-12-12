import { useState } from "react";

function JobDescriptionForm({ onAnalyze, onReset, disabled }) {
  const [jobDescription, setJobDescription] = useState("");
  const [maxCandidates, setMaxCandidates] = useState(10);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (jobDescription.trim()) {
      onAnalyze(jobDescription, maxCandidates);
    }
  };

  const handleReset = () => {
    setJobDescription("");
    setMaxCandidates(10);
    onReset();
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label
          htmlFor="job-description"
          className="block text-sm font-medium text-gray-700 mb-2"
        >
          Job Description
        </label>
        <textarea
          id="job-description"
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
          disabled={disabled}
          rows={12}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          placeholder="Paste the job description here...&#10;&#10;Example:&#10;We are looking for a Senior Python Developer with 5+ years of experience in Django, PostgreSQL, and React..."
        />
      </div>

      <div>
        <label
          htmlFor="max-candidates"
          className="block text-sm font-medium text-gray-700 mb-2"
        >
          Max Candidates
        </label>
        <input
          id="max-candidates"
          type="number"
          min="1"
          max="50"
          value={maxCandidates}
          onChange={(e) => setMaxCandidates(parseInt(e.target.value) || 10)}
          disabled={disabled}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
        />
      </div>

      <div className="flex gap-3">
        <button
          type="submit"
          disabled={disabled || !jobDescription.trim()}
          className="flex-1 bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {disabled ? "Analyzing..." : "Find Candidates"}
        </button>

        <button
          type="button"
          onClick={handleReset}
          disabled={disabled}
          className="px-6 py-3 border border-gray-300 rounded-lg font-semibold hover:bg-gray-50 disabled:bg-gray-100 disabled:cursor-not-allowed transition-colors"
        >
          Reset
        </button>
      </div>
    </form>
  );
}

export default JobDescriptionForm;
