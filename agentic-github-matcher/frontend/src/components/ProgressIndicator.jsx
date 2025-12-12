function ProgressIndicator({ progress, message, strategy }) {
  // Define process steps
  const steps = [
    { name: "Initializing", threshold: 0, icon: "⚙️" },
    { name: "Validating Input", threshold: 5, icon: "🔒" },
    { name: "Analyzing Job Description", threshold: 10, icon: "📋" },
    { name: "Searching GitHub", threshold: 35, icon: "🔍" },
    { name: "Fetching Profiles", threshold: 45, icon: "👤" },
    { name: "Analyzing Skills", threshold: 55, icon: "🔬" },
    { name: "Scoring Candidates", threshold: 70, icon: "🤖" },
    { name: "Formatting Results", threshold: 85, icon: "📊" },
    { name: "Complete", threshold: 100, icon: "✅" },
  ];

  // Find current step
  const currentStepIndex = steps.findIndex(
    (step, index) =>
      progress >= step.threshold &&
      (index === steps.length - 1 || progress < steps[index + 1].threshold)
  );
  const currentStep =
    currentStepIndex >= 0 ? steps[currentStepIndex] : steps[0];

  return (
    <div className="space-y-6">
      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="bg-gray-200 rounded-full h-4 overflow-hidden">
          <div
            className="bg-gradient-to-r from-blue-500 to-blue-600 h-full transition-all duration-500 ease-out shadow-sm"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500">
          <span>0%</span>
          <span className="font-medium text-blue-600">
            {Math.round(progress)}%
          </span>
          <span>100%</span>
        </div>
      </div>

      {/* Current Status */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{currentStep.icon}</span>
          <div className="flex-1">
            <p className="text-sm font-semibold text-blue-900">
              {message || currentStep.name}
            </p>
            <div className="flex items-center gap-2 mt-1">
              <p className="text-xs text-blue-700">{currentStep.name}</p>
              {strategy && (
                <>
                  <span className="text-blue-400">•</span>
                  <p className="text-xs text-blue-600">
                    <span className="font-medium">Strategy:</span>{" "}
                    <span
                      className={`font-semibold ${
                        strategy === "MCP"
                          ? "text-green-600"
                          : "text-orange-600"
                      }`}
                    >
                      {strategy}
                    </span>
                  </p>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Step Indicators */}
      <div className="space-y-2">
        <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">
          Process Steps
        </p>
        <div className="space-y-2">
          {steps.slice(0, -1).map((step, index) => {
            const isActive = progress >= step.threshold;
            const isCurrent = currentStepIndex === index;

            return (
              <div
                key={index}
                className={`flex items-center gap-3 text-sm transition-all ${
                  isActive
                    ? isCurrent
                      ? "text-blue-700 font-semibold"
                      : "text-green-600"
                    : "text-gray-400"
                }`}
              >
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${
                    isActive
                      ? isCurrent
                        ? "bg-blue-600 text-white"
                        : "bg-green-500 text-white"
                      : "bg-gray-200 text-gray-400"
                  }`}
                >
                  {isActive ? (isCurrent ? "⟳" : "✓") : index + 1}
                </div>
                <span>
                  {step.icon} {step.name}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default ProgressIndicator;
