function ResultsDisplay({ results, strategy }) {
  const { analysis, results: searchResults } = results || {};
  const developers = searchResults?.developers || [];
  const skills = analysis?.skills || [];

  return (
    <div className="space-y-6">
      {/* Search Strategy Info */}
      {strategy && (
        <div
          className={`border rounded-lg p-3 ${
            strategy === "MCP"
              ? "bg-green-50 border-green-200"
              : "bg-orange-50 border-orange-200"
          }`}
        >
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-700">
              Search Method:
            </span>
            <span
              className={`text-sm font-semibold ${
                strategy === "MCP" ? "text-green-700" : "text-orange-700"
              }`}
            >
              {strategy}
            </span>
            {strategy === "MCP" && (
              <span className="text-xs text-green-600 ml-2">
                (Model Context Protocol)
              </span>
            )}
            {strategy === "Direct REST API" && (
              <span className="text-xs text-orange-600 ml-2">
                (Fallback mode)
              </span>
            )}
          </div>
        </div>
      )}

      {/* Job Analysis Summary */}
      {analysis && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-blue-900 mb-2">
            Job Analysis Summary
          </h3>
          <div className="space-y-2 text-sm">
            <p>
              <span className="font-medium">Title:</span>{" "}
              {analysis.title || "N/A"}
            </p>
            <p>
              <span className="font-medium">Experience Level:</span>{" "}
              {analysis.experience_level || "N/A"}
            </p>
            {skills.length > 0 && (
              <div>
                <span className="font-medium">Skills:</span>{" "}
                <span className="text-blue-700">
                  {skills.slice(0, 10).join(", ")}
                  {skills.length > 10 && ` +${skills.length - 10} more`}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Candidates Table */}
      {developers.length > 0 ? (
        <div>
          <h3 className="text-lg font-semibold text-gray-800 mb-3">
            Matching Candidates ({developers.length})
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Rank
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Developer
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Skills Match
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Followers
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Repos
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Score
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {developers.map((dev, idx) => (
                  <tr key={dev.username} className="hover:bg-gray-50">
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                      #{idx + 1}
                      {dev.is_exact_match && (
                        <span className="ml-2 px-2 py-0.5 text-xs bg-green-100 text-green-800 rounded">
                          Exact
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div>
                        <a
                          href={dev.html_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm font-medium text-blue-600 hover:text-blue-800"
                        >
                          @{dev.username}
                        </a>
                        {dev.name && (
                          <p className="text-xs text-gray-500">{dev.name}</p>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="text-sm">
                        <div className="text-gray-900">
                          {dev.matching_skills?.slice(0, 3).join(", ") || "N/A"}
                          {dev.matching_skills?.length > 3 && (
                            <span className="text-gray-500">
                              {" "}
                              +{dev.matching_skills.length - 3}
                            </span>
                          )}
                        </div>
                        {dev.skill_match_percentage !== undefined && (
                          <div className="text-xs text-gray-500 mt-1">
                            {dev.skill_match_percentage.toFixed(0)}% match
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                      {dev.followers?.toLocaleString() || 0}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                      {dev.public_repos || 0}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span
                        className={`text-sm font-medium ${
                          dev.relevance_score >= 50
                            ? "text-green-600"
                            : dev.relevance_score >= 30
                            ? "text-yellow-600"
                            : "text-red-600"
                        }`}
                      >
                        {dev.relevance_score?.toFixed(0) || 0}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="text-center text-gray-500 py-8">
          <p>No matching candidates found.</p>
        </div>
      )}

      {/* Repositories Summary */}
      {searchResults?.repositories && searchResults.repositories.length > 0 && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">
            Related Repositories
          </h3>
          <p className="text-sm text-gray-600">
            Found {searchResults.total_repos_found?.toLocaleString() || 0}{" "}
            repositories
          </p>
        </div>
      )}
    </div>
  );
}

export default ResultsDisplay;
