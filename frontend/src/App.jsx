import React, { useEffect, useMemo, useState } from 'react'

const defaultApiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const MAX_QUERY_LENGTH = 5000

const quickPrompts = [
  'Recommend assessments for Java developers with collaboration skills.',
  'Suggest tests for sales graduates with communication and customer focus.',
  'Find assessments for a Python + SQL data analyst role.',
  'Balance hard and soft skill tests for a project manager JD.'
]

async function requestJson(url, options = {}) {
  const { timeoutMs = 15000, ...fetchOptions } = options
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs)
  let response

  try {
    response = await fetch(url, { ...fetchOptions, signal: controller.signal })
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw new Error('Request timed out. Backend might be down or waking up.')
    }
    throw error
  } finally {
    window.clearTimeout(timeoutId)
  }

  let payload = null

  try {
    payload = await response.json()
  } catch (error) {
    payload = { error: 'Non-JSON response', detail: String(error) }
  }

  if (!response.ok) {
    const base = payload?.error || payload?.detail || `HTTP ${response.status}`
    const message = `${base} (status ${response.status})`
    throw new Error(message)
  }

  return payload
}

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(defaultApiBase)
  const [query, setQuery] = useState(quickPrompts[0])
  const [topK, setTopK] = useState(10)
  const [balanceSkills, setBalanceSkills] = useState(true)
  const [includeExplanation, setIncludeExplanation] = useState(true)
  const [showDeveloper, setShowDeveloper] = useState(false)

  const [rootInfo, setRootInfo] = useState(null)
  const [healthInfo, setHealthInfo] = useState(null)
  const [recommendationInfo, setRecommendationInfo] = useState(null)
  const [error, setError] = useState('')
  const [healthLoading, setHealthLoading] = useState(false)
  const [recommendLoading, setRecommendLoading] = useState(false)

  const normalizedApiBase = useMemo(() => apiBaseUrl.replace(/\/$/, ''), [apiBaseUrl])

  const fetchRoot = async () => {
    setError('')
    setHealthLoading(true)
    try {
      const data = await requestJson(`${normalizedApiBase}/`)
      setRootInfo(data)
    } catch (eventError) {
      setError(`Root endpoint failed: ${eventError.message}`)
    } finally {
      setHealthLoading(false)
    }
  }

  const fetchHealth = async () => {
    setError('')
    setHealthLoading(true)
    try {
      const data = await requestJson(`${normalizedApiBase}/health`)
      setHealthInfo(data)
    } catch (eventError) {
      setError(`Health endpoint failed: ${eventError.message}`)
    } finally {
      setHealthLoading(false)
    }
  }

  useEffect(() => {
    fetchHealth()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!error) {
      return undefined
    }

    const timeoutId = window.setTimeout(() => {
      setError('')
    }, 4500)

    return () => window.clearTimeout(timeoutId)
  }, [error])

  const fetchRecommendations = async (event) => {
    event.preventDefault()
    setError('')
    setRecommendLoading(true)
    setRecommendationInfo(null)

    try {
      const data = await requestJson(`${normalizedApiBase}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: Number(topK),
          balance_skills: balanceSkills,
          include_explanation: includeExplanation
        })
      })
      setRecommendationInfo(data)
    } catch (eventError) {
      setError(`Recommend endpoint failed: ${eventError.message}`)
    } finally {
      setRecommendLoading(false)
    }
  }

  const healthStatus = healthInfo?.status === 'healthy' && healthInfo?.engine_status === 'healthy'
  const recommendedAssessments = recommendationInfo?.recommended_assessments
    || recommendationInfo?.recommendations
    || []

  return (
    <main className="screen">
      {error && (
        <div className="toast toast-error" role="alert" aria-live="assertive">
          <span>{error}</span>
          <button type="button" aria-label="Dismiss error" onClick={() => setError('')}>
            ×
          </button>
        </div>
      )}

      <section className="main">
        <div className="center-wrap">
          <header className="hero">
            <h1>
              SHL Assessment
            </h1>
            <h2>Recommendation Assistant</h2>
            <p>Enter a natural language query, paste JD text, or provide a JD URL to get relevant SHL test recommendations.</p>
          </header>

          <section className="prompt-grid">
            {quickPrompts.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="prompt-card"
                onClick={() => setQuery(prompt)}
                disabled={recommendLoading}
              >
                <span>{prompt}</span>
                <i>◌</i>
              </button>
            ))}
          </section>

            <button type="button" className="refresh-btn" onClick={() => setQuery(quickPrompts[0])} disabled={recommendLoading}>
            ↻ Reset Prompt
          </button>

          <form className="composer" onSubmit={fetchRecommendations}>
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Enter a query, paste JD text, or paste a JD URL..."
                maxLength={MAX_QUERY_LENGTH}
                required
              />

            <div className="composer-mid">
              <button type="button" className="chip" disabled>◎ SHL Catalog ▾</button>
            </div>

            <div className="composer-bottom">
              <div className="left-actions">
                <button type="button" className="ghost" onClick={() => setQuery('')} disabled={recommendLoading}>⊕ Clear</button>
                <button type="button" className="ghost" onClick={() => setQuery(quickPrompts[Math.floor(Math.random() * quickPrompts.length)])} disabled={recommendLoading}>↺ Random Prompt</button>
              </div>
              <div className="right-actions">
                <span>{query.length}/{MAX_QUERY_LENGTH}</span>
                <button type="submit" className="send" disabled={recommendLoading}>{recommendLoading ? '…' : '➜'}</button>
              </div>
            </div>
          </form>

          <button
            type="button"
            className="dev-toggle"
            onClick={() => setShowDeveloper((value) => !value)}
          >
            Developer Controls {showDeveloper ? '▴' : '▾'}
          </button>

          {showDeveloper && (
            <section className="dev-panel">
              <div className="dev-grid">
                <label>
                  API Base URL
                  <input
                    type="text"
                    value={apiBaseUrl}
                    onChange={(event) => setApiBaseUrl(event.target.value)}
                  />
                </label>

                <label>
                  Top K
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={topK}
                    onChange={(event) => setTopK(event.target.value)}
                  />
                </label>

                <label className="inline">
                  <input
                    type="checkbox"
                    checked={balanceSkills}
                    onChange={(event) => setBalanceSkills(event.target.checked)}
                  />
                  Balance skills
                </label>

                <label className="inline">
                  <input
                    type="checkbox"
                    checked={includeExplanation}
                    onChange={(event) => setIncludeExplanation(event.target.checked)}
                  />
                  Include explanation
                </label>

                <button type="button" onClick={fetchRoot} disabled={healthLoading || recommendLoading}>Test /</button>
                <button type="button" onClick={fetchHealth} disabled={healthLoading || recommendLoading}>Test /health</button>
              </div>

              <p className="status-line">
                Health: <strong>{healthInfo ? `${healthInfo.status}/${healthInfo.engine_status}` : 'not checked'}</strong>
                {' '}| Root: <strong>{rootInfo ? 'loaded' : 'not checked'}</strong>
                {' '}| Model Flow: <strong>{healthStatus ? 'ready' : 'check required'}</strong>
              </p>

            </section>
          )}

          {recommendationInfo && (
            <section className="results">
              <h3>Recommendations ({recommendedAssessments.length})</h3>
              {Number(recommendedAssessments.length || 0) === 0 && (
                <p className="error-box">
                  No assessments were returned for this query. Try another query or disable balanced skill filtering in Developer Controls.
                </p>
              )}
              <div className="results-table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Assessment Name</th>
                      <th>URL</th>
                      <th>Test Type</th>
                      <th>Duration (mins)</th>
                      <th>Remote Support</th>
                      <th>Adaptive Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recommendedAssessments.map((item, index) => {
                      const assessmentUrl = item.url || item.assessment_url
                      const assessmentName = item.name || item.assessment_name
                      const testType = Array.isArray(item.test_type) ? item.test_type.join(', ') : item.test_type

                      return (
                        <tr key={`${assessmentUrl}-${index}`}>
                          <td>{index + 1}</td>
                          <td>{assessmentName || 'NA'}</td>
                          <td>
                            {assessmentUrl ? (
                              <a href={assessmentUrl} target="_blank" rel="noreferrer">
                                {assessmentUrl}
                              </a>
                            ) : (
                              'NA'
                            )}
                          </td>
                          <td>{testType || 'NA'}</td>
                          <td>{item.duration ?? 'NA'}</td>
                          <td>{item.remote_support || 'NA'}</td>
                          <td>{item.adaptive_support || 'NA'}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </div>
      </section>
    </main>
  )
}
