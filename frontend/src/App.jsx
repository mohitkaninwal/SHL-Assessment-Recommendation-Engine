import React, { useEffect, useMemo, useState } from 'react'

const defaultApiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

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
            <p>Paste a job description or pick a prompt to get relevant SHL test recommendations.</p>
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
                placeholder="Enter hiring requirement or job description..."
                maxLength={1000}
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
                <span>{query.length}/1000</span>
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
              <h3>Recommendations ({recommendationInfo.returned})</h3>
              <p className="status-line">
                Total retrieved from backend: <strong>{recommendationInfo.total_found ?? 0}</strong>
              </p>
              {Number(recommendationInfo.returned || 0) === 0 && (
                <p className="error-box">
                  No assessments were returned for this query. Try another query or disable balanced skill filtering in Developer Controls.
                </p>
              )}
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Assessment</th>
                    <th>Type</th>
                    <th>Score</th>
                    <th>URL</th>
                  </tr>
                </thead>
                <tbody>
                  {recommendationInfo.recommendations.map((item, index) => (
                    <tr key={`${item.assessment_url}-${index}`}>
                      <td>{index + 1}</td>
                      <td>{item.assessment_name}</td>
                      <td>{item.test_type}</td>
                      <td>{Number(item.similarity_score).toFixed(3)}</td>
                      <td><a href={item.assessment_url} target="_blank" rel="noreferrer">open</a></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </div>
      </section>
    </main>
  )
}
