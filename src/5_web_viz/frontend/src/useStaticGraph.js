import { useState, useEffect } from 'react'

export function useStaticGraph() {
  const [graphData, setGraphData]       = useState({ nodes: [], links: [] })
  const [stats, setStats]               = useState({ events: 0, players: 0 })
  const [loading, setLoading]           = useState(true)
  const [error, setError]               = useState(null)
  const [selectedNode, setSelectedNode] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch('/api/graph')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(data => {
        const all = data.nodes || []
        setGraphData({ nodes: all, links: data.links || [] })
        setStats({
          events:  all.filter(n => !['match', 'team', 'player'].includes(n.nodeType)).length,
          players: all.filter(n => n.nodeType === 'player').length,
        })
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  return { graphData, stats, loading, error, selectedNode, setSelectedNode }
}
