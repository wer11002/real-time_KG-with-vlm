import { useState, useEffect, useRef, useCallback } from 'react'

const toLabel = s => (s || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

export function useKgSocket() {
  const [graphData, setGraphData]     = useState({ nodes: [], links: [] })
  const [stats, setStats]             = useState({ events: 0, players: 0 })
  const [connected, setConnected]     = useState(false)
  const [lastUpdate, setLastUpdate]   = useState(null)
  const [selectedNode, setSelectedNode] = useState(null)
  const [recentNodeIds, setRecentNodeIds] = useState([])

  // Mutable state — not React state (no re-renders needed from these)
  const s = useRef({
    nodeMap:          new Map(),  // id → node object
    linkKeys:         new Set(),  // dedup links
    lastEventByMatch: new Map(),  // match_id → prev event_id
    teamSideMap:      new Map(),  // team_id → 'home'|'away'
    eventCount:  0,
    playerCount: 0,
  })

  const processEvent = useCallback((evt) => {
    const st = s.current
    const newNodes = []
    const newLinks = []

    const addNode = (id, data) => {
      if (st.nodeMap.has(id)) return false
      const node = { id, ...data }
      st.nodeMap.set(id, node)
      newNodes.push(node)
      return true
    }

    const addLink = (source, target, label) => {
      const key = `${source}|${target}|${label}`
      if (st.linkKeys.has(key)) return
      st.linkKeys.add(key)
      newLinks.push({ source, target, label })
    }

    // Match node
    const matchNodeId = `match::${evt.match_id}`
    addNode(matchNodeId, {
      nodeType: 'match',
      label: toLabel(evt.match_id),
      rawData: { match_id: evt.match_id },
    })

    // Team node — track home/away from VLM team_side field
    if (evt.team_id) {
      if (evt.team_side && !st.teamSideMap.has(evt.team_id)) {
        st.teamSideMap.set(evt.team_id, evt.team_side)
      }
      const side = st.teamSideMap.get(evt.team_id) || 'home'
      const teamNodeId = `team::${evt.team_id}`
      addNode(teamNodeId, {
        nodeType: 'team',
        side,
        label: toLabel(evt.team_id),
        rawData: { team_id: evt.team_id, side },
      })
    }

    // Player node
    if (evt.player_id) {
      const side = st.teamSideMap.get(evt.team_id) || 'home'
      const playerNodeId = `player::${evt.player_id}`
      const isNew = addNode(playerNodeId, {
        nodeType: 'player',
        side,
        label: toLabel(evt.player_id),
        rawData: { player_id: evt.player_id, team_id: evt.team_id },
      })
      if (isNew) st.playerCount++
    }

    // Event node
    const eventNodeId = `event::${evt.event_id}`
    addNode(eventNodeId, {
      nodeType: evt.event_type || 'event',
      label: `${evt.event_type || 'event'} ${evt.time_raw || ''}`.trim(),
      rawData: evt,
    })
    st.eventCount++

    // Links
    if (evt.player_id) addLink(`player::${evt.player_id}`, eventNodeId, 'PERFORMED')
    if (evt.team_id)   addLink(`team::${evt.team_id}`,   eventNodeId, 'INVOLVED_IN')
    addLink(eventNodeId, matchNodeId, 'IN_MATCH')

    const prevId = st.lastEventByMatch.get(evt.match_id)
    if (prevId) addLink(eventNodeId, `event::${prevId}`, 'PRECEDED_BY')
    st.lastEventByMatch.set(evt.match_id, evt.event_id)

    // Commit to React state
    if (newNodes.length > 0 || newLinks.length > 0) {
      setGraphData(prev => ({
        nodes: newNodes.length ? [...prev.nodes, ...newNodes] : prev.nodes,
        links: newLinks.length ? [...prev.links, ...newLinks] : prev.links,
      }))
      if (newNodes.length > 0) {
        setRecentNodeIds(newNodes.map(n => n.id))
      }
    }

    setStats({ events: st.eventCount, players: st.playerCount })
    setLastUpdate(new Date().toLocaleTimeString())
  }, [])

  // WebSocket with auto-reconnect
  useEffect(() => {
    let ws
    let reconnectTimer
    let mounted = true

    const connect = () => {
      if (!mounted) return
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
      ws = new WebSocket(`${proto}//${location.host}/ws`)

      ws.onopen  = () => { if (mounted) setConnected(true) }
      ws.onclose = () => {
        if (!mounted) return
        setConnected(false)
        reconnectTimer = setTimeout(connect, 3000)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (e) => {
        if (!mounted) return
        try { processEvent(JSON.parse(e.data)) } catch {}
      }
    }

    connect()
    return () => {
      mounted = false
      clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [processEvent])

  return { graphData, stats, connected, lastUpdate, selectedNode, setSelectedNode, recentNodeIds }
}
