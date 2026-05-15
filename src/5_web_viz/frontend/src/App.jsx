import { useRef, useEffect, useCallback } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { useKgSocket } from './useKgSocket.js'
import { getNodeColor, getNodeSize, LEGEND_ITEMS } from './nodeColors.js'

const HEADER_H = 44
const LEGEND_H = 36

export default function App() {
  const glowingRef = useRef(new Set())

  const {
    graphData, stats, connected, lastUpdate,
    selectedNode, setSelectedNode, recentNodeIds,
  } = useKgSocket()

  // Track which nodes should glow (new nodes pulse for 2 seconds)
  useEffect(() => {
    if (!recentNodeIds.length) return
    recentNodeIds.forEach(id => glowingRef.current.add(id))
    const ids = [...recentNodeIds]
    const t = setTimeout(() => ids.forEach(id => glowingRef.current.delete(id)), 2000)
    return () => clearTimeout(t)
  }, [recentNodeIds])

  const drawNode = useCallback((node, ctx, globalScale) => {
    const size  = getNodeSize(node)
    const color = getNodeColor(node)
    const glow  = glowingRef.current.has(node.id)

    // Glow ring for newly added nodes
    if (glow) {
      ctx.save()
      ctx.beginPath()
      ctx.arc(node.x, node.y, size + 5, 0, 2 * Math.PI)
      ctx.fillStyle   = 'rgba(255,255,255,0.15)'
      ctx.shadowBlur  = 20
      ctx.shadowColor = 'rgba(255,255,255,0.9)'
      ctx.fill()
      ctx.shadowBlur  = 0
      ctx.restore()
    }

    // Node circle
    ctx.beginPath()
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI)
    ctx.fillStyle   = color
    ctx.fill()
    ctx.strokeStyle = 'rgba(255,255,255,0.2)'
    ctx.lineWidth   = 0.5
    ctx.stroke()

    // Always-visible label for match and team nodes
    if (node.nodeType === 'match' || node.nodeType === 'team') {
      const fontSize = Math.max(9, 11 / globalScale)
      ctx.font          = `${fontSize}px monospace`
      ctx.textAlign     = 'center'
      ctx.textBaseline  = 'top'
      ctx.fillStyle     = 'rgba(255,255,255,0.9)'
      ctx.fillText(node.label, node.x, node.y + size + 2)
    }
  }, [])

  const nodeTooltip = useCallback(node =>
    (node.nodeType !== 'match' && node.nodeType !== 'team') ? node.label : ''
  , [])

  const handleNodeClick = useCallback(node =>
    setSelectedNode(prev => prev?.id === node.id ? null : node)
  , [setSelectedNode])

  return (
    <div style={styles.root}>

      {/* Stats bar */}
      <div style={styles.header}>
        <span style={styles.title}>Soccer EKG</span>
        <span>Events: <b>{stats.events}</b></span>
        <span>Players: <b>{stats.players}</b></span>
        {lastUpdate && <span style={styles.dim}>Last update: {lastUpdate}</span>}
        <span style={styles.liveGroup}>
          <span style={{ ...styles.dot, background: connected ? '#40c057' : '#e03131' }} />
          <span style={{ color: connected ? '#40c057' : '#e03131' }}>
            {connected ? 'LIVE' : 'DISCONNECTED'}
          </span>
        </span>
      </div>

      {/* Graph */}
      <div style={styles.graphWrap}>
        <ForceGraph2D
          graphData={graphData}
          backgroundColor="#1a1a2e"
          width={window.innerWidth}
          height={window.innerHeight - HEADER_H - LEGEND_H}
          nodeCanvasObject={drawNode}
          nodeCanvasObjectMode={() => 'replace'}
          nodeLabel={nodeTooltip}
          nodeVal={getNodeSize}
          onNodeClick={handleNodeClick}
          linkColor={() => '#3a3a5a'}
          linkWidth={1}
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={1}
          linkLabel={link => link.label}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.35}
          cooldownTicks={150}
        />

        {/* Node detail panel */}
        {selectedNode && (
          <div style={styles.panel}>
            <div style={styles.panelHeader}>
              <span style={{ color: getNodeColor(selectedNode), fontWeight: 'bold' }}>
                {selectedNode.nodeType}
              </span>
              <button style={styles.closeBtn} onClick={() => setSelectedNode(null)}>✕</button>
            </div>
            <div style={{ color: '#b0b8d4', marginBottom: 8 }}>{selectedNode.label}</div>
            <hr style={styles.hr} />
            {Object.entries(selectedNode.rawData || {}).map(([k, v]) =>
              v != null && v !== '' ? (
                <div key={k} style={{ marginBottom: 4, lineHeight: 1.4 }}>
                  <span style={{ color: '#7ba7d4' }}>{k}: </span>
                  <span style={{ color: '#c8d0e8', wordBreak: 'break-all' }}>{String(v)}</span>
                </div>
              ) : null
            )}
          </div>
        )}
      </div>

      {/* Legend */}
      <div style={styles.legend}>
        {LEGEND_ITEMS.map(({ label, color }) => (
          <div key={label} style={styles.legendItem}>
            <span style={{ ...styles.swatch, background: color }} />
            <span style={{ color: '#aaa' }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

const styles = {
  root: {
    display: 'flex', flexDirection: 'column',
    height: '100vh', background: '#1a1a2e',
    color: '#e0e0e0', fontFamily: 'monospace',
  },
  header: {
    display: 'flex', alignItems: 'center', gap: 24,
    padding: '0 16px', height: HEADER_H,
    background: '#0d0d1a', borderBottom: '1px solid #2a2a4a',
    fontSize: 13, flexShrink: 0,
  },
  title:    { fontWeight: 'bold', color: '#7ba7d4', marginRight: 8 },
  dim:      { color: '#666' },
  liveGroup:{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 },
  dot:      { width: 8, height: 8, borderRadius: '50%', display: 'inline-block' },
  graphWrap:{ flex: 1, position: 'relative', overflow: 'hidden' },
  legend: {
    display: 'flex', flexWrap: 'wrap', gap: 12,
    padding: '0 16px', height: LEGEND_H, alignItems: 'center',
    background: '#0d0d1a', borderTop: '1px solid #2a2a4a',
    fontSize: 11, flexShrink: 0,
  },
  legendItem: { display: 'flex', alignItems: 'center', gap: 5 },
  swatch:     { width: 10, height: 10, borderRadius: '50%', display: 'inline-block' },
  panel: {
    position: 'absolute', top: 8, right: 8, width: 280,
    background: '#0d0d1a', border: '1px solid #2a2a4a',
    borderRadius: 6, padding: 12, fontSize: 12,
    maxHeight: 'calc(100% - 16px)', overflowY: 'auto', zIndex: 10,
  },
  panelHeader: {
    display: 'flex', justifyContent: 'space-between',
    alignItems: 'center', marginBottom: 6,
  },
  closeBtn: {
    background: 'none', border: 'none', color: '#888',
    cursor: 'pointer', fontSize: 14, lineHeight: 1,
  },
  hr: { borderColor: '#2a2a4a', margin: '8px 0' },
}
