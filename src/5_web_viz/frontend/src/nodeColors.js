export const COLORS = {
  match:             '#4a6fa5',
  team_home:         '#2d6a4f',
  team_away:         '#c1121f',
  player_home:       '#52b788',
  player_away:       '#e07070',
  GoalEvent:         '#40c057',
  ShotEvent:         '#f59f00',
  FoulEvent:         '#c92a2a',
  YellowCardEvent:   '#f08c00',
  RedCardEvent:      '#e03131',
  CornerEvent:       '#7048e8',
  FreeKickEvent:     '#1c7ed6',
  OffsideEvent:      '#868e96',
  SubstitutionEvent: '#ae8a4e',
  event:             '#868e96',
}

const SIZES = { match: 16, team: 14, player: 10 }
const EVENT_SIZE = 8

export function getNodeColor(node) {
  if (!node) return COLORS.event
  if (node.nodeType === 'match') return COLORS.match
  if (node.nodeType === 'team')
    return node.side === 'away' ? COLORS.team_away : COLORS.team_home
  if (node.nodeType === 'player')
    return node.side === 'away' ? COLORS.player_away : COLORS.player_home
  return COLORS[node.nodeType] || COLORS.event
}

export function getNodeSize(node) {
  if (!node) return EVENT_SIZE
  return SIZES[node.nodeType] ?? EVENT_SIZE
}

export const LEGEND_ITEMS = [
  { label: 'Match',       color: COLORS.match },
  { label: 'Home Team',   color: COLORS.team_home },
  { label: 'Away Team',   color: COLORS.team_away },
  { label: 'Home Player', color: COLORS.player_home },
  { label: 'Away Player', color: COLORS.player_away },
  { label: 'Goal',        color: COLORS.GoalEvent },
  { label: 'Shot',        color: COLORS.ShotEvent },
  { label: 'Foul',        color: COLORS.FoulEvent },
  { label: 'Yellow Card', color: COLORS.YellowCardEvent },
  { label: 'Corner',      color: COLORS.CornerEvent },
  { label: 'Free Kick',   color: COLORS.FreeKickEvent },
  { label: 'Other',       color: COLORS.event },
]
