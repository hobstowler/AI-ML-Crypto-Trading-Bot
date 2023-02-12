export default function SessionListItem(session, i) {
  return (
    <div className="sessionListItem">
      <h3>{session.name}</h3>
      <div className="sessionInfo">
        <div>{session.model}</div>
        <div>{session.currency}</div>
      </div>
      <div className="sessionDates">
        <div>{session.start}</div>
        <div>{" - "}</div>
        <div>{session.end}</div>
      </div>
    </div>
  )
}