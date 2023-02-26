export default function SessionListItem({session, setActiveSession, active}) {
  const setActive = () => {
    setActiveSession(session.id)
      console.log('hello')
  }

  return (
    <div className={active ? "sessionListItemActive" : "sessionListItem"} onClick={setActive}>
      <h3>{session.session_name}</h3>
      <div className="sessionInfo">
        <div>{session.model_name ? session.model_name : "<null>"}</div>
          <div><pre> | </pre></div>
        <div>{session.type ? session.type : "<null>"}</div>
      </div>
      <div className="sessionDates">
        <div><b>Start: </b> {session.session_start.toLocaleString()}</div>
      <div><b>End: </b>{session.session_end ? session.session_end.toLocaleString() : ""}</div>
      </div>
    </div>
  )
}