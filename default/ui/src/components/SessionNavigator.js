import SessionListItem from "./SessionListItem";

export default function SessionNavigator(sessions, activeSession, setActiveSession) {
  return (
    <div className="sessionNavigator">
      <h2>Session List</h2>
      {sessions.map((session, i) => <SessionListItem session={session}
                                                     active={session.name === activeSession ? true : false}
                                                     setActiveSession={setActiveSession}
                                                     key={i}/>)}
    </div>
  )
}