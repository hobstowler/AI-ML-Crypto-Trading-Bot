export default function SessionDetail(session) {
  return (
    <div className="sessionDetail">
      <h2>Details for {session.name ? session.name : "<null>"}</h2>
    </div>
  )
}