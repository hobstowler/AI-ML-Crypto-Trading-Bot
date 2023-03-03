import {HiTrash} from 'react-icons/hi';

export default function SessionListItem({session, setActiveSession, deleteSession, key, active}) {
  const setActive = () => {
    if (active) setActiveSession(0)
    else setActiveSession(session.id)
  }

  const del_self = () => {
    deleteSession(session.id, key);
  }

  return (
    <div className={active ? "sessionListItemActive" : "sessionListItem"} onClick={setActive}>
      <h3><div>{session.session_name}</div>
        {active ? <div className="deleteSession"><HiTrash/></div> : null}</h3>
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