import SessionListItem from "./SessionListItem";
import {useState, useEffect} from "react";
import {HiRefresh} from 'react-icons/hi';

export default function SessionNavigator(sessionList, activeSession, setActiveSession) {
  const [sessions, setSessions] = useState([]);

  useEffect(() => {
    getSessions();
  }, [])

  const refreshSessions = () => {getSessions();}

  const getSessions = () => {
    fetch('/sessions', {

    })
      .then(response => {
        if (!response.ok) throw Error("invalid response")
        else return response.json();
      })
      .then(json => {
        console.log(json);
      })
      .catch((error) => {
        console.log(error);
      })
  }

  return (
    <div className="sessionNavigator">
      <div className="sessionHeader">
        <h2>
          Session List
          <div className="refreshButton" onClick={refreshSessions}><HiRefresh/></div>
        </h2>
      </div>

      {sessions.map((session, i) => <SessionListItem session={session}
                                                     active={session.name === activeSession ? true : false}
                                                     setActiveSession={setActiveSession}
                                                     key={i}/>)}
    </div>
  )
}