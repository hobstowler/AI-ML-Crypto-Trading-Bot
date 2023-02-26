import SessionListItem from "./SessionListItem";
import {useState, useEffect} from "react";
import {HiRefresh} from 'react-icons/hi';
import SessionNavigatorTab from "./SessionNavigatorTab";

export default function SessionNavigator({activeSession, setActiveSessionId}) {
  const [sessions, setSessions] = useState({});
  const [sessionTypes, setSessionTypes] = useState([]);
  const [activeSessionType, setActiveType] = useState("")

  useEffect(() => {
    getSessions();
  }, [])

  const refreshSessions = () => {getSessions();}

  const getSessions = () => {
    fetch('/sessions', {})
      .then(response => {
        if (!response.ok) throw Error("invalid response")
        else return response.json();
      })
      .then(json => {
        setSessions(json);
        setSessionTypes(Object.keys(json));
        setActiveType(Object.keys(json)[0]);
        console.log(Object.keys(json))
      })
      .catch((error) => {
        console.log(error);
      })
  }

  return (
      <div className="sessionNavigatorWrapper">
      <table className="sessionNavigator">
        <thead>
          <th className="sessionHeader" colSpan={2}>
            <h2>
              <div>Session List</div>
              <div className="refreshButton" onClick={refreshSessions}><HiRefresh/></div>
            </h2>
          </th>
        </thead>
        <tbody className="sessionNavigatorBody">
          <tr>
            <td className="sessionNavigatorTabs">
              {sessionTypes.map((type, i) => <SessionNavigatorTab sessionType={type}
                                                                          activeSession={type === activeSessionType ? true : false}
                                                                          setActiveType={setActiveType}
                                                                          key={i}
                                                                          />)}
            </td>
            <td className="sessions">
              {sessions[activeSessionType] ? sessions[activeSessionType].map((session, i) => <SessionListItem session={session}
                                                         active={session.id === activeSession ? true : false}
                                                         setActiveSession={setActiveSessionId}
                                                         key={i}/>) : `No sessions of type: ${activeSessionType}`}
            </td>
          </tr>
        </tbody>

      </table>
    </div>
  )
}