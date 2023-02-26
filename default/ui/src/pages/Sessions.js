import {useEffect, useState} from "react";
import SessionGraph from "../components/SessionGraph";
import SessionNavigator from "../components/SessionNavigator";
import SessionDetail from "../components/SessionDetail";
import TransactionList from "../components/TransactionList";

export default function Sessions() {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(0);

  useEffect(() => {
    getSessions();
  }, [])

  const getSessions = () => {

  }

  const refreshSessions = () => {
    getSessions();
  }

  return (
    <div>
      <SessionNavigator activeSession={activeSessionId} setActiveSessionId={setActiveSessionId}/>
      <div>
        <SessionGraph/>
        <TransactionList/>
      </div>
      <SessionDetail/>
    </div>
  )
}