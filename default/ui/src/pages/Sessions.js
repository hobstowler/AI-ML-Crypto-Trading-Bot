import {useEffect, useState} from "react";
import SessionGraph from "../components/SessionGraph";
import SessionNavigator from "../components/SessionNavigator";
import SessionDetail from "../components/SessionDetail";
import TransactionList from "../components/TransactionList";

export default function Sessions() {
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);

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
      <SessionNavigator/>
      <div>
        <SessionGraph/>
        <TransactionList/>
      </div>
      <SessionDetail/>
    </div>
  )
}