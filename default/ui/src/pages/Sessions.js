import {useEffect, useState} from "react";
import SessionGraph from "../components/SessionGraph";

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
      <SessionGraph/>
    </div>
  )
}