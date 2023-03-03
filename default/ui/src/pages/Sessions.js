import {useEffect, useState} from "react";
import SessionGraph from "../components/SessionGraph";
import SessionNavigator from "../components/SessionNavigator";
import SessionDetail from "../components/SessionDetail";
import TransactionList from "../components/TransactionList";

export default function Sessions() {
  const [activeSessionId, setActiveSessionId] = useState(0);

  return (
    <div className="sessionPage">
      <SessionNavigator activeSession={activeSessionId} setActiveSessionId={setActiveSessionId}/>
      <div className="sessionMid">
        <SessionGraph sessionId={activeSessionId} />
      </div>
      <div className="sessionDetails">
        <SessionDetail sessionId={activeSessionId} />
        <TransactionList sessionId={activeSessionId} />
      </div>
    </div>
  )
}