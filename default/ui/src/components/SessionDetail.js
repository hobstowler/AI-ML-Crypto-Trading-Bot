import {useEffect, useState} from "react";

export default function SessionDetail({sessionId}) {
  const [session, setSession] = useState({})

  useEffect(() => {
    if (sessionId !== 0) getDetails();
    else setSession({})
  }, [sessionId])

  const getDetails = () => {
    fetch(`/session/${sessionId}`, {})
      .then(response => {
        if (!response.ok) throw Error()
        else return response.json()
      })
      .then(json => setSession(json))
      .catch(error => console.log(error))
  }
  console.log(session)

  return (
    <div className="sessionDetail">
      <h2>Details for "{session.session_name ? session.session_name : "<null>"}"</h2>
      <div className="sessionDetailInfo">
        <div><b>Session Name:</b> {session.session_name ? session.session_name : "<null session name>"}</div>
        <div><b>Session Type: </b>{session.type ? session.type : "<null session type>"}</div>
        <div><b>Session ID: </b> {session.id !== null ? session.id : "<null session id>"}</div>
        <div><b>Model Name: </b> {session.model_name ? session.model_name : "<null model name>"}</div>
        <div><b>Crypto Type: </b> {session.crypto_type ? session.crypto_type : "<null crypto type>"}</div>
        <div><b>Date Range: </b> {session.session_start ? session.session_start.toLocaleString() : "<null start date>"} -> {session.session_end ? session.session_end.toLocaleString() : "<null end date>"}</div>
        <div><b>Balance: </b> {session.starting_balance ? session.starting_balance.toLocaleString("en-US", {style: "currency", currency: "USD"}) : "<null start balance>"} -> {session.ending_balance ? session.ending_balance.toLocaleString("en-US", {style: "currency", currency: "USD"}) : "<null end balance>"}</div>
        <div><b>Coins: </b> {session.starting_coins !== null ? session.starting_coins : "<null start coins>"} -> {session.ending_coins ? session.ending_coins : "<null end coins>"}</div>
        <div><b>Coins Bought/Sold: </b>{session.coins_bought ? session.coins_bought : "<null coins bought>"}/{session.coins_sold ? session.coins_sold : "<null coins sold>"}</div>
        <div><b>Total Trades: </b> {session.number_of_trades ? session.number_of_trades : "<null trade quantity>"}</div>
      </div>
    </div>
  )
}