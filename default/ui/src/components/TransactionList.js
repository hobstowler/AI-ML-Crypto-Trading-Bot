import {useEffect, useState} from "react";
import TransactionListItem from "./TransactionListItem";
import {HiRefresh} from 'react-icons/hi';

export default function TransactionList({sessionId}) {
  console.log(sessionId)
  const [transactions, setTransactions] = useState([]);
  const [activeTransactionId, setActiveTransactionId] = useState(-1)

  useEffect(() => {
    setTransactions([])
    //getTransactions();
  }, [sessionId])

  const getTransactions = () => {
    if (sessionId === 0) return
    fetch(`/raw_transactions?session_id=${sessionId}`, {
      method: 'GET'
    })
      .then(response => {
        if (!response.ok) throw Error('invalid response');
        else return response.json();
      })
      .then(json => {
        setTransactions(json)
      })
  }

  const refreshTransactions = () => {getTransactions();}

  return (
    <div className="transactionDetail">
      <h2>
        <div>Transactions List</div>
        <div className="refreshButton" onClick={refreshTransactions}><HiRefresh/></div>
      </h2>
      <div className="transactionListDetail">
        {transactions.length == 0 ? <button onClick={getTransactions}>Get Transactions</button> : null}
        {transactions.map((transaction, i) => <TransactionListItem transaction={transaction}
                                                                 setActiveTransaction={setActiveTransactionId}
                                                                 active={i == activeTransactionId ? true : false}
                                                                 key={i}/>)}
      </div>

    </div>
  )
}