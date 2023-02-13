import {useEffect, useState} from "react";
import TransactionListItem from "./TransactionListItem";
import {HiRefresh} from 'react-icons/hi';

export default function TransactionList(activeSession) {
  const [transactions, setTransactions] = useState([]);
  const [activeTransaction, setActiveTransaction] = useState(-1)

  useEffect(() => {
    getTransactions();
  }, [activeSession])

  const getTransactions = () => {
    if (!activeSession) return
    fetch(`/transactions?session_id=${activeSession.id}`, {
      method: 'GET'
    })
      .then(response => {

      })
      .then(json => {

      })
  }

  const refreshTransactions = () => {getTransactions();}

  return (
    <div className="transactionList">
      <h2>
        Transactions for {activeSession.name ? activeSession.name : "<null>"}
        <div className="refreshButton" onClick={refreshTransactions}><HiRefresh/></div>
      </h2>
      {transactions.map((transaction, i) => <TransactionListItem transaction={transaction}
                                                                 setActiveTransaction={setActiveTransaction}
                                                                 active={i == activeTransaction ? true : false}
                                                                 key={i}/>)}
    </div>
  )
}