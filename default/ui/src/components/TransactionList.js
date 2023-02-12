import {useState} from "react";

export default function TransactionList(sessionName, transactions) {
  const [activeTransaction, setActiveTransaction] = useState(-1)

  return (
    <div>
      <h2>Transactions for {sessionName}</h2>
      {transactions.map((transaction, i) => <TransactionListItem transaction={transaction}
                                                                 active={i == activeTransaction ? true : false}
                                                                 key={i}/>)}
    </div>
  )
}