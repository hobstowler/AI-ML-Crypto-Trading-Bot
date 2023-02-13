export default function TransactionListItem(transaction, i, active, setActiveTransaction) {
  const activate = () => {
    setActiveTransaction(i)
  }

  return (
    <div className="transactionListItem" onClick={activate}>
      <h3>{transaction.time}</h3>
    </div>
  )
}