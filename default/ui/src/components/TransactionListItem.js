export default function TransactionListItem({transaction, i, active, setActiveTransaction}) {
  const activate = () => {
    setActiveTransaction(i)
  }

  return (
    <div className="transactionListItem" onClick={activate}>
      <h3>{transaction.transaction_id}</h3>
      <h4>"{transaction.type}"</h4>
    </div>
  )
}