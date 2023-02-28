export default function TransactionListItem({transaction, i, active, setActiveTransaction}) {
  const activate = () => {
    setActiveTransaction(i)
  }

  return (
    <div className="transactionListItem" onClick={activate}>
      {transaction.step}: {transaction.id} - "{transaction.type}"
    </div>
  )
}