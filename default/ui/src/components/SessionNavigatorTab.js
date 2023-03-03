export default function SessionNavigatorTab({sessionType, activeSession, setActiveType}) {
    const setType = () => {
        setActiveType(sessionType)
    }

    return (
        <div onClick={setType} className={activeSession ? "sessionNavigatorTabActive" : "sessionNavigatorTab"}>
            {sessionType}
        </div>
    )
}