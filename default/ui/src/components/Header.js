export default function Header() {
  return (
    <div>
      <div className="headerWrapper">
        <h1>AI/ML Trading Bot Project</h1>
        <div className="navWrapper">
          <ul>
            <li><Link to="/">Home</Link></li>
            <li><Link to="/sessions">Sessions</Link></li>
          </ul>
        </div>
      </div>
    </div>
  )
}