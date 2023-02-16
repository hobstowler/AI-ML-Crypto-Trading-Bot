import {Link} from 'react-router-dom';

export default function Home() {
  return (
    <div>
      <p>Go to the <Link to="/models">Session Management</Link> page.</p>
    </div>
  )
}