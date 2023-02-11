import {BrowserRouter, Route, Routes, useSearchParams} from 'react-router-dom';
import './App.css';
import Home from './pages/Home'
import Sessions from "./pages/Sessions";
import Header from "./components/Header";

function App() {
  return (
    <div className="App">
      <Header/>
      <BrowserRouter>
        <Routes>
          <Route path="/"><Home/></Route>
          <Route path="/sessions"><Sessions/></Route>
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
