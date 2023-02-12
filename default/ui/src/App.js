import {BrowserRouter, Route, Routes, useSearchParams} from 'react-router-dom';
import './App.css';
import Home from './pages/Home'
import Sessions from "./pages/Sessions";
import Header from "./components/Header";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Header/>
        <Routes>
          <Route path="/" element={<Home/>}/>
          <Route path="/sessions" element={<Sessions/>}/>
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
