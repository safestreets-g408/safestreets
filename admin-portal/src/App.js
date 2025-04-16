import React from 'react';
import './App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import Layout from './pages/Layout'; 
import Dashboard from './pages/Dashboard';
import Reports from './pages/Reports';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout/>}>
          <Route index element={<Dashboard/>}/>
          <Route path='/reports' element={<Reports/>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
