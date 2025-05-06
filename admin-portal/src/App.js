import React from 'react';
import './App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import Layout from './pages/Layout'; 
import Dashboard from './pages/Dashboard';
import Reports from './pages/Reports';
import MapView from './pages/MapView';
import Analytics from './pages/Analytics';
import Repair from './pages/Repair';
import Historical from './pages/Historical';
import Login from './pages/Login';
import Profile from './pages/Profile';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout/>}>
          <Route index element={<Dashboard/>}/>
          <Route path='/reports' element={<Reports/>} />
          <Route path='/map' element={<MapView/>} />
          <Route path='/analytics' element={<Analytics/>} />
          <Route path='repairs' element={<Repair/>} />
          <Route path='/historical' element={<Historical/>} />
          <Route path='/profile' element={<Profile/>} />
        </Route>
        <Route path='/login' element={<Login/>} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
