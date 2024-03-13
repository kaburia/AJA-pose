import { BrowserRouter, Routes, Route } from "react-router-dom";
import React from "react";
import Landing from "./Components/Landing/Landing";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route>
          <Route path="/" element={<Landing/>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
