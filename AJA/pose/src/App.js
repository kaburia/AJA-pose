import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Landing } from "./Components/Landing/Landing";
import React from "react";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route>
          <Route path="/" element={<Landing />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
