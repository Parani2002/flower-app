import type { ReactNode } from "react";
import { Navigate, NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { clearToken, getToken } from "./api";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Training from "./pages/Training";
import Predict from "./pages/Predict";
import How from "./pages/How";

function Guard({ children }: { children: ReactNode }) {
  if (!getToken()) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function Layout({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  return (
    <div className="shell">
      <aside className="side">
        <div className="mark">
          <div className="mark-seal">A</div>
          <div>
            <strong>Aegis Ward</strong>
            <small>Federated care net</small>
          </div>
        </div>
        <nav>
          <NavLink to="/" end>
            Network
          </NavLink>
          <NavLink to="/training">Federation</NavLink>
          <NavLink to="/predict">Ward board</NavLink>
          <NavLink to="/how">How it works</NavLink>
        </nav>
        <div className="side-foot">
          Patient records stay on each hospital node. Only model weights move.
          <div>
            <button
              onClick={() => {
                clearToken();
                navigate("/login");
              }}
            >
              Sign out
            </button>
          </div>
        </div>
      </aside>
      <div className="main">{children}</div>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/"
        element={
          <Guard>
            <Layout>
              <Dashboard />
            </Layout>
          </Guard>
        }
      />
      <Route
        path="/training"
        element={
          <Guard>
            <Layout>
              <Training />
            </Layout>
          </Guard>
        }
      />
      <Route
        path="/predict"
        element={
          <Guard>
            <Layout>
              <Predict />
            </Layout>
          </Guard>
        }
      />
      <Route
        path="/how"
        element={
          <Guard>
            <Layout>
              <How />
            </Layout>
          </Guard>
        }
      />
    </Routes>
  );
}
