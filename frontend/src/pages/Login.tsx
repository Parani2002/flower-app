import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, setToken } from "../api";

export default function Login() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("sthelens");
  const [password, setPassword] = useState("ward-demo");
  const [error, setError] = useState("");

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    try {
      const res = await api.login(username, password);
      setToken(res.token);
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not sign in");
    }
  }

  return (
    <div className="login">
      <section className="login-art">
        <div className="kicker">Final year product</div>
        <h1>Aegis Ward</h1>
        <p className="hint" style={{ color: "#ead9bb", maxWidth: 420 }}>
          Three hospitals. One readmission model. No shared patient files.
        </p>
      </section>
      <section className="login-box">
        <form className="login-card" onSubmit={onSubmit}>
          <div className="kicker">Hospital access</div>
          <h2 className="serif">Sign in to the care net</h2>
          <label htmlFor="user">Username</label>
          <input id="user" value={username} onChange={(e) => setUsername(e.target.value)} />
          <label htmlFor="pass">Password</label>
          <input
            id="pass"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {error && <p className="error">{error}</p>}
          <button className="btn" style={{ marginTop: 18, width: "100%" }} type="submit">
            Enter ward
          </button>
          <p className="hint">
            Hospital: <code>sthelens</code> / <code>riverside</code> / <code>oakridge</code> —
            password <code>ward-demo</code>. Coordinator: <code>admin</code> /{" "}
            <code>aegis-admin</code>.
          </p>
        </form>
      </section>
    </div>
  );
}
