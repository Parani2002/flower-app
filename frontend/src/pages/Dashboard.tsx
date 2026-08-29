import { useEffect, useState } from "react";
import { api } from "../api";

type Overview = {
  patients_in_network: number;
  early_readmits: number;
  early_rate: number;
  serving_ready: boolean;
  metrics: {
    auc?: number;
    f1?: number;
    recall?: number;
    used_federated_meta?: boolean;
    built_at?: string;
  } | null;
  local?: { patients: number; early_readmits: number; hospital: { name: string; city: string } };
  hospitals: { name: string; city: string; focus: string }[];
};

export default function Dashboard() {
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .overview()
      .then((d) => setData(d as Overview))
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p>Loading network…</p>;

  const m = data.metrics;
  return (
    <>
      <div className="kicker">Care network</div>
      <div className="hero">
        <div>
          <h1>{data.local ? data.local.hospital.name : "Aegis federation"}</h1>
          <p>
            Diabetic encounters stay on each ward. Flower averages a small meta-learner so the
            network shares skill, not records.
          </p>
        </div>
      </div>
      <div className="grid">
        <div className="card">
          <div className="label">Network patients</div>
          <div className="value">{data.patients_in_network.toLocaleString()}</div>
        </div>
        <div className="card">
          <div className="label">Early readmit rate</div>
          <div className="value">{(data.early_rate * 100).toFixed(1)}%</div>
        </div>
        <div className="card">
          <div className="label">Global AUC</div>
          <div className="value">{m?.auc ? m.auc.toFixed(3) : "—"}</div>
        </div>
        <div className="card">
          <div className="label">Serving model</div>
          <div className="value" style={{ fontSize: 22 }}>
            {data.serving_ready ? (m?.used_federated_meta ? "Federated" : "Local pack") : "Not published"}
          </div>
        </div>
      </div>
      {data.local && (
        <div className="panel">
          <h3>Your partition</h3>
          <p className="hint">
            {data.local.patients.toLocaleString()} encounters on this node ·{" "}
            {data.local.early_readmits.toLocaleString()} early readmits · {data.local.hospital.city}
          </p>
        </div>
      )}
      <div className="panel">
        <h3>Member hospitals</h3>
        <table className="table">
          <thead>
            <tr>
              <th>Hospital</th>
              <th>City</th>
              <th>Focus</th>
            </tr>
          </thead>
          <tbody>
            {data.hospitals.map((h) => (
              <tr key={h.name}>
                <td>{h.name}</td>
                <td>{h.city}</td>
                <td>{h.focus}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {m?.built_at && (
          <p className="hint">Last published {m.built_at}. F1 {m.f1?.toFixed(3)} · recall {m.recall?.toFixed(3)}</p>
        )}
      </div>
    </>
  );
}
