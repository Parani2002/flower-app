import { useEffect, useState } from "react";
import { api, Patient } from "../api";

export default function Predict() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [hospital, setHospital] = useState("");
  const [result, setResult] = useState<{
    probability: number;
    risk: string;
    explanation: string;
    patient: Patient;
  } | null>(null);
  const [error, setError] = useState("");
  const [busyId, setBusyId] = useState<number | null>(null);

  useEffect(() => {
    api
      .patients()
      .then((d) => {
        setPatients(d.patients);
        setHospital(d.hospital.name);
      })
      .catch((e) => setError(e.message));
  }, []);

  return (
    <>
      <div className="kicker">Clinical board</div>
      <div className="hero">
        <div>
          <h1>{hospital || "Ward"}</h1>
          <p>
            Score a current encounter with the published stacked model. The label from the dataset
            is shown after scoring so you can discuss false positives in the viva — it is not shown
            to the model.
          </p>
        </div>
      </div>
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="panel">
          <span className={`pill ${result.risk}`}>{result.risk} risk</span>
          <h2 className="serif">P(readmit &lt; 30 days) = {(result.probability * 100).toFixed(1)}%</h2>
          <p className="hint">{result.explanation}</p>
          <p>
            Recorded outcome in the file:{" "}
            <strong>{result.patient.actual_early_readmit ? "early readmit" : result.patient.readmitted_label}</strong>
          </p>
        </div>
      )}
      <div className="panel">
        <h3>Encounters on this node</h3>
        <table className="table">
          <thead>
            <tr>
              <th>Encounter</th>
              <th>Age</th>
              <th>Stay</th>
              <th>Prior inpatient</th>
              <th>A1C</th>
              <th>Insulin</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {patients.map((p) => (
              <tr key={p.encounter_id}>
                <td>{p.encounter_id}</td>
                <td>
                  {p.gender} {p.age}
                </td>
                <td>{p.time_in_hospital}d</td>
                <td>{p.number_inpatient}</td>
                <td>{p.A1Cresult}</td>
                <td>{p.insulin}</td>
                <td>
                  <button
                    className="btn ghost"
                    disabled={busyId != null}
                    onClick={async () => {
                      setBusyId(p.encounter_id);
                      setError("");
                      try {
                        setResult(await api.predict(p.encounter_id));
                      } catch (e) {
                        setError(e instanceof Error ? e.message : "Predict failed");
                      } finally {
                        setBusyId(null);
                      }
                    }}
                  >
                    {busyId === p.encounter_id ? "Scoring…" : "Score"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
