import { useEffect, useState } from "react";
import { api } from "../api";

type Status = {
  running: boolean;
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  error: string | null;
  log_tail: string;
};

export default function Training() {
  const [status, setStatus] = useState<Status | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function refresh() {
    const s = (await api.trainingStatus()) as Status;
    setStatus(s);
  }

  useEffect(() => {
    refresh().catch((e) => setMessage(e.message));
    const id = setInterval(() => refresh().catch(() => undefined), 4000);
    return () => clearInterval(id);
  }, []);

  return (
    <>
      <div className="kicker">Flower simulation</div>
      <div className="hero">
        <div>
          <h1>Run the federation</h1>
          <p>
            Coordinator starts <code>flwr run</code> in the existing PyTorch app: three SuperNodes,
            FedAvg on the meta-MLP, Opacus on each client. Then publish a serving pack so the ward
            board can score patients.
          </p>
        </div>
      </div>
      <div className="row">
        <button
          className="btn"
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            setMessage("");
            try {
              const res = await api.startTraining();
              setMessage(res.detail);
              await refresh();
            } catch (e) {
              setMessage(e instanceof Error ? e.message : "Failed");
            } finally {
              setBusy(false);
            }
          }}
        >
          Start Flower round
        </button>
        <button
          className="btn ghost"
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            setMessage("Publishing serving pack — this can take a few minutes…");
            try {
              const res = await api.publish();
              setMessage(
                `Published. AUC ${Number(res.auc).toFixed(3)} · federated meta: ${String(res.used_federated_meta)}`,
              );
            } catch (e) {
              setMessage(e instanceof Error ? e.message : "Publish failed");
            } finally {
              setBusy(false);
            }
          }}
        >
          Publish serving model
        </button>
      </div>
      {message && <p className="hint">{message}</p>}
      <div className="panel">
        <h3>Run status</h3>
        <p className="hint">
          {status?.running ? "Simulation running" : "Idle"}
          {status?.started_at ? ` · started ${status.started_at}` : ""}
          {status?.exit_code != null ? ` · exit ${status.exit_code}` : ""}
        </p>
        {status?.error && <p className="error">{status.error}</p>}
        <pre className="log">{status?.log_tail || "No Flower log yet."}</pre>
      </div>
    </>
  );
}
