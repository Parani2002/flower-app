export default function How() {
  return (
    <>
      <div className="kicker">Architecture</div>
      <div className="hero">
        <div>
          <h1>One product, three layers</h1>
          <p>
            This is the same Flower project you already trained, wrapped so a hospital user can sign
            in, a coordinator can run federation, and a ward can score a patient.
          </p>
        </div>
      </div>
      <div className="panel steps">
        <div className="step">
          <b>1</b>
          <div>
            <h3>Frontend</h3>
            <p className="hint">
              React ward console: network stats, Flower controls, encounter scoring. Hospitals only
              see their partition.
            </p>
          </div>
        </div>
        <div className="step">
          <b>2</b>
          <div>
            <h3>Backend</h3>
            <p className="hint">
              FastAPI issues demo sessions, lists ward patients from the UCI diabetes file, starts{" "}
              <code>flwr run</code>, and publishes a serving bundle (LR + RF + XGBoost + meta-MLP).
            </p>
          </div>
        </div>
        <div className="step">
          <b>3</b>
          <div>
            <h3>Federated core</h3>
            <p className="hint">
              Each simulated hospital trains base learners on local rows, then a differentially
              private PyTorch meta-learner. FedAvg averages only those small MLP weights. That is
              the model the product serves after you publish.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
