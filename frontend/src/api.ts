export type Role = "admin" | "hospital";

export type Hospital = {
  id: number;
  slug: string;
  name: string;
  city: string;
  focus: string;
};

export type User = {
  username: string;
  role: Role;
  hospital_id: number | null;
  name: string;
};

export type Patient = {
  encounter_id: number;
  age: string;
  gender: string;
  race: string;
  time_in_hospital: number;
  num_medications: number;
  number_inpatient: number;
  number_emergency: number;
  A1Cresult: string;
  insulin: string;
  diabetesMed: string;
  actual_early_readmit: boolean;
  readmitted_label: string;
};

const TOKEN_KEY = "aegis-token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  if (token) {
    (headers as Record<string, string>).Authorization = `Bearer ${token}`;
  }
  const res = await fetch(path, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error((data as { detail?: string }).detail || res.statusText);
  }
  return data as T;
}

export const api = {
  login: (username: string, password: string) =>
    request<{ token: string; user: User; hospital: Hospital | null }>("/api/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  overview: () => request<Record<string, unknown>>("/api/overview"),
  patients: () => request<{ hospital: Hospital; patients: Patient[] }>("/api/patients"),
  predict: (encounter_id: number) =>
    request<{
      probability: number;
      risk: string;
      patient: Patient;
      explanation: string;
    }>("/api/predict", {
      method: "POST",
      body: JSON.stringify({ encounter_id }),
    }),
  startTraining: () =>
    request<{ ok: boolean; detail: string }>("/api/training/start", { method: "POST" }),
  trainingStatus: () => request<Record<string, unknown>>("/api/training/status"),
  publish: () =>
    request<Record<string, unknown>>("/api/serving/publish", {
      method: "POST",
      body: JSON.stringify({ max_rows: 8000 }),
    }),
};
