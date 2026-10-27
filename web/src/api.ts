// The one door to the API. Every request carries X-Staff-Id (development identity, see the README),
// and every error becomes an ApiError with the API's own message, so screens show what went wrong.

export type Role = "manager" | "counselor";

export interface Staff {
  id: number;
  full_name: string;
  role: Role;
}

export interface Group {
  id: number;
  name: string;
  cohort_id: number;
  counselor_ids: number[];
}

export interface Trainee {
  id: number;
  first_name: string;
  last_name: string;
  cohort_id: number;
  group_id: number;
  primary_counselor_id: number;
  status: "active" | "on_leave" | "left" | "graduated";
}

export interface Goal {
  id: number;
  title: string;
  category: string;
  status: "not_started" | "in_progress" | "achieved" | "dropped";
  progress: number;
  target_date: string | null;
}

export interface Note {
  id: number;
  kind: "worked" | "did_not_work" | "observation" | "follow_up";
  text: string;
  created_at: string;
}

export interface Session {
  id: number;
  cohort_id: number;
  group_id: number | null;
  on: string;
  weekday: string;
  start: string;
  end: string;
  title: string;
  slot_id: number | null;
  cancelled: boolean;
}

export type Mark = "present" | "absent" | "excused";

export interface Sheet {
  session: Session;
  trainees: { trainee_id: number; name: string; status: Mark | null }[];
}

export interface AttendanceSummary {
  trainee_id: number;
  name: string;
  group_id: number;
  rate: number | null;
  percent: number | null;
  present: number;
  absent: number;
  excused: number;
  unmarked: number;
  missed_by_title: Record<string, number>;
}

export interface Approval {
  id: number;
  kind: string;
  trainee_id: number;
  reason: string;
  created_at: string;
}

export interface Task {
  id: number;
  description: string;
  owner_id: number;
  due_date: string | null;
  status: "open" | "done";
  created_by_id: number;
  source_meeting_id: number | null;
  trainee_id: number | null;
  completed_at: string | null;
}

export interface NewTask {
  description: string;
  due_date?: string | null;
  owner_id?: number | null;
  trainee_id?: number | null;
}

export interface Referral {
  message: string;
  steps: { action: string; contact: string | null }[];
}

export interface AgentStep {
  tool: string;
  arguments: Record<string, unknown>;
  ok: boolean;
  summary: string;
  latency_ms: number;
}

export interface AgentAnswer {
  outcome: "answer" | "referral" | "not_sent" | "stopped";
  answer: string | null;
  referral: Referral | null;
  note: string;
  steps: AgentStep[];
  sources: { source_id: number; citation: string; evidence: string; section: string | null }[];
  cost_usd: number | null;
  rounds: number;
}

export interface Turn {
  role: "user" | "assistant";
  content: string;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

const STAFF_KEY = "pacer.staff";

export function savedStaffId(): number | null {
  try {
    const value = Number(localStorage.getItem(STAFF_KEY));
    return Number.isInteger(value) && value > 0 ? value : null;
  } catch {
    return null; // storage can be blocked: the person simply signs in again
  }
}

export function saveStaffId(id: number | null): void {
  try {
    if (id === null) localStorage.removeItem(STAFF_KEY);
    else localStorage.setItem(STAFF_KEY, String(id));
  } catch {
    // not kept: fine
  }
}

let staffId: number | null = savedStaffId();

export function useStaff(id: number | null): void {
  staffId = id;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  if (staffId !== null) headers["X-Staff-Id"] = String(staffId);
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "השרת לא זמין. בדקו שה-API רץ (uvicorn) ונסו שוב.");
  }
  if (!response.ok) {
    let detail = `שגיאה ${response.status}`;
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
      else if (Array.isArray(data.detail)) detail = data.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      // keep the status line
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T; // deleted: no body
  return (await response.json()) as T;
}

export const api = {
  me: () => request<Staff>("GET", "/me"),
  groups: () => request<Group[]>("GET", "/groups"),
  trainees: () => request<Trainee[]>("GET", "/trainees"),
  goals: (id: number) => request<Goal[]>("GET", `/trainees/${id}/goals`),
  notes: (id: number) => request<Note[]>("GET", `/trainees/${id}/notes`),
  traineeAttendance: (id: number) => request<AttendanceSummary>("GET", `/trainees/${id}/attendance`),
  attendance: (cohortId: number) => request<AttendanceSummary[]>("GET", `/attendance?cohort_id=${cohortId}`),
  week: (day: string, cohortId: number) => request<Session[]>("GET", `/schedule/weeks/${day}?cohort_id=${cohortId}`),
  sheet: (sessionId: number) => request<Sheet>("GET", `/schedule/sessions/${sessionId}/attendance`),
  mark: (sessionId: number, marks: { trainee_id: number; status: Mark }[]) =>
    request<Sheet>("PUT", `/schedule/sessions/${sessionId}/attendance`, { marks }),
  pendingApprovals: () => request<Approval[]>("GET", "/approvals/pending"),
  staff: () => request<Staff[]>("GET", "/staff"),
  myTasks: () => request<Task[]>("GET", "/tasks/mine"),
  allTasks: () => request<Task[]>("GET", "/tasks"),
  traineeTasks: (id: number) => request<Task[]>("GET", `/trainees/${id}/tasks`),
  createTask: (task: NewTask) => request<Task>("POST", "/tasks", task),
  completeTask: (id: number) => request<Task>("POST", `/tasks/${id}/complete`),
  rescheduleTask: (id: number, dueDate: string | null) => request<Task>("PATCH", `/tasks/${id}`, { due_date: dueDate }),
  deleteTask: (id: number) => request<void>("DELETE", `/tasks/${id}`),
  chat: (message: string, history: Turn[]) => request<AgentAnswer>("POST", "/agent/chat", { message, history }),
};
