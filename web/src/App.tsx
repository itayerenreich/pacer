import { useCallback, useEffect, useMemo, useState } from "react";

import { api, ApiError, saveStaffId, savedStaffId, useStaff, type Group, type Staff, type Trainee, type Turn } from "./api";
import { Chat, type Exchange } from "./screens/Chat";
import { Dashboard } from "./screens/Dashboard";
import { Schedule } from "./screens/Schedule";
import { SignIn } from "./screens/SignIn";
import { Tasks } from "./screens/Tasks";
import { Trainees } from "./screens/Trainees";

// What every screen may need: who is signed in, and the names behind the ids.
export interface Academy {
  me: Staff;
  groups: Map<number, Group>;
  trainees: Map<number, Trainee>;
  cohortId: number;
}

const SCREENS = [
  { id: "chat", label: "שאלה לפייסר", short: "שאלה" },
  { id: "trainees", label: "חניכים", short: "חניכים" },
  { id: "schedule", label: "לו״ז ונוכחות", short: "לו״ז" },
  { id: "tasks", label: "משימות", short: "משימות" },
  { id: "overview", label: "תמונת מצב", short: "מצב" },
] as const;

type ScreenId = (typeof SCREENS)[number]["id"];

function screenFromHash(): ScreenId {
  const id = window.location.hash.replace("#/", "");
  return SCREENS.some((s) => s.id === id) ? (id as ScreenId) : "chat";
}

export function App() {
  const [academy, setAcademy] = useState<Academy | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [checking, setChecking] = useState(savedStaffId() !== null);
  const [screen, setScreen] = useState<ScreenId>(screenFromHash);
  // The conversation lives here, so it survives moving between screens. Nothing is stored anywhere else.
  const [exchanges, setExchanges] = useState<Exchange[]>([]);

  useEffect(() => {
    const onHash = () => setScreen(screenFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const signIn = useCallback(async (id: number) => {
    useStaff(id);
    setChecking(true);
    setProblem(null);
    try {
      const [me, groups, trainees] = await Promise.all([api.me(), api.groups(), api.trainees()]);
      saveStaffId(id);
      setAcademy({
        me,
        groups: new Map(groups.map((g) => [g.id, g])),
        trainees: new Map(trainees.map((t) => [t.id, t])),
        cohortId: groups[0]?.cohort_id ?? 1,
      });
    } catch (error) {
      useStaff(null);
      saveStaffId(null);
      setProblem(error instanceof ApiError && error.status === 401 ? "אין איש צוות עם המספר הזה." : String((error as Error).message));
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    const saved = savedStaffId();
    if (saved !== null) void signIn(saved);
  }, [signIn]);

  const signOut = () => {
    useStaff(null);
    saveStaffId(null);
    setAcademy(null);
    setExchanges([]);
  };

  const history = useMemo<Turn[]>(
    () =>
      exchanges.flatMap((e) =>
        e.reply?.answer ? [{ role: "user" as const, content: e.question }, { role: "assistant" as const, content: e.reply.answer }] : [],
      ),
    [exchanges],
  );

  if (!academy) return <SignIn onSignIn={signIn} problem={problem} busy={checking} />;

  return (
    <div className="shell">
      <aside className="rail">
        <div className="mark" aria-hidden="true">
          Pacer
        </div>
        <nav aria-label="מסכים">
          {SCREENS.map((s) => (
            <a key={s.id} href={`#/${s.id}`} aria-current={screen === s.id ? "page" : undefined}>
              <span className="long">{s.label}</span>
              <span className="short" aria-hidden="true">
                {s.short}
              </span>
            </a>
          ))}
        </nav>
        <div className="who">
          {academy.me.full_name}
          <br />
          {academy.me.role === "manager" ? "מנהל/ת" : "מדריך/ה"}
          <br />
          <button type="button" onClick={signOut}>
            החלפת משתמש
          </button>
        </div>
      </aside>
      <main className="page">
        {screen === "chat" && <Chat academy={academy} exchanges={exchanges} setExchanges={setExchanges} history={history} />}
        {screen === "trainees" && <Trainees academy={academy} />}
        {screen === "schedule" && <Schedule academy={academy} />}
        {screen === "tasks" && <Tasks academy={academy} />}
        {screen === "overview" && <Dashboard academy={academy} />}
      </main>
    </div>
  );
}
