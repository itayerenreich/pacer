import { useEffect, useRef, useState, type Dispatch, type FormEvent, type KeyboardEvent, type SetStateAction } from "react";

import { api, type AgentAnswer, type AgentStep, type Turn } from "../api";
import type { Academy } from "../App";
import { RichText } from "../RichText";

export interface Exchange {
  id: number;
  question: string;
  reply: AgentAnswer | null; // null while the agent works
  error: string | null;
}

const LEGS: Record<string, string> = {
  find_trainees: "חיפוש חניך",
  trainee_record: "התיק החינוכי",
  attendance: "נוכחות",
  week_schedule: "הלו״ז",
  search_library: "הספרייה המקצועית",
  ask_coach: "התייעצות עם המלווה",
  report_exceptional_incident: "אירוע חריג",
};

const STARTERS = {
  counselor: [
    "מי בקבוצה שלי מגיע הכי פחות לפעילויות בחודש האחרון?",
    "איך לעזור לבן שפירא להגיע בזמן לספורט בוקר?",
    "מה יש בלו״ז של מחר?",
  ],
  manager: [
    "איזו קבוצה מגיעה יותר טוב בחודש האחרון?",
    "איך להתמודד עם נדב נחום שלא עומד ביעדים?",
    "מה אומרים המקורות על ללמד כל חניך לפי היכולת שלו?",
  ],
};

interface Props {
  academy: Academy;
  exchanges: Exchange[];
  setExchanges: Dispatch<SetStateAction<Exchange[]>>;
  history: Turn[];
}

export function Chat({ academy, exchanges, setExchanges, history }: Props) {
  const [draft, setDraft] = useState("");
  const busy = exchanges.some((e) => e.reply === null && e.error === null);
  const nextId = useRef(1);
  const latest = useRef<HTMLElement>(null);
  const last = exchanges[exchanges.length - 1];

  // Bring each new question, and then its answer, into view: the answer can be long, and a referral
  // must never sit unseen below the fold.
  useEffect(() => {
    latest.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [last?.id, last?.reply, last?.error]);

  const ask = async (question: string) => {
    const text = question.trim();
    if (text.length < 5 || busy) return;
    const id = nextId.current++;
    setDraft("");
    setExchanges((all) => [...all, { id, question: text, reply: null, error: null }]);
    try {
      const reply = await api.chat(text, history);
      setExchanges((all) => all.map((e) => (e.id === id ? { ...e, reply } : e)));
    } catch (error) {
      setExchanges((all) => all.map((e) => (e.id === id ? { ...e, error: (error as Error).message } : e)));
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void ask(draft);
  };

  const onKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void ask(draft);
    }
  };

  return (
    <>
      <header>
        <h1>שאלה לפייסר</h1>
        <p className="quiet">
          פייסר בודק בעצמו את מה שצריך: חניכים, נוכחות, הלו״ז, הספרייה והמלווה. הוא רואה רק את מה שמותר לך
          לראות, ולא משנה דבר. שמות ומידע בריאותי לא יוצאים מהמערכת.
        </p>
      </header>

      {exchanges.length === 0 && (
        <div className="starters">
          {STARTERS[academy.me.role].map((s) => (
            <button key={s} type="button" onClick={() => void ask(s)}>
              {s}
            </button>
          ))}
        </div>
      )}

      <div className="conversation" aria-live="polite">
        {exchanges.map((e) => (
          <article key={e.id} ref={e === last ? latest : undefined}>
            <p className="question">{e.question}</p>
            {e.reply === null && e.error === null && <p className="thinking">בודק...</p>}
            {e.error && <p className="error">{e.error}</p>}
            {e.reply && <Reply reply={e.reply} academy={academy} />}
          </article>
        ))}
      </div>

      <form className="composer" onSubmit={submit}>
        <label htmlFor="message" className="visually-hidden">
          השאלה שלך
        </label>
        <textarea
          id="message"
          className="field"
          rows={2}
          maxLength={2000}
          placeholder="למשל: איך לעזור לחניך שמתבייש לדבר בקבוצה?"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKey}
        />
        <button className="button" type="submit" disabled={busy || draft.trim().length < 5}>
          שליחה
        </button>
      </form>
    </>
  );
}

function Reply({ reply, academy }: { reply: AgentAnswer; academy: Academy }) {
  if (reply.outcome === "referral" && reply.referral) {
    return (
      <div className="referral" role="alert">
        <h3>זה עלול להיות אירוע חריג</h3>
        <p>{reply.referral.message}</p>
        <ol>
          {reply.referral.steps.map((step) => (
            <li key={step.action}>
              {step.action}
              {step.contact && <span className="contact">{step.contact}</span>}
            </li>
          ))}
        </ol>
      </div>
    );
  }
  if (reply.outcome === "not_sent") {
    return <p>{reply.note}</p>;
  }
  return (
    <div className="exchange">
      <div className="answer">
        <RichText text={reply.answer ?? ""} />
        {reply.sources.length > 0 && (
          <div className="sources">
            <strong>מקורות שנבדקו</strong>
            <ul>
              {reply.sources.map((s) => (
                <li key={`${s.source_id}-${s.section}`}>
                  {s.citation}
                  {s.section ? `, ${s.section}` : ""} ({s.evidence})
                </li>
              ))}
            </ul>
          </div>
        )}
        <p className="procedure-note">{reply.note}</p>
      </div>
      {reply.steps.length > 0 && <Route steps={reply.steps} academy={academy} stopped={reply.outcome === "stopped"} />}
    </div>
  );
}

function Route({ steps, academy, stopped }: { steps: AgentStep[]; academy: Academy; stopped: boolean }) {
  return (
    <aside aria-label="מה פייסר בדק">
      <div className="route-title">מה נבדק בדרך</div>
      <ol className="route arriving">
        {steps.map((step, index) => (
          <li key={index} className={step.ok ? undefined : "failed"} style={{ animationDelay: `${index * 90}ms` }}>
            <span className="leg">{LEGS[step.tool] ?? step.tool}</span>
            {describe(step, academy) && <span className="detail">{describe(step, academy)}</span>}
            <span className="detail">{step.summary}</span>
          </li>
        ))}
        <li className="end" style={{ animationDelay: `${steps.length * 90}ms` }}>
          <span className="leg">{stopped ? "נעצר לפני תשובה" : "תשובה"}</span>
        </li>
      </ol>
    </aside>
  );
}

// What a step was about, as words: a trainee's name (if the asker may see them), a group, a query.
function describe(step: AgentStep, academy: Academy): string | null {
  const args = step.arguments;
  const parts: string[] = [];
  if (typeof args.trainee_id === "number") {
    const t = academy.trainees.get(args.trainee_id);
    parts.push(t ? `${t.first_name} ${t.last_name}` : `חניך ${args.trainee_id}`);
  }
  if (typeof args.group_id === "number") {
    parts.push(academy.groups.get(args.group_id)?.name ?? `קבוצה ${args.group_id}`);
  }
  for (const key of ["name", "query", "question"]) {
    if (typeof args[key] === "string") parts.push(`״${args[key] as string}״`);
  }
  if (typeof args.day === "string") parts.push(args.day as string);
  return parts.length ? parts.join(", ") : null;
}
