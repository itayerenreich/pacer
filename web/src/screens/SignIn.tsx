import { useState, type FormEvent } from "react";

// Development identity: the same X-Staff-Id the API reads. Real sign-in comes before real users.
export function SignIn({ onSignIn, problem, busy }: { onSignIn: (id: number) => void; problem: string | null; busy: boolean }) {
  const [value, setValue] = useState("");

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const id = Number(value);
    if (Number.isInteger(id) && id > 0) onSignIn(id);
  };

  return (
    <div className="sign-in">
      <form onSubmit={submit}>
        <div className="mark">Pacer</div>
        <p>העוזר של צוות המכינה: חניכים, נוכחות, לו״ז, ומה עושים כשמשהו לא מסתדר.</p>
        <label htmlFor="staff-id">מספר איש צוות</label>
        <input
          id="staff-id"
          className="field"
          inputMode="numeric"
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value.replace(/\D/g, ""))}
        />
        <button className="button" type="submit" disabled={busy || !value}>
          {busy ? "בודק..." : "כניסה"}
        </button>
        {problem && <p className="error">{problem}</p>}
        <p className="quiet">
          זו כניסת פיתוח, בלי סיסמה. בנתוני הדמו: 1 היא המנהלת, 2 עד 5 הם המדריכים.
        </p>
      </form>
    </div>
  );
}
