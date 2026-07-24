import { useState, type FormEvent } from "react";
import { useAuth } from "../auth";

export function LoginForm() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(username, password);
      } else {
        await register(username, password);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="h-screen bg-white text-neutral-900 flex items-center justify-center">
      <form
        onSubmit={handleSubmit}
        className="w-80 flex flex-col gap-4 border border-neutral-200 rounded-lg p-6"
      >
        <div>
          <h1 className="text-lg font-semibold">Enterprise Knowledge Assistant</h1>
          <p className="text-xs text-neutral-500 mt-1">
            {mode === "login" ? "Sign in to continue." : "The first account created becomes admin."}
          </p>
        </div>

        <label className="flex flex-col gap-1 text-sm text-neutral-600">
          Username
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            minLength={3}
            className="border border-neutral-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-neutral-500"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm text-neutral-600">
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={8}
            className="border border-neutral-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-neutral-500"
          />
        </label>

        {error && <p className="text-red-700 text-xs">{error}</p>}

        <button
          type="submit"
          disabled={submitting}
          className="bg-neutral-900 text-white text-sm rounded-md py-2 hover:bg-neutral-800 transition-colors disabled:opacity-50"
        >
          {submitting ? "..." : mode === "login" ? "Sign in" : "Create account"}
        </button>

        <button
          type="button"
          onClick={() => {
            setMode(mode === "login" ? "register" : "login");
            setError(null);
          }}
          className="text-xs text-neutral-500 hover:text-neutral-900"
        >
          {mode === "login" ? "Need an account? Register" : "Already have an account? Sign in"}
        </button>
      </form>
    </div>
  );
}
