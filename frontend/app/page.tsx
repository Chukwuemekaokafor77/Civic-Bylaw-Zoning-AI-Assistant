import { Landmark, ShieldCheck } from "lucide-react";

import { Assistant } from "@/components/Assistant";

/**
 * A React Server Component wrapping the client-side assistant. Selection
 * state and all data fetching live in <Assistant>, so this stays static
 * and streams to the browser without waiting on the registry.
 */
export default function Home() {
  return (
    <div className="app-surface flex min-h-full flex-1 flex-col">
      <header className="border-b border-border/60 bg-background/70 backdrop-blur-sm">
        <div className="mx-auto flex w-full max-w-4xl items-center justify-between gap-4 px-5 py-4">
          <div className="flex items-center gap-2.5">
            <span className="grid size-9 place-items-center rounded-xl bg-primary text-primary-foreground shadow-sm">
              <Landmark className="size-4.5" aria-hidden />
            </span>
            <div className="leading-tight">
              <p className="text-sm font-semibold tracking-tight">
                Civic Bylaw &amp; Zoning Assistant
              </p>
              <p className="text-xs text-muted-foreground">Canadian municipalities</p>
            </div>
          </div>

          {/* States the sourcing rule up front. It is the whole premise of
              the tool, and a reader who sees it before asking knows what
              kind of answer to expect. */}
          <span className="hidden items-center gap-1.5 rounded-full border border-border/70 bg-card px-3 py-1.5 text-xs text-muted-foreground shadow-xs sm:flex">
            <ShieldCheck className="size-3.5 text-primary" aria-hidden />
            Every answer cites its bylaw section
          </span>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-8 px-5 pt-10 pb-16">
        <div className="max-w-2xl space-y-3">
          <h1 className="text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
            What does your zoning bylaw actually say?
          </h1>
          <p className="text-pretty text-muted-foreground">
            Answers about land use, secondary suites, setbacks and home
            businesses — drawn only from official municipal bylaws, and cited
            back to the section they came from.
          </p>
        </div>

        <Assistant />
      </main>

      <footer className="border-t border-border/60 py-6">
        <p className="mx-auto max-w-4xl px-5 text-xs text-muted-foreground text-balance">
          This is an informational tool, not legal or planning advice. Bylaws
          are amended regularly — always confirm with your municipality&apos;s
          planning department before relying on an answer.
        </p>
      </footer>
    </div>
  );
}
