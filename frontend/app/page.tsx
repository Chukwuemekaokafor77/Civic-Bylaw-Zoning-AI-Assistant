import { Landmark } from "lucide-react";

import { ChatBox } from "@/components/ChatBox";
import { RegionalSelector } from "@/components/RegionalSelector";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/**
 * Phase 1 shell. A React Server Component that composes the two
 * placeholder client components. No data fetching yet — the registry query
 * and the streamed answer arrive in Phase 4.
 */
export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-4 py-10">
      <header className="space-y-2">
        <div className="flex items-center gap-2">
          <Landmark className="size-5" aria-hidden />
          <h1 className="text-2xl font-semibold tracking-tight text-balance">
            Canadian Civic Bylaw &amp; Zoning Assistant
          </h1>
        </div>
        <p className="text-muted-foreground text-sm text-pretty">
          Answers about land use, secondary suites, setbacks, and home
          businesses, drawn only from official municipal bylaws and cited back
          to the section they came from. Launching across Atlantic Canada and
          expanding nationwide.
        </p>
      </header>

      <Alert>
        <AlertTitle>Phase 1 — foundation only</AlertTitle>
        <AlertDescription>
          The interface below is a structural placeholder. Bylaw indexing
          (Phase 2) and question answering (Phases 3–4) are not yet wired up,
          so the controls are intentionally disabled.
        </AlertDescription>
      </Alert>

      <Card>
        <CardHeader>
          <CardTitle>Choose a municipality</CardTitle>
          <CardDescription>
            Bylaws differ between municipalities, so every answer is scoped to
            a single one. Nothing is ever mixed across municipalities or
            provinces.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <RegionalSelector />
          <ChatBox />
        </CardContent>
      </Card>
    </main>
  );
}
