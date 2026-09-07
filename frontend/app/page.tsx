import { Landmark } from "lucide-react";

import { Assistant } from "@/components/Assistant";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/**
 * A React Server Component wrapping the client-side assistant. Selection
 * state and all data fetching live in <Assistant>, so this stays static
 * and streams to the browser without waiting on the registry.
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
          to the section they came from.
        </p>
      </header>

      <Alert>
        <AlertTitle>Early access — coverage is still limited</AlertTitle>
        <AlertDescription>
          Only municipalities listed below have had their bylaws indexed, and
          that text has not yet been verified against each municipality&apos;s
          current consolidation. Always confirm with local planning staff
          before relying on an answer.
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
          <Assistant />
        </CardContent>
      </Card>
    </main>
  );
}
