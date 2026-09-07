"use client";

/**
 * Question input for the bylaw assistant.
 *
 * PHASE 1 SCOPE: structure only. The textarea and button render and are
 * deliberately disabled — no submit handler, no SSE connection, no
 * TanStack Query usage. Phase 4, Step 2 wires this to POST /stream.
 *
 * Design notes carried forward to Phase 4:
 *  - Responses stream as Server-Sent Events, so this cannot use a plain
 *    `useQuery`. TanStack Query tracks request state while tokens are
 *    appended to a local buffer as they arrive.
 *  - Submission must be blocked until a municipality is selected. Every
 *    retrieval path filters on municipality_id, so a query without one is
 *    not answerable rather than merely unfiltered.
 *  - Rate limiting (Phase 5) returns HTTP 429; the UI needs a distinct,
 *    non-alarming message for it, separate from a genuine error.
 */

import { Send } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export function ChatBox() {
  return (
    <form
      className="space-y-3"
      onSubmit={(event) => event.preventDefault()}
      aria-describedby="chatbox-status"
    >
      <Textarea
        id="question"
        name="question"
        rows={3}
        disabled
        placeholder="e.g. Can I build a secondary suite in an R1 zone?"
        className="resize-none"
      />

      <div className="flex items-center justify-between gap-3">
        <p id="chatbox-status" className="text-muted-foreground text-xs">
          Select a municipality to begin.
        </p>
        <Button type="submit" disabled>
          <Send className="size-4" aria-hidden />
          Ask
        </Button>
      </div>
    </form>
  );
}

export default ChatBox;
