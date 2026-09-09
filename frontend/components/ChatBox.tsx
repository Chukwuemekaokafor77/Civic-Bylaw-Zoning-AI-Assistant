"use client";

/**
 * Question input and streamed answer (Phase 4, Step 2).
 *
 * TanStack Query tracks request state via useMutation while tokens are
 * appended to React state as they arrive. A plain useQuery cannot express
 * this: the answer is not a value that resolves once, it is a stream that
 * mutates a partial result many times before completing.
 *
 * Submission is blocked until a municipality is selected. Every retrieval
 * path filters on municipality_id, so a question without one is not
 * under-filtered, it is unanswerable.
 */

import { useMutation } from "@tanstack/react-query";
import { CornerDownLeft, Loader2, MessageSquareText, Send } from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { MessageList, type Message } from "@/components/MessageList";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { RegionalSelection } from "@/components/RegionalSelector";
import { StreamHttpError, streamAnswer, type Citation } from "@/lib/stream";

const MIN_QUERY_LENGTH = 3;
const MAX_QUERY_LENGTH = 1000;

/**
 * Shown before the first question.
 *
 * Not decoration: a blank box gives no sense of what this corpus can
 * answer, and a resident's first guess is often phrasing the bylaw does
 * not use. These are real topics with indexed sections behind them.
 */
const STARTERS = [
  "Can I build a garden suite?",
  "Can I run a business from my home?",
  "Am I allowed to put an apartment in my basement?",
  "How far from the property line does a swimming pool have to be?",
  "Can I keep backyard chickens?",
];

type Props = {
  selection: RegionalSelection;
};

export function ChatBox({ selection }: Props) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const municipality = selection.municipality;
  const ready = Boolean(municipality);

  const updateAnswer = useCallback((id: string, patch: Partial<Message>) => {
    setMessages((current) =>
      current.map((message) =>
        message.id === id ? { ...message, ...patch } : message,
      ),
    );
  }, []);

  const ask = useMutation({
    mutationFn: async (text: string) => {
      if (!municipality) throw new Error("Select a municipality first.");

      const answerId = `a-${Date.now()}`;
      setMessages((current) => [
        ...current,
        { id: `q-${Date.now()}`, role: "user", content: text },
        { id: answerId, role: "assistant", content: "", streaming: true },
      ]);

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      // Accumulated locally and flushed into state, so each token is not
      // a separate render of the whole transcript.
      let answer = "";
      let citations: Citation[] = [];

      try {
        await streamAnswer(
          {
            query: text,
            municipality_id: municipality.id,
            province_code: municipality.province_code,
            language: selection.language,
          },
          {
            onCitations: (received) => {
              citations = received;
              updateAnswer(answerId, { citations: received });
            },
            onToken: (token) => {
              answer += token;
              updateAnswer(answerId, { content: answer });
            },
            onError: (message) => {
              updateAnswer(answerId, { error: message, streaming: false });
            },
          },
          controller.signal,
        );

        updateAnswer(answerId, {
          content: answer,
          citations,
          streaming: false,
        });
      } catch (error) {
        if (controller.signal.aborted) {
          updateAnswer(answerId, { streaming: false });
          return;
        }

        const message =
          error instanceof StreamHttpError && error.isRateLimited
            ? "This assistant is rate limited to keep it free to run. Please wait a moment and ask again."
            : error instanceof Error
              ? error.message
              : "Something went wrong.";

        updateAnswer(answerId, { error: message, streaming: false });
      } finally {
        abortRef.current = null;
      }
    },
  });

  const trimmed = question.trim();
  const tooShort = trimmed.length > 0 && trimmed.length < MIN_QUERY_LENGTH;
  const canSubmit =
    ready && !ask.isPending && trimmed.length >= MIN_QUERY_LENGTH;

  function submit(text: string) {
    if (!ready || ask.isPending) return;
    if (text.trim().length < MIN_QUERY_LENGTH) return;
    ask.mutate(text.trim());
    setQuestion("");
  }

  function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    submit(question);
  }

  const showStarters = messages.length === 0;

  return (
    <section className="space-y-5">
      {messages.length > 0 && (
        <div className="rounded-2xl border border-border/70 bg-card p-5 shadow-sm sm:p-6">
          <MessageList messages={messages} />
        </div>
      )}

      <form
        className="overflow-hidden rounded-2xl border border-border/70 bg-card shadow-sm transition-shadow focus-within:border-primary/40 focus-within:shadow-md"
        onSubmit={onSubmit}
        aria-describedby="chatbox-status"
      >
        <Textarea
          id="question"
          name="question"
          rows={3}
          value={question}
          maxLength={MAX_QUERY_LENGTH}
          disabled={!ready || ask.isPending}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            // Enter sends; Shift+Enter is a newline. Zoning questions are
            // usually one sentence, so requiring a click would be friction.
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit(question);
            }
          }}
          placeholder={
            ready
              ? `Ask about zoning in ${municipality?.name}…`
              : "Select a municipality to begin."
          }
          className="min-h-0 resize-none rounded-none border-0 bg-transparent px-4 py-3.5 text-base shadow-none focus-visible:ring-0 md:text-sm"
        />

        <div className="flex items-center justify-between gap-3 border-t border-border/60 bg-muted/40 px-3 py-2.5">
          <p
            id="chatbox-status"
            className="flex items-center gap-1.5 text-xs text-muted-foreground"
          >
            {!ready ? (
              "Select a municipality to begin."
            ) : tooShort ? (
              `Ask at least ${MIN_QUERY_LENGTH} characters.`
            ) : ask.isPending ? (
              <>
                <Loader2 className="size-3 animate-spin" aria-hidden />
                Searching the bylaw…
              </>
            ) : (
              <>
                <CornerDownLeft className="size-3" aria-hidden />
                Enter to send · Shift + Enter for a new line
              </>
            )}
          </p>

          <Button type="submit" size="sm" disabled={!canSubmit}>
            {ask.isPending ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : (
              <Send className="size-4" aria-hidden />
            )}
            Ask
          </Button>
        </div>
      </form>

      {showStarters && (
        <div className="space-y-2.5">
          <p className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <MessageSquareText className="size-3.5" aria-hidden />
            Try one of these
          </p>
          <div className="flex flex-wrap gap-2">
            {STARTERS.map((starter) => (
              <button
                key={starter}
                type="button"
                disabled={!ready || ask.isPending}
                onClick={() => submit(starter)}
                className="rounded-full border border-border/70 bg-card px-3.5 py-1.5 text-xs text-foreground/80 shadow-xs transition-colors hover:border-primary/40 hover:bg-accent hover:text-accent-foreground disabled:cursor-not-allowed disabled:opacity-50"
              >
                {starter}
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export default ChatBox;
