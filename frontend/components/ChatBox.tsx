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
import { Loader2, Send } from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { MessageList, type Message } from "@/components/MessageList";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { RegionalSelection } from "@/components/RegionalSelector";
import { StreamHttpError, streamAnswer, type Citation } from "@/lib/stream";

const MIN_QUERY_LENGTH = 3;
const MAX_QUERY_LENGTH = 1000;

type Props = {
  selection: RegionalSelection;
};

export function ChatBox({ selection }: Props) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const municipality = selection.municipality;
  const ready = Boolean(municipality);

  const updateAnswer = useCallback(
    (id: string, patch: Partial<Message>) => {
      setMessages((current) =>
        current.map((message) =>
          message.id === id ? { ...message, ...patch } : message,
        ),
      );
    },
    [],
  );

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

  function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;
    ask.mutate(trimmed);
    setQuestion("");
  }

  return (
    <div className="space-y-6">
      {messages.length > 0 && <MessageList messages={messages} />}

      <form
        className="space-y-3"
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
              onSubmit(event);
            }
          }}
          placeholder={
            ready
              ? `e.g. Can I build a garden suite in ${municipality?.name}?`
              : "Select a municipality to begin."
          }
          className="resize-none"
        />

        <div className="flex items-center justify-between gap-3">
          <p id="chatbox-status" className="text-xs text-muted-foreground">
            {!ready
              ? "Select a municipality to begin."
              : tooShort
                ? `Ask at least ${MIN_QUERY_LENGTH} characters.`
                : ask.isPending
                  ? "Searching the bylaw…"
                  : `Answers are drawn only from ${municipality?.name}'s indexed bylaws.`}
          </p>

          <Button type="submit" disabled={!canSubmit}>
            {ask.isPending ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : (
              <Send className="size-4" aria-hidden />
            )}
            Ask
          </Button>
        </div>
      </form>
    </div>
  );
}

export default ChatBox;
