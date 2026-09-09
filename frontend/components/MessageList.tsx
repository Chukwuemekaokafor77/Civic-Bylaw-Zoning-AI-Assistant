"use client";

/**
 * Transcript of questions and streamed answers (Phase 4, Step 2).
 *
 * Answers arrive as markdown from the model. They are rendered with a
 * deliberately small formatter rather than a markdown library: the output
 * is constrained by Section 5 Rule 5 to bullets, bold and plain
 * paragraphs, and a general markdown renderer would also interpret raw
 * HTML and links from model output that this application has no reason to
 * trust or display.
 */

import { AlertTriangle, Loader2, Sparkles, User } from "lucide-react";
import { Fragment } from "react";

import { CitationCard } from "@/components/CitationCard";
import type { Citation } from "@/lib/stream";

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  error?: string;
  streaming?: boolean;
};

/**
 * Renders **bold** spans, and sets inline citations apart from the prose.
 *
 * The citation is the load-bearing part of every sentence here, so it is
 * given its own treatment rather than being left to blend into the text
 * as another parenthetical.
 */
function renderInline(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|\[[^\]]+?, Section [^\]]+?\])/g);

  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={index} className="font-semibold">
          {part.slice(2, -2)}
        </strong>
      );
    }
    const citation = part.match(/^\[(.+), Section (.+)\]$/);
    if (citation) {
      const [, source, section] = citation;
      // Only the section number is shown. Rule 3 requires a citation on
      // every statement, so a long answer repeats the same municipality
      // and bylaw a dozen times - rendering all of it inline buries the
      // answer under its own footnotes. The full reference stays
      // available to assistive tech and on hover, and the card above
      // already names the document.
      return (
        <span
          key={index}
          title={`${source}, Section ${section}`}
          className="mx-0.5 rounded border border-primary/20 bg-primary/8 px-1 py-px align-baseline text-[0.78em] font-medium whitespace-nowrap text-primary"
        >
          <span aria-hidden>§ {section}</span>
          <span className="sr-only">{`${source}, Section ${section}`}</span>
        </span>
      );
    }
    return <Fragment key={index}>{part}</Fragment>;
  });
}

function AnswerBody({ content }: { content: string }) {
  const blocks: React.ReactNode[] = [];
  let bullets: string[] = [];

  const flushBullets = () => {
    if (bullets.length === 0) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="space-y-2">
        {bullets.map((item, index) => (
          <li key={index} className="flex gap-2.5">
            <span
              className="mt-2 size-1.5 shrink-0 rounded-full bg-primary/50"
              aria-hidden
            />
            <span className="flex-1">{renderInline(item)}</span>
          </li>
        ))}
      </ul>,
    );
    bullets = [];
  };

  for (const rawLine of content.split("\n")) {
    const line = rawLine.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);

    if (bullet) {
      bullets.push(bullet[1]);
      continue;
    }
    flushBullets();

    if (!line.trim()) continue;

    // The mandatory disclaimer arrives wrapped in single asterisks.
    const isDisclaimer = line.startsWith("*") && line.endsWith("*");
    blocks.push(
      <p
        key={`p-${blocks.length}`}
        className={
          isDisclaimer
            ? "mt-4 border-t border-border/70 pt-3 text-xs leading-relaxed text-muted-foreground italic"
            : undefined
        }
      >
        {isDisclaimer ? line.slice(1, -1) : renderInline(line)}
      </p>,
    );
  }
  flushBullets();

  return <div className="space-y-3.5 text-sm leading-relaxed">{blocks}</div>;
}

export function MessageList({ messages }: { messages: Message[] }) {
  if (messages.length === 0) return null;

  return (
    <div className="divide-y divide-border/60">
      {messages.map((message, index) =>
        message.role === "user" ? (
          <div
            key={message.id}
            className={`flex items-start gap-3 pb-4 ${index === 0 ? "" : "pt-6"}`}
          >
            <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-lg bg-secondary text-secondary-foreground">
              <User className="size-3.5" aria-hidden />
            </span>
            <p className="flex-1 pt-0.5 text-[0.95rem] font-medium text-balance">
              {message.content}
            </p>
          </div>
        ) : (
          <div key={message.id} className="flex items-start gap-3 py-5">
            <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary">
              <Sparkles className="size-3.5" aria-hidden />
            </span>

            <div className="min-w-0 flex-1 space-y-3.5">
              {message.error ? (
                <p className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2.5 text-sm text-destructive">
                  <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
                  <span>{message.error}</span>
                </p>
              ) : (
                <>
                  {/* Sources first: a reader who stops partway through has
                      still been shown which sections the answer rests on. */}
                  {message.citations && message.citations.length > 0 && (
                    <CitationCard citations={message.citations} />
                  )}

                  {message.content ? (
                    <div className="animate-rise">
                      <AnswerBody content={message.content} />
                    </div>
                  ) : (
                    message.streaming && (
                      <p className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="size-3.5 animate-spin" aria-hidden />
                        Reading the bylaw…
                      </p>
                    )
                  )}
                </>
              )}
            </div>
          </div>
        ),
      )}
    </div>
  );
}

export default MessageList;
