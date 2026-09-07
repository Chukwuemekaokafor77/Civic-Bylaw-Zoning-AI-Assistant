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

import { AlertCircle, Loader2, User } from "lucide-react";
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

/** Renders **bold** spans; everything else is escaped by React. */
function renderInline(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={index} className="font-semibold">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <Fragment key={index}>{part}</Fragment>
    ),
  );
}

function AnswerBody({ content }: { content: string }) {
  const blocks: React.ReactNode[] = [];
  let bullets: string[] = [];

  const flushBullets = () => {
    if (bullets.length === 0) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="ml-4 list-disc space-y-1.5">
        {bullets.map((item, index) => (
          <li key={index}>{renderInline(item)}</li>
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
            ? "border-t pt-3 text-xs text-muted-foreground italic"
            : undefined
        }
      >
        {isDisclaimer ? line.slice(1, -1) : renderInline(line)}
      </p>,
    );
  }
  flushBullets();

  return <div className="space-y-3 text-sm leading-relaxed">{blocks}</div>;
}

export function MessageList({ messages }: { messages: Message[] }) {
  if (messages.length === 0) return null;

  return (
    <div className="space-y-6">
      {messages.map((message) =>
        message.role === "user" ? (
          <div key={message.id} className="flex items-start gap-2.5">
            <div className="mt-0.5 rounded-full bg-muted p-1.5">
              <User className="size-3.5" aria-hidden />
            </div>
            <p className="flex-1 pt-0.5 text-sm font-medium">
              {message.content}
            </p>
          </div>
        ) : (
          <div key={message.id} className="space-y-3 pl-9">
            {message.error ? (
              <p className="flex items-start gap-1.5 text-sm text-destructive">
                <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden />
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
                  <AnswerBody content={message.content} />
                ) : (
                  message.streaming && (
                    <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
                      <Loader2 className="size-3.5 animate-spin" aria-hidden />
                      Reading the bylaw…
                    </p>
                  )
                )}
              </>
            )}
          </div>
        ),
      )}
    </div>
  );
}

export default MessageList;
