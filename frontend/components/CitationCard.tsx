"use client";

/**
 * Source citations for one answer (Phase 4, Step 3).
 *
 * Every citation links to the official bylaw PDF the chunk came from. The
 * link is the point of the component: an answer a resident may act on has
 * to be checkable against the municipality's own document, not just
 * plausible-sounding. `source_url` is the citable URL recorded at
 * ingestion - for a bilingual municipality that is the document in the
 * chunk's own language, so a French citation never links to the English
 * PDF.
 *
 * Page numbers are shown when known because these documents run to
 * hundreds of pages; "Section 8.14(2)" alone is a long scroll.
 */

import { ArrowUpRight, FileText } from "lucide-react";

import type { Citation } from "@/lib/stream";

function citationLabel(citation: Citation): string {
  // Mirrors the Section 5 Rule 3 format the model is required to emit
  // inline, so the card and the prose read as the same reference.
  return `${citation.municipality_name} - ${citation.bylaw_name}, Section ${citation.section_number}`;
}

export function CitationCard({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;

  return (
    <section
      aria-label="Bylaw sections used for this answer"
      className="overflow-hidden rounded-xl border border-border/70 bg-muted/40"
    >
      <h3 className="flex items-center gap-1.5 border-b border-border/60 px-3 py-2 text-[0.7rem] font-semibold tracking-wide text-muted-foreground uppercase">
        <FileText className="size-3.5" aria-hidden />
        Sections used for this answer ({citations.length})
      </h3>

      <ul className="divide-y divide-border/50">
        {citations.map((citation) => (
          <li key={citation.chunk_id}>
            <a
              href={citation.source_url}
              target="_blank"
              // noreferrer alongside noopener: these are third-party
              // municipal sites, and there is no reason to hand them the
              // referring URL.
              rel="noopener noreferrer"
              className="group flex items-center gap-2.5 px-3 py-2.5 transition-colors hover:bg-card"
            >
              <span className="min-w-0 flex-1">
                <span className="text-xs font-semibold text-foreground">
                  Section {citation.section_number}
                </span>
                {citation.section_title && (
                  <span className="text-xs text-muted-foreground">
                    {" "}
                    · {citation.section_title}
                  </span>
                )}
                {citation.page_number !== null && (
                  <span className="text-xs text-muted-foreground">
                    {" "}
                    · p. {citation.page_number}
                  </span>
                )}
                <span className="sr-only">{citationLabel(citation)}</span>
              </span>

              <ArrowUpRight
                className="size-3.5 shrink-0 text-muted-foreground transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-primary"
                aria-hidden
              />
            </a>
          </li>
        ))}
      </ul>

      <p className="border-t border-border/60 px-3 py-2 text-[0.7rem] text-muted-foreground">
        Links open the municipality&apos;s official bylaw document.
      </p>
    </section>
  );
}

export default CitationCard;
