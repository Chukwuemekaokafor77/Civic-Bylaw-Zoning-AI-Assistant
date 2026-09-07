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

import { ExternalLink, FileText } from "lucide-react";

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
      className="rounded-lg border bg-muted/40 p-3"
    >
      <h3 className="mb-2 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        <FileText className="size-3.5" aria-hidden />
        Sections used for this answer ({citations.length})
      </h3>

      <ul className="space-y-1.5">
        {citations.map((citation) => (
          <li key={citation.chunk_id}>
            <a
              href={citation.source_url}
              target="_blank"
              // noreferrer alongside noopener: these are third-party
              // municipal sites, and there is no reason to hand them the
              // referring URL.
              rel="noopener noreferrer"
              className="group flex items-start gap-1.5 text-xs hover:underline"
            >
              <ExternalLink
                className="mt-0.5 size-3 shrink-0 text-muted-foreground"
                aria-hidden
              />
              <span>
                <span className="font-medium">
                  Section {citation.section_number}
                </span>
                {citation.section_title && (
                  <span className="text-muted-foreground">
                    {" "}
                    — {citation.section_title}
                  </span>
                )}
                {citation.page_number !== null && (
                  <span className="text-muted-foreground">
                    {" "}
                    (p. {citation.page_number})
                  </span>
                )}
                <span className="sr-only">{citationLabel(citation)}</span>
              </span>
            </a>
          </li>
        ))}
      </ul>

      <p className="mt-2 text-[11px] text-muted-foreground">
        Links open the municipality&apos;s official bylaw document.
      </p>
    </section>
  );
}

export default CitationCard;
