"use client";

/**
 * Client shell owning the current selection (Phase 4).
 *
 * The province, municipality and language chosen here become the retrieval
 * filter for every request, so they live in one place rather than being
 * re-derived by each component.
 */

import { useState } from "react";

import { ChatBox } from "@/components/ChatBox";
import {
  RegionalSelector,
  type RegionalSelection,
} from "@/components/RegionalSelector";

const EMPTY_SELECTION: RegionalSelection = {
  provinceCode: null,
  municipality: null,
  language: "en",
};

export function Assistant() {
  const [selection, setSelection] = useState<RegionalSelection>(EMPTY_SELECTION);

  return (
    <div className="space-y-5">
      <section className="rounded-2xl border border-border/70 bg-card p-5 shadow-sm sm:p-6">
        <div className="mb-5 space-y-1">
          <h2 className="text-base font-semibold tracking-tight">
            Choose a municipality
          </h2>
          <p className="text-pretty text-sm text-muted-foreground">
            Bylaws differ between municipalities, so every answer is scoped to
            one. Nothing is ever mixed across municipalities or provinces.
          </p>
        </div>

        <RegionalSelector value={selection} onChange={setSelection} />
      </section>

      <ChatBox selection={selection} />
    </div>
  );
}

export default Assistant;
