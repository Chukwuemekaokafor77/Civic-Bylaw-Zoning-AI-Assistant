"use client";

/**
 * Client shell owning the current selection (Phase 4, Step 1).
 *
 * The province, municipality and language chosen here become the retrieval
 * filter for every request, so they live in one place rather than being
 * re-derived by each component. Step 2 hands this selection to ChatBox.
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
    <div className="space-y-6">
      <RegionalSelector value={selection} onChange={setSelection} />
      <ChatBox />
    </div>
  );
}

export default Assistant;
