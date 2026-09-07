"use client";

/**
 * Cascading Province -> Municipality selector.
 *
 * PHASE 1 SCOPE: structure only. The dropdowns render and are deliberately
 * disabled — no registry fetch, no state, no selection handling. Phase 4,
 * Step 1 wires this to the municipalities registry.
 *
 * Design notes carried forward to Phase 4:
 *  - Options come from `provinces` and active `municipalities`, both
 *    readable with the Supabase anon key under RLS, so this needs no
 *    FastAPI round-trip.
 *  - `is_active = false` municipalities must never appear. Halifax and
 *    Charlottetown are currently inactive (see
 *    backend/scripts/municipalities_config.json for why).
 *  - A municipality whose `languages` includes "fr" needs a language
 *    control; the Phase 0 bilingual lock makes language part of the
 *    retrieval filter, not a display preference.
 */

import { MapPin } from "lucide-react";

import {
  Select,
  SelectContent,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const PROVINCES = [
  { code: "NB", name: "New Brunswick" },
  { code: "NS", name: "Nova Scotia" },
  { code: "PE", name: "Prince Edward Island" },
  { code: "NL", name: "Newfoundland and Labrador" },
] as const;

export function RegionalSelector() {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
      <div className="flex-1 space-y-1.5">
        <label
          htmlFor="province"
          className="flex items-center gap-1.5 text-sm font-medium"
        >
          <MapPin className="size-3.5" aria-hidden />
          Province
        </label>
        <Select disabled>
          <SelectTrigger id="province" className="w-full">
            <SelectValue placeholder={`Select a province (${PROVINCES.length})`} />
          </SelectTrigger>
          <SelectContent />
        </Select>
      </div>

      <div className="flex-1 space-y-1.5">
        <label htmlFor="municipality" className="text-sm font-medium">
          Municipality
        </label>
        <Select disabled>
          <SelectTrigger id="municipality" className="w-full">
            <SelectValue placeholder="Select a province first" />
          </SelectTrigger>
          <SelectContent />
        </Select>
      </div>
    </div>
  );
}

export default RegionalSelector;
