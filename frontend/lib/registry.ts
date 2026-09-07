/**
 * Registry types and TanStack Query hooks (Phase 4, Step 1).
 *
 * Both queries rely on Row Level Security to do the filtering rather than
 * adding `.eq()` clauses here. `provinces` returns only live jurisdictions
 * and `municipalities` only active ones, enforced in Postgres. A client-side
 * filter would be a second place for that rule to live, and the one that
 * silently stops matching when the policy changes.
 */

"use client";

import { useQuery } from "@tanstack/react-query";

import { isSupabaseConfigured, supabase } from "@/lib/supabase";

export type Province = {
  code: string;
  name: string;
  type: "province" | "territory";
};

export type Municipality = {
  id: string;
  name: string;
  province_code: string;
  source_bylaw_name: string;
  source_url: string;
  languages: string[];
  bylaw_last_verified_at: string | null;
};

export class RegistryUnavailableError extends Error {}

function requireClient() {
  if (!isSupabaseConfigured || !supabase) {
    throw new RegistryUnavailableError(
      "Supabase is not configured. Copy frontend/.env.local.example to " +
        ".env.local and set NEXT_PUBLIC_SUPABASE_URL and " +
        "NEXT_PUBLIC_SUPABASE_ANON_KEY.",
    );
  }
  return supabase;
}

export function useProvinces() {
  return useQuery({
    queryKey: ["provinces"],
    queryFn: async (): Promise<Province[]> => {
      const { data, error } = await requireClient()
        .from("provinces")
        .select("code,name,type")
        .order("name");

      if (error) throw new Error(error.message);
      return (data ?? []) as Province[];
    },
  });
}

export function useMunicipalities(provinceCode: string | null) {
  return useQuery({
    // Keyed by province so switching back to a previous one is instant.
    queryKey: ["municipalities", provinceCode],
    enabled: Boolean(provinceCode),
    queryFn: async (): Promise<Municipality[]> => {
      const { data, error } = await requireClient()
        .from("municipalities")
        .select(
          "id,name,province_code,source_bylaw_name,source_url,languages,bylaw_last_verified_at",
        )
        .eq("province_code", provinceCode as string)
        .order("name");

      if (error) throw new Error(error.message);
      return (data ?? []) as Municipality[];
    },
  });
}

/**
 * How a municipality's verification date should read in the UI.
 *
 * Null is the normal state right now, not an error: ingestion records when
 * a document was last FETCHED, and only a human confirming with municipal
 * planning staff sets this. Section 5 Rule 6 prints it to the public, so
 * an unverified corpus has to say so rather than show a fetch date dressed
 * up as a verification.
 */
export type VerificationStatus = {
  verified: boolean;
  label: string;
};

export function verificationStatus(
  municipality: Municipality | null,
): VerificationStatus | null {
  if (!municipality) return null;

  const date = municipality.bylaw_last_verified_at;
  return date
    ? { verified: true, label: `Bylaw text last verified ${date}` }
    : {
        verified: false,
        label:
          "Bylaw text has not yet been verified against this municipality's " +
          "current consolidation — it may be out of date.",
      };
}
