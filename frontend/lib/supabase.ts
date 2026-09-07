/**
 * Browser Supabase client, anon key only.
 *
 * The registry (`provinces`, `municipalities`) is read directly rather than
 * through FastAPI because Row Level Security already scopes exactly what
 * the browser may see: `provinces` exposes only live jurisdictions, and
 * `municipalities` only active ones (PART 7 of backend/db/001_init_schema.sql).
 * A backend proxy would re-implement that filter in a second place.
 *
 * This must never be given the service-role key. NEXT_PUBLIC_ variables are
 * inlined into the browser bundle, and the service-role key bypasses RLS -
 * it would expose every table, including the query_log of user questions.
 */

import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

/** Whether the registry can be read at all. */
export const isSupabaseConfigured = Boolean(url && anonKey);

/**
 * Null when unconfigured rather than throwing at import time, so a missing
 * .env.local surfaces as a readable message in the selector instead of a
 * blank page from a module that failed to evaluate.
 */
export const supabase = isSupabaseConfigured
  ? createClient(url as string, anonKey as string, {
      auth: { persistSession: false },
    })
  : null;
