"use client";

/**
 * Cascading Province -> Municipality selector (Phase 4, Step 1).
 *
 * Options come from the registry via the Supabase anon key. Row Level
 * Security already restricts this to live jurisdictions and active
 * municipalities, so Halifax and Charlottetown - both marked inactive
 * because their bylaw sources are unusable - never appear as choices.
 *
 * Language is a control, not a display preference. The Phase 0 bilingual
 * lock makes language part of the retrieval filter: a French question
 * must retrieve French chunks and cite the French document. It is shown
 * only for municipalities that actually publish in more than one
 * language, so unilingual ones are not given a pointless dropdown.
 */

import { AlertCircle, Globe, Loader2, MapPin } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  useMunicipalities,
  useProvinces,
  verificationLabel,
  type Municipality,
} from "@/lib/registry";

const LANGUAGE_LABELS: Record<string, string> = {
  en: "English",
  fr: "Français",
};

export type RegionalSelection = {
  provinceCode: string | null;
  municipality: Municipality | null;
  language: string;
};

type Props = {
  value: RegionalSelection;
  onChange: (next: RegionalSelection) => void;
};

function FieldError({ message }: { message: string }) {
  return (
    <p className="flex items-start gap-1.5 text-xs text-destructive">
      <AlertCircle className="mt-0.5 size-3 shrink-0" aria-hidden />
      <span>{message}</span>
    </p>
  );
}

export function RegionalSelector({ value, onChange }: Props) {
  // Held in a ref so the correction effect below does not depend on the
  // caller passing a stable onChange identity. Assigned in an effect
  // rather than during render, which React forbids.
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  const provinces = useProvinces();
  const municipalities = useMunicipalities(value.provinceCode);

  const available = municipalities.data ?? [];

  // Memoised because it is an effect dependency: rebuilding the array each
  // render would re-run the effect below on every render.
  const languages = useMemo(
    () => value.municipality?.languages ?? ["en"],
    [value.municipality],
  );

  // A municipality that does not publish in the selected language would
  // retrieve nothing at all, so fall back rather than leave the selector
  // in a state that can only produce the "not found" answer.
  const { municipality, language } = value;
  useEffect(() => {
    if (municipality && !languages.includes(language)) {
      onChangeRef.current({
        provinceCode: municipality.province_code,
        municipality,
        language: languages[0] ?? "en",
      });
    }
  }, [municipality, language, languages]);

  // base-ui's Select yields `string | null`; null means "cleared".
  function selectProvince(code: string | null) {
    if (!code) return;
    // Clearing the municipality is required, not tidiness: keeping a
    // Fredericton selection while the province reads "Nova Scotia" would
    // send a request whose province and municipality disagree.
    onChange({ provinceCode: code, municipality: null, language: "en" });
  }

  function selectMunicipality(id: string | null) {
    const chosen = id ? (available.find((m) => m.id === id) ?? null) : null;
    onChange({
      ...value,
      municipality: chosen,
      language: chosen?.languages?.[0] ?? "en",
    });
  }

  const showLanguage = languages.length > 1;

  return (
    <div className="space-y-2">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
        <div className="flex-1 space-y-1.5">
          <label
            htmlFor="province"
            className="flex items-center gap-1.5 text-sm font-medium"
          >
            <MapPin className="size-3.5" aria-hidden />
            Province or territory
          </label>
          <Select
            value={value.provinceCode}
            onValueChange={selectProvince}
            disabled={provinces.isLoading || Boolean(provinces.error)}
          >
            <SelectTrigger id="province" className="w-full">
              {/* Without a formatter this renders the raw value - "NB"
                  rather than "New Brunswick". */}
              <SelectValue>
                {(code: string | null) =>
                  (code &&
                    provinces.data?.find((p) => p.code === code)?.name) ||
                  (provinces.isLoading ? "Loading…" : "Select a province")
                }
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {(provinces.data ?? []).map((province) => (
                <SelectItem key={province.code} value={province.code}>
                  {province.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex-1 space-y-1.5">
          <label htmlFor="municipality" className="text-sm font-medium">
            Municipality
          </label>
          <Select
            value={value.municipality?.id ?? null}
            onValueChange={selectMunicipality}
            disabled={!value.provinceCode || municipalities.isLoading}
          >
            <SelectTrigger id="municipality" className="w-full">
              {/* "nb_fredericton" is a database key, not something to
                  show a resident. */}
              <SelectValue>
                {(id: string | null) =>
                  (id && available.find((m) => m.id === id)?.name) ||
                  (!value.provinceCode
                    ? "Select a province first"
                    : municipalities.isLoading
                      ? "Loading…"
                      : available.length === 0
                        ? "No municipalities available yet"
                        : "Select a municipality")
                }
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {available.map((municipality) => (
                <SelectItem key={municipality.id} value={municipality.id}>
                  {municipality.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {showLanguage && (
          <div className="space-y-1.5 sm:w-40">
            <label
              htmlFor="language"
              className="flex items-center gap-1.5 text-sm font-medium"
            >
              <Globe className="size-3.5" aria-hidden />
              Language
            </label>
            <Select
              value={value.language}
              onValueChange={(language) =>
                onChange({ ...value, language: language ?? "en" })
              }
            >
              <SelectTrigger id="language" className="w-full">
                <SelectValue>
                  {(code: string | null) =>
                    (code && LANGUAGE_LABELS[code]) || code || "English"
                  }
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {languages.map((code) => (
                  <SelectItem key={code} value={code}>
                    {LANGUAGE_LABELS[code] ?? code}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
      </div>

      {provinces.isLoading && (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Loader2 className="size-3 animate-spin" aria-hidden />
          Loading the municipality registry…
        </p>
      )}

      {provinces.error && (
        <FieldError message={(provinces.error as Error).message} />
      )}
      {municipalities.error && (
        <FieldError message={(municipalities.error as Error).message} />
      )}

      {value.provinceCode &&
        !municipalities.isLoading &&
        !municipalities.error &&
        available.length === 0 && (
          <p className="text-xs text-muted-foreground">
            No municipalities in this province have been indexed yet.
          </p>
        )}

      {/* Section 7 asks for the verification date to be visible, not buried
          in the answer. A reader deciding whether to trust a setback figure
          should see the corpus's status before they ask, not after. */}
      {value.municipality && (
        <p className="text-xs text-muted-foreground">
          {verificationLabel(value.municipality)}
        </p>
      )}
    </div>
  );
}

export default RegionalSelector;
