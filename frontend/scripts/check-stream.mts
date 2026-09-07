/**
 * Exercises lib/stream.ts against (a) synthetic frames delivered on
 * deliberately awkward chunk boundaries, and (b) the live backend.
 *
 * Run with: npx tsx scripts/check-stream.mts
 *
 * This is a development check, not part of the build. The interesting case
 * is (a): network chunks split wherever TCP puts them, so a parser that
 * treats each chunk independently loses the token straddling the split.
 * That failure is invisible in normal use - the answer is simply missing a
 * word - so it is worth provoking on purpose.
 */

import { streamAnswer, type Citation } from "../lib/stream.js";

function encode(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

/** Serves `body` split at the given byte offsets. */
function mockFetch(body: string, splits: number[]) {
  const bytes = encode(body);
  const pieces: Uint8Array[] = [];
  let previous = 0;
  for (const at of [...splits, bytes.length]) {
    pieces.push(bytes.slice(previous, at));
    previous = at;
  }

  return async () =>
    new Response(
      new ReadableStream({
        start(controller) {
          for (const piece of pieces) controller.enqueue(piece);
          controller.close();
        },
      }),
      { status: 200, headers: { "Content-Type": "text/event-stream" } },
    );
}

/**
 * CRLF, because that is what sse-starlette actually emits. The first
 * version of this script used LF and passed against a parser that could
 * not read a single real frame - the synthetic fixture agreed with the
 * parser's assumption instead of with the server.
 */
function sse(lineEnding: "\r\n" | "\n"): string {
  const frames = [
    [
      "event: citations",
      'data: [{"chunk_id":"c1","municipality_name":"Fredericton",' +
        '"bylaw_name":"Zoning By-law Z-5","section_number":"8.14(2)",' +
        '"section_title":"Uses","page_number":167,' +
        '"source_url":"https://example.ca/z5.pdf","language":"en"}]',
    ],
    ["event: token", 'data: "Kennels "'],
    ["event: token", 'data: "are conditional."'],
    ["event: done", "data: null"],
  ];
  return frames
    .map((lines) => lines.join(lineEnding) + lineEnding + lineEnding)
    .join("");
}

const SSE = sse("\r\n");

async function collect(splits: number[], body: string = SSE) {
  const original = globalThis.fetch;
  globalThis.fetch = mockFetch(body, splits) as typeof fetch;

  let text = "";
  let citations: Citation[] = [];
  try {
    await streamAnswer(
      {
        query: "q",
        municipality_id: "nb_fredericton",
        province_code: "NB",
        language: "en",
      },
      {
        onToken: (t) => {
          text += t;
        },
        onCitations: (c) => {
          citations = c;
        },
      },
    );
  } finally {
    globalThis.fetch = original;
  }
  return { text, citations };
}

function assert(label: string, condition: boolean) {
  console.log(`${condition ? "PASS" : "FAIL"}  ${label}`);
  if (!condition) process.exitCode = 1;
}

const EXPECTED = "Kennels are conditional.";

// Whole body in one chunk.
const whole = await collect([]);
assert("single chunk: full answer", whole.text === EXPECTED);
assert("single chunk: citation parsed", whole.citations.length === 1);

// Split mid-frame, mid-JSON, and between the two token frames.
for (const at of [40, 120, 260, 300, 330, 360]) {
  const result = await collect([at]);
  assert(
    `split at byte ${at}: full answer, nothing dropped`,
    result.text === EXPECTED,
  );
}

// Split every 7 bytes — the pathological case.
const many = await collect(
  Array.from({ length: Math.floor(SSE.length / 7) }, (_, i) => (i + 1) * 7),
);
assert("split every 7 bytes: full answer", many.text === EXPECTED);
assert("split every 7 bytes: citation intact", many.citations.length === 1);

// LF-only servers must still parse, so the fix accepts both.
const lf = await collect([], sse("\n"));
assert("LF line endings: full answer", lf.text === EXPECTED);
assert("LF line endings: citation parsed", lf.citations.length === 1);

// Live backend, if it is running.
const base = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
try {
  await fetch(`${base}/health`);
} catch {
  console.log("SKIP  live backend not reachable");
  process.exit(process.exitCode ?? 0);
}

let liveText = "";
let liveCitations: Citation[] = [];
let liveError: string | null = null;
await streamAnswer(
  {
    query: "In the RR-CH zone, is a kennel permitted?",
    municipality_id: "nb_fredericton",
    province_code: "NB",
    language: "en",
  },
  {
    onToken: (t) => {
      liveText += t;
    },
    onCitations: (c) => {
      liveCitations = c;
    },
    // Without this, an error frame is indistinguishable from an empty
    // answer - the exact blind spot this script exists to close.
    onError: (message) => {
      liveError = message;
    },
  },
);

console.log("\n--- live answer ---");
console.log(liveError ? `ERROR EVENT: ${liveError}` : liveText);
assert("live: no error event", liveError === null);
assert("live: citations received", liveCitations.length > 0);
assert("live: answer non-empty", liveText.length > 50);
assert("live: ascii citation present", /\[Fredericton - .+?, Section .+?\]/.test(liveText));
assert("live: disclaimer present", liveText.includes("*Disclaimer:"));
assert(
  "live: citation links to the source document",
  liveCitations.every((c) => c.source_url.startsWith("http")),
);
