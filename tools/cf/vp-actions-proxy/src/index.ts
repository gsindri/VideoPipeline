/**
 * vp-actions-proxy — Cloudflare Worker
 *
 * Provides a **stable** base URL for ChatGPT Actions while the Quick
 * Tunnel hostname rotates on every launch.
 *
 * Architecture:
 *   ChatGPT Action
 *     → https://vp-actions-proxy.<you>.workers.dev  (stable)
 *     → https://<random>.trycloudflare.com          (KV "base")
 *     → http://127.0.0.1:<port>                     (Studio)
 *
 * The upstream Quick Tunnel URL is stored in Workers KV (key: "base")
 * and is automatically updated by studio_quick_tunnel.ps1.
 */

export interface Env {
  /** KV namespace that stores the upstream Quick Tunnel base URL. */
  UPSTREAM: KVNamespace;
  /** Bearer token for authenticating to Studio (set via `wrangler secret put`). */
  VP_API_TOKEN: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** CORS preflight response (permissive — the real auth is the bearer token). */
function corsPreflightResponse(request: Request): Response {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
      "Access-Control-Allow-Headers":
        request.headers.get("Access-Control-Request-Headers") || "*",
      "Access-Control-Max-Age": "86400",
    },
  });
}

/** Add permissive CORS headers to an outgoing response. */
function addCorsHeaders(headers: Headers): void {
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Access-Control-Expose-Headers", "*");
}

/**
 * Rewrite the OpenAPI document so `servers` points at *this* Worker and
 * patch the two "empty-object" schemas that trip up the Actions importer.
 */
function patchOpenApiDoc(doc: Record<string, any>, workerOrigin: string): void {
  doc.servers = [{ url: workerOrigin }];

  const schemas = doc?.components?.schemas;
  if (!schemas) return;

  // The Actions importer rejects schemas with `type: "object"` but no
  // `properties` key.  Add an empty one so validation passes.
  for (const name of ["ResultsSummaryResponse", "DiagnosticsResponse"]) {
    if (
      schemas[name]?.type === "object" &&
      schemas[name].properties == null
    ) {
      schemas[name].properties = {};
    }
  }
}

// ---------------------------------------------------------------------------
// Fetch handler
// ---------------------------------------------------------------------------

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // ── CORS preflight ────────────────────────────────────────────────
    if (request.method === "OPTIONS") {
      return corsPreflightResponse(request);
    }

    // ── Read upstream base from KV ────────────────────────────────────
    const upstream = (await env.UPSTREAM.get("base"))?.trim();
    if (!upstream) {
      return new Response(
        "Upstream not configured. Run studio_quick_tunnel.ps1 to set KV 'base'.",
        { status: 503 },
      );
    }
    if (!upstream.startsWith("https://")) {
      return new Response("Invalid upstream base URL in KV.", { status: 500 });
    }

    const url = new URL(request.url);

    // ── Special case: /api/actions/openapi.json ───────────────────────
    //    Fetch from upstream using the secret bearer token, then rewrite
    //    servers.url to this Worker's origin so the Action keeps working.
    if (url.pathname === "/api/actions/openapi.json") {
      const target = new URL(upstream);
      target.pathname = "/api/actions/openapi.json";
      target.search = "";

      const resp = await fetch(target.toString(), {
        headers: { Authorization: `Bearer ${env.VP_API_TOKEN}` },
      });

      const text = await resp.text();
      if (!resp.ok) {
        return new Response(text || resp.statusText, { status: resp.status });
      }

      const doc = JSON.parse(text);
      patchOpenApiDoc(doc, url.origin);

      return new Response(JSON.stringify(doc), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "no-store",
        },
      });
    }

    // ── Generic proxy for everything else ─────────────────────────────
    const target = new URL(upstream);
    target.pathname = url.pathname;
    target.search = url.search;

    // The incoming Request object is immutable; build a mutable copy of
    // the headers and drop `host` so the upstream sees its own hostname.
    const headers = new Headers(request.headers);
    headers.delete("host");

    const init: RequestInit = {
      method: request.method,
      headers,
      redirect: "manual",
    };

    // Forward body for non-GET / non-HEAD.
    if (request.method !== "GET" && request.method !== "HEAD") {
      init.body = request.body;
    }

    const upstreamResp = await fetch(target.toString(), init);

    const outHeaders = new Headers(upstreamResp.headers);
    addCorsHeaders(outHeaders);

    return new Response(upstreamResp.body, {
      status: upstreamResp.status,
      headers: outHeaders,
    });
  },
} satisfies ExportedHandler<Env>;
