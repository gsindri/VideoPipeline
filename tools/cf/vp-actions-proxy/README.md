# vp-actions-proxy — Stable URL for ChatGPT Actions

A Cloudflare Worker that gives your ChatGPT Action a **permanent** base URL
while Quick Tunnel hostnames keep rotating.

```
ChatGPT Action
  → https://vp-actions-proxy.<you>.workers.dev   (stable, never changes)
  → https://<random>.trycloudflare.com            (KV "base", auto-updated)
  → http://127.0.0.1:<port>                       (Studio on your machine)
```

---

## One-time setup

### 1. Install dependencies

```powershell
cd tools/cf/vp-actions-proxy
npm install
```

### 2. Authenticate with Cloudflare

```bash
npx wrangler login
```

### 3. Create the KV namespace

```bash
npx wrangler kv namespace create VP_UPSTREAM
```

Copy the printed `id` into `wrangler.toml`:

```toml
[[kv_namespaces]]
binding = "UPSTREAM"
id = "<paste-id-here>"
```

### 4. Store your Studio bearer token as a Worker secret

```bash
npx wrangler secret put VP_API_TOKEN
```

Paste your `VP_API_TOKEN` (the same 64-hex-char token Studio uses).

### 5. Deploy

```bash
npx wrangler deploy
```

You'll get a stable URL like:

```
https://vp-actions-proxy.<your-subdomain>.workers.dev
```

### 6. Set environment variables for auto-KV-update

The tunnel script (`studio_quick_tunnel.ps1`) will automatically push the
latest Quick Tunnel URL into KV — but it needs three env vars.

| Variable | Where to find it |
|---|---|
| `CF_ACCOUNT_ID` | Cloudflare dashboard → Workers & Pages → Account details |
| `CF_KV_NAMESPACE_ID` | The `id` from step 3 |
| `CF_API_TOKEN` | Create a token at dash.cloudflare.com/profile/api-tokens with **Workers KV Storage: Edit** permission |

Set them permanently (reopen terminals afterward):

```powershell
setx CF_ACCOUNT_ID     "your-account-id"
setx CF_KV_NAMESPACE_ID "your-namespace-id"
setx CF_API_TOKEN       "your-cloudflare-api-token"
```

### 7. Import the schema into your GPT Action (once)

In the GPT builder, import from:

```
https://vp-actions-proxy.<you>.workers.dev/api/actions/openapi.json
```

No `?token=` needed — the Worker fetches the upstream schema using its secret.

Set **Authentication** → **API Key** → **Bearer** → paste your `VP_API_TOKEN`.

---

## Daily workflow

Just run `studio_quick_tunnel.ps1` as usual. It will:

1. Start Studio (if `-StartStudio`)
2. Start the Quick Tunnel
3. **Automatically** update Workers KV with the new tunnel URL

Your GPT Action keeps working — no re-imports, no editing.

---

## Commands

| Command | Description |
|---|---|
| `npm run dev` | Local dev server (for testing the Worker itself) |
| `npm run deploy` | Deploy to Cloudflare |
| `npm run tail` | Live-tail Worker logs |
