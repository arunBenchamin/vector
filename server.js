// server.js — Local Embedding Service with OpenAI-compatible response
// Node 18+, ESM ("type": "module" in package.json)
import express from "express";
import { pipeline } from "@xenova/transformers";

// ── Config ─────────────────────────────────────────────────────────────
const PORT = Number(process.env.PORT || 8788);
const MODEL_ID = process.env.EMBED_MODEL_ID || "Xenova/bge-small-en-v1.5"; // 384-dim, fast English
const ENABLE_CORS = true;
const NORMALIZE = true; // L2 normalize vectors (recommended for cosine)

// ── App setup ──────────────────────────────────────────────────────────
const app = express();
app.use(express.json({ limit: "16mb" }));
if (ENABLE_CORS) {
  app.use((req, res, next) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
    if (req.method === "OPTIONS") return res.sendStatus(204);
    next();
  });
}

// ── Load model once (first call warms up) ──────────────────────────────
const embedder = await pipeline("feature-extraction", MODEL_ID, { quantized: true });

// Mean-pool last hidden state → single vector
function meanPoolTensor(t) {
  const [/*batch*/, tokens, dim] = t.dims; // [1, T, D]
  const data = t.data; // Float32Array length T*D
  const out = new Array(dim).fill(0);
  for (let i = 0; i < tokens * dim; i++) out[i % dim] += data[i];
  for (let d = 0; d < dim; d++) out[d] /= tokens;
  return { vec: out, tokens };
}
function l2norm(v) {
  let s = 0; for (const x of v) s += x * x;
  const n = Math.sqrt(s) || 1;
  return v.map(x => x / n);
}

// Health
app.get("/health", async (_req, res) => {
  try {
    const t = await embedder("ok"); // no pooling → [1,T,D]
    const { vec } = meanPoolTensor(t);
    res.json({ ok: true, model: MODEL_ID, dim: vec.length });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e?.message || e) });
  }
});

/**
 * POST /embed
 * Body:
 *   { input: string | string[],
 *     model?: string }  // returned verbatim in response for compatibility
 *
 * Response (OpenAI-compatible):
 * {
 *   "object":"list",
 *   "data":[{"object":"embedding","index":0,"embedding":[...]}],
 *   "model":"text-embedding-3-small",
 *   "usage":{"prompt_tokens":N,"total_tokens":N}
 * }
 */
app.post("/embed", async (req, res) => {
  try {
    const input = req.body?.input;
    const responseModel = req.body?.model || "text-embedding-3-small"; // just a label for clients

    // Normalize to array and avoid empty strings
    const items = Array.isArray(input) ? input : [String(input ?? " ")];
    const texts = items.map(t => {
      const s = String(t ?? "").trim();
      return s.length ? s : " ";
    });

    const data = [];
    let totalTokens = 0;

    for (let i = 0; i < texts.length; i++) {
      const raw = await embedder(texts[i]); // no pooling → access token count
      const { vec, tokens } = meanPoolTensor(raw);
      const emb = NORMALIZE ? l2norm(vec) : vec;
      totalTokens += tokens;

      data.push({
        object: "embedding",
        index: i,
        embedding: emb.map(Number),
      });
    }

    // OpenAI-compatible shape
    return res.json({
      object: "list",
      data,
      model: responseModel,
      usage: {
        prompt_tokens: totalTokens,
        total_tokens: totalTokens,
      },
    });
  } catch (e) {
    console.error("[/embed] error:", e);
    return res.status(500).json({
      object: "list",
      data: [],
      model: req.body?.model || "text-embedding-3-small",
      usage: { prompt_tokens: 0, total_tokens: 0 },
      error: String(e?.message || e),
    });
  }
});

app.listen(PORT, () =>
  console.log(`Local embeddings on :${PORT}  (model=${MODEL_ID}, normalize=${NORMALIZE})`)
);
