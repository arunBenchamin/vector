// server.js — Local Embedding Service (OpenAI-compatible, padded to 1536)
// Node 18+, ESM ("type": "module" in package.json)
import express from "express";
import { pipeline } from "@xenova/transformers";

// ── Config ─────────────────────────────────────────────────────────────
const PORT = Number(process.env.PORT || 8788);
const MODEL_ID = process.env.EMBED_MODEL_ID || "Xenova/bge-small-en-v1.5"; // 384-dim English
const ENABLE_CORS = true;
const NORMALIZE = true;                   // L2-normalize vectors (recommended for cosine)
const TARGET_DIM = Number(process.env.TARGET_DIM || 1536); // pad/truncate to DB size
const RESPONSE_MODEL_LABEL = process.env.RESPONSE_MODEL || "text-embedding-3-small";
// ───────────────────────────────────────────────────────────────────────

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

// Load model once (first call warms up)
const embedder = await pipeline("feature-extraction", MODEL_ID, { quantized: true });

// Mean-pool last hidden state → single vector, also return token count
function meanPoolTensor(t) {
  const [/*batch*/, tokens, dim] = t.dims; // [1, T, D]
  const data = t.data;                      // Float32Array length T*D
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
function adjustDim(vec, target = TARGET_DIM) {
  if (vec.length === target) return vec;
  if (vec.length < target) return vec.concat(Array(target - vec.length).fill(0));
  return vec.slice(0, target);
}

// Health
app.get("/health", async (_req, res) => {
  try {
    const raw = await embedder("ok");
    const { vec } = meanPoolTensor(raw);
    res.json({ ok: true, model: MODEL_ID, base_dim: vec.length, target_dim: TARGET_DIM });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e?.message || e) });
  }
});

/**
 * POST /embed
 * Body: { input: string | string[], model?: string }
 * Returns the OpenAI-compatible shape, with embeddings padded/truncated to TARGET_DIM.
 */
app.post("/embed", async (req, res) => {
  try {
    const input = req.body?.input;
    const responseModel = req.body?.model || RESPONSE_MODEL_LABEL;

    // Normalize to array; avoid empty strings
    const items = Array.isArray(input) ? input : [String(input ?? " ")];
    const texts = items.map(t => {
      const s = String(t ?? "").trim();
      return s.length ? s : " ";
    });

    const data = [];
    let totalTokens = 0;

    for (let i = 0; i < texts.length; i++) {
      const raw = await embedder(texts[i]);      // [1, T, D]
      const { vec, tokens } = meanPoolTensor(raw);
      let emb = NORMALIZE ? l2norm(vec) : vec;   // normalize (optional)
      emb = adjustDim(emb, TARGET_DIM);          // pad/truncate to 1536 (or env)
      totalTokens += tokens;

      data.push({
        object: "embedding",
        index: i,
        embedding: emb.map(Number),
      });
    }

    // Exact OpenAI shape
    return res.json({
      object: "list",
      data,
      model: responseModel,                      // label only; client-friendly
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
      model: req.body?.model || RESPONSE_MODEL_LABEL,
      usage: { prompt_tokens: 0, total_tokens: 0 },
      error: String(e?.message || e),
    });
  }
});

app.listen(PORT, () =>
  console.log(`Local embeddings on :${PORT} (model=${MODEL_ID}, target_dim=${TARGET_DIM}, normalize=${NORMALIZE})`)
);
