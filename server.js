// server.js (Node 18+, ESM)
import express from "express";
import { pipeline } from "@xenova/transformers";

// —— config ——————————————————————————————————————————
const PORT = process.env.PORT ? Number(process.env.PORT) : 8788;
const MODEL_ID = process.env.EMBED_MODEL_ID || "Xenova/bge-small-en-v1.5"; // fast 384-dim English
const ENABLE_CORS = true;               // set false if only server-to-server
const NORMALIZE = true;                 // L2-normalize vectors (recommended)
const USE_PREFIX_FOR_BGE = true;        // add "query:" / "passage:" prefixes for BGE models
// ——————————————————————————————————————————————————————

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

// Load model once (first call will warm it up)
const embedder = await pipeline("feature-extraction", MODEL_ID, { quantized: true });

// Try built-in pooling; fall back to manual mean pool if not available
function meanPoolTensor(t) {
  const [/*batch*/, tokens, dim] = t.dims; // [1, T, D]
  const data = t.data; // Float32Array length T*D
  const out = new Array(dim).fill(0);
  for (let i = 0; i < tokens * dim; i++) out[i % dim] += data[i];
  for (let d = 0; d < dim; d++) out[d] /= tokens;
  return out;
}

function withPrefix(str, kind) {
  // For BGE models, adding prefixes improves performance
  if (!USE_PREFIX_FOR_BGE || !/bge/i.test(MODEL_ID)) return str;
  if (kind === "query") return `query: ${str}`;
  return `passage: ${str}`; // default for documents
}

// Health
app.get("/health", async (_req, res) => {
  try {
    // single token warmcheck
    const out = await embedder("ok", { pooling: "mean", normalize: NORMALIZE }).catch(() => null);
    const dim = out?.dims?.at(-1) ?? out?.data?.length ?? null;
    res.json({ ok: true, model: MODEL_ID, dim, normalize: NORMALIZE });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e?.message || e) });
  }
});

// POST /embed  { input: string|string[], type?: 'query'|'passage', normalize?: boolean }
app.post("/embed", async (req, res) => {
  try {
    const input = req.body?.input;
    const kind = req.body?.type === "query" ? "query" : "passage";
    const normalize = req.body?.normalize ?? NORMALIZE;

    const arr = Array.isArray(input) ? input : [String(input ?? " ")];
    const texts = arr.map(t => {
      const s = String(t ?? "").trim();
      return withPrefix(s.length ? s : " ", kind);
    });

    const vectors = [];
    let dim = 0;

    for (const text of texts) {
      // Try built-in pooling/normalization (Transformers.js supports these options)
      let out = await embedder(text, { pooling: "mean", normalize }).catch(() => null);
      let vec;

      if (out?.data && out?.dims?.length >= 1) {
        // Built-in produced a single vector
        vec = Array.from(out.data, Number);
        dim = dim || vec.length;
      } else {
        // Fallback: raw token embeddings → manual mean
        const raw = await embedder(text);
        vec = meanPoolTensor(raw).map(Number);
        if (normalize) {
          let sum = 0; for (const v of vec) sum += v*v;
          const norm = Math.sqrt(sum) || 1;
          vec = vec.map(v => v / norm);
        }
        dim = dim || vec.length;
      }

      vectors.push(vec);
    }

    // Return both shapes: your own ('vectors') and OpenAI-compatible ('data')
    res.json({
      ok: true,
      model: MODEL_ID,
      dim,
      vectors,
      data: vectors.map(v => ({ embedding: v })), // OpenAI-compatible
    });
  } catch (e) {
    console.error("[/embed]", e);
    res.status(500).json({ ok: false, error: e?.message || String(e) });
  }
});

app.listen(PORT, () => console.log(`Local embeddings on :${PORT} (${MODEL_ID})`));
