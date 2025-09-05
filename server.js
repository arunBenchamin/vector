import express from "express";
import { pipeline } from "@xenova/transformers";

const app = express();
app.use(express.json({ limit: "10mb" }));

// English-only model; change to "Xenova/bge-small-en-v1.5" or "Xenova/all-MiniLM-L6-v2"
const MODEL_ID = "Xenova/bge-small-en-v1.5";

// Load once (WASM/ONNX, CPU). First call warms up.
const embedder = await pipeline("feature-extraction", MODEL_ID, { quantized: true });

// Mean-pool last hidden state into a single vector
function meanPool(tensor) {
  const [_, tokens, dim] = tensor.dims;        // [1, T, D]
  const data = tensor.data;                    // Float32Array length T*D
  const out = new Array(dim).fill(0);
  for (let i = 0; i < tokens * dim; i++) out[i % dim] += data[i];
  for (let d = 0; d < dim; d++) out[d] /= tokens;
  return out;
}

// POST /embed  { input: string | string[] }
app.post("/embed", async (req, res) => {
  try {
    const input = req.body?.input;
    const arr = Array.isArray(input) ? input : [String(input ?? " ")];
    const vectors = [];
    for (const t of arr) {
      const out = await embedder(String(t || " "));
      vectors.push(meanPool(out));
    }
    return res.json({ ok: true, model: MODEL_ID, dim: vectors[0]?.length || 0, vectors });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ ok: false, error: e.message });
  }
});

app.listen(7071, () => console.log("Local embeddings on :7071"));
