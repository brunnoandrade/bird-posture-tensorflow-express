import express from "express";
import multer from "multer";
import * as tf from "@tensorflow/tfjs-node";
import path from "path";
import { fileURLToPath } from "url";

// Simular __dirname em ES Modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;
const upload = multer({ storage: multer.memoryStorage() });

const CLASSES = ["posturando", "nao_posturando"]; // MESMA ordem do treino
const IMAGE_SIZE = 224;
const THRESHOLD = 0.8; // ajuste conforme necessário

let model;

// Carregamento do modelo
(async () => {
  try {
    console.log("⏳ Carregando modelo...");
    model = await tf.loadLayersModel(
      "file://" + path.join(__dirname, "modelo_tfjs", "model.json")
    );
    console.log("✅ Modelo carregado com sucesso!");
  } catch (error) {
    console.error("❌ Erro ao carregar modelo:", error);
  }
})();

// Rota de análise
app.post("/analisar", upload.single("imagem"), async (req, res) => {
  try {
    if (!model) {
      return res.status(503).json({ error: "Modelo ainda não carregado" });
    }

    if (!req.file) {
      return res.status(400).json({ error: "Nenhuma imagem enviada" });
    }

    console.log(`📷 Imagem recebida com tamanho: ${req.file.size} bytes`);

    const tensor = tf.node
      .decodeImage(req.file.buffer, 3)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
      .div(255.0)
      .expandDims(0); // [1, 224, 224, 3]

    const prediction = model.predict(tensor);
    const result = prediction.dataSync(); // Ex: [0.8, 0.2]
    const maxProb = Math.max(...result);
    const index = result.indexOf(maxProb);

    let label = "ndeterminado";

    if (maxProb >= THRESHOLD) {
      label = CLASSES[index];
    }

    console.log("📊 Probabilidades:", result);
    console.log("🏷️ Classificado como:", label);

    res.json({
      status: "ok",
      resultado: {
        prob: result.reduce((acc, val, i) => {
          acc[CLASSES[i]] = val;
          return acc;
        }, {}),
        label,
      },
    });

    tf.dispose([tensor, prediction]);
  } catch (err) {
    console.error("❌ Erro ao classificar imagem:", err);
    res.status(500).json({
      error: "Erro ao classificar imagem",
      detalhes: err.message,
    });
  }
});

// Inicialização do servidor
app.listen(port, () => {
  console.log(`🚀 Servidor rodando em http://localhost:${port}`);
});
