import * as tf from "@tensorflow/tfjs-node";
import { fileURLToPath } from "url";
import fs from "fs";
import path from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CLASSES = ["posturando", "nao_posturando"];
const IMAGE_SIZE = 224;
const EPOCHS = 15;
const BATCH_SIZE = 16;

const preprocessImage = (buffer) =>
  tf.tidy(() =>
    tf.node
      .decodeImage(buffer, 3)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
      .toFloat()
      .div(255.0)
  );

const loadDataset = () => {
  const images = [];
  const labels = [];

  for (const [i, classe] of CLASSES.entries()) {
    const folderPath = path.join(__dirname, "dataset", classe);
    if (!fs.existsSync(folderPath)) {
      console.warn(`⚠️ Pasta não encontrada: ${folderPath}`);
      continue;
    }

    const files = fs
      .readdirSync(folderPath)
      .filter((f) => /\.(jpe?g|png)$/i.test(f));

    for (const file of files) {
      const buffer = fs.readFileSync(path.join(folderPath, file));
      const imageTensor = preprocessImage(buffer);
      images.push(imageTensor);
      labels.push(i);
    }
  }

  if (images.length === 0) {
    throw new Error("❌ Nenhuma imagem encontrada no dataset.");
  }

  const xs = tf.stack(images);
  const ys = tf.oneHot(tf.tensor1d(labels, "int32"), CLASSES.length);
  return { xs, ys };
};

const buildModel = () => {
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],
      filters: 32,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: CLASSES.length, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(0.0003),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
};

const treinarModelo = async () => {
  let xs, ys;
  try {
    const dataset = loadDataset();
    xs = dataset.xs;
    ys = dataset.ys;

    const model = buildModel();
    model.summary();

    console.log("🚀 Iniciando treinamento...");

    await model.fit(xs, ys, {
      epochs: EPOCHS,
      batchSize: BATCH_SIZE,
      validationSplit: 0.2,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          console.log(
            `🧠 Época ${epoch + 1} - loss: ${logs.loss.toFixed(
              4
            )} - acc: ${logs.acc?.toFixed(4)}`
          );
        },
      },
    });

    const savePath = "file://" + path.join(__dirname, "modelo_tfjs");
    await model.save(savePath);

    console.log("✅ Modelo salvo em 'modelo_tfjs/'");
  } catch (error) {
    console.error("❌ Erro durante o treinamento:", error);
  } finally {
    xs?.dispose();
    ys?.dispose();
  }
};

treinarModelo();
