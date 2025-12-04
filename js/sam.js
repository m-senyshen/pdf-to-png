// js/sam.js
// Exports: initModel(modelUrl?) -> Promise, segmentBoxOnCanvas(canvas, box) -> Promise<{mask, width, height}>
// Fallback: color-threshold mask if model not present or fails

let ortSession = null;
let modelReady = false;
let modelInfo = null;

/**
 * initModel - optionally load ONNX MobileSAM model
 * @param {string} modelUrl - path to your mobile_sam.onnx (optional)
 */
export async function initModel(modelUrl) {
  if (!modelUrl) {
    console.log('sam.js: No model URL provided — running in fallback mode.');
    modelReady = false;
    return;
  }
  // load ONNX runtime (assumes ort is already loaded globally: <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>)
  if (!window.ort) {
    console.warn('sam.js: onnxruntime-web (ort) not found. Please include ort.min.js to enable ONNX model. Falling back to color-threshold.');
    modelReady = false;
    return;
  }
  try {
    console.log('sam.js: Loading ONNX model from', modelUrl);
    ortSession = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    modelReady = true;
    console.log('sam.js: ONNX model loaded (note: you may need to adjust preprocessing based on model specifics).');
  } catch (err) {
    console.error('sam.js: Failed to load ONNX model — falling back to color-threshold', err);
    ortSession = null;
    modelReady = false;
  }
}

/**
 * segmentBoxOnCanvas - primary function: produce a binary mask for the bounding box.
 * @param {HTMLCanvasElement} sourceCanvas - canvas containing the image rendered (full-resolution for the preview).
 * @param {{x:number,y:number,w:number,h:number}} box - bounding box in canvas coordinates
 * @returns {Promise<{mask:Uint8ClampedArray, width:number, height:number}>}
 */
export async function segmentBoxOnCanvas(sourceCanvas, box) {
  // crop the box area onto a temporary canvas
  const sx = Math.max(0, Math.floor(box.x));
  const sy = Math.max(0, Math.floor(box.y));
  const sw = Math.max(1, Math.floor(box.w));
  const sh = Math.max(1, Math.floor(box.h));

  const crop = document.createElement('canvas');
  crop.width = sw;
  crop.height = sh;
  const ctx = crop.getContext('2d');
  ctx.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, sw, sh);

  // If model present, try to run it
  if (modelReady && ortSession) {
    try {
      // NOTE: Many ONNX MobileSAM builds expect a fixed input size (e.g., 1024). We attempt a generic pipeline:
      // - resize to 1024 (or to model size if known)
      // - normalize to [0,1] and convert to float32
      // The exact input/output names may differ; you might need to update them here.

      const MODEL_SIZE = 1024; // common mobile sam size; adjust if your model uses a different resolution
      // resize to MODEL_SIZE while preserving aspect by stretching (SAM variants often expect square)
      const tmp = document.createElement('canvas');
      tmp.width = MODEL_SIZE;
      tmp.height = MODEL_SIZE;
      const tctx = tmp.getContext('2d');
      tctx.drawImage(crop, 0, 0, MODEL_SIZE, MODEL_SIZE);
      const imgData = tctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
      // create Float32 tensor [1, 3, H, W]
      const floatData = new Float32Array(MODEL_SIZE * MODEL_SIZE * 3);
      for (let i = 0, p = 0; i < imgData.data.length; i += 4, p++) {
        floatData[p * 3 + 0] = imgData.data[i] / 255.0;
        floatData[p * 3 + 1] = imgData.data[i + 1] / 255.0;
        floatData[p * 3 + 2] = imgData.data[i + 2] / 255.0;
      }
      // NOTE: The exact input name will vary by model; try common names
      const inputTensor = new ort.Tensor('float32', floatData, [1, 3, MODEL_SIZE, MODEL_SIZE]);

      // try multiple common input names - the model may expect 'input', 'image', 'images', etc.
      const inputNamesToTry = ortSession.inputNames || ['input', 'images', 'image', 'input_image'];
      let feeds = {};
      let usedName = null;
      for (const name of inputNamesToTry) {
        try {
          feeds[name] = inputTensor;
          // run (some models might require additional inputs for prompts; we try a simple image-only run)
          const results = await ortSession.run(feeds);
          // pick the first output as candidate mask (model-specific)
          const outName = Object.keys(results)[0];
          const outTensor = results[outName]; // may be float32 logits or probabilities
          // Convert outTensor to a binary mask scaled to tmp resolution (assume outTensor dims match)
          let outData = outTensor.data;
          let ow = outTensor.dims[outTensor.dims.length - 1];
          let oh = outTensor.dims[outTensor.dims.length - 2];
          // reshape & threshold
          const mask = new Uint8ClampedArray(ow * oh);
          for (let i = 0; i < ow * oh; i++) {
            const val = outData[i]; // if logits, may be >1; threshold at 0.5
            mask[i] = val > 0.5 ? 255 : 0;
          }
          // If mask resolution differs from cropped area, resample mask to (sw, sh)
          const finalMask = await resampleMaskTo(mask, ow, oh, sw, sh);
          return { mask: finalMask, width: sw, height: sh };
        } catch (runErr) {
          // clear feeds and try next input name
          feeds = {};
          continue;
        }
      }
      // If we reach here, model attempts failed — fall back
      console.warn('sam.js: ONNX model ran but outputs did not match expected shapes or names; falling back to color-threshold.');
    } catch (err) {
      console.error('sam.js: ONNX model inference failed, falling back to color mask', err);
    }
  }

  // Fallback color-threshold segmentation (blue-ish detection) - reliable test mode
  const cropData = ctx.getImageData(0, 0, sw, sh);
  const mask = new Uint8ClampedArray(sw * sh);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const idx = (y * sw + x) * 4;
      const r = cropData.data[idx];
      const g = cropData.data[idx + 1];
      const b = cropData.data[idx + 2];
      // simple blue-ish test (HSV-ish): b significantly greater than r,g
      const isBlue = (b > 100) && (b > r + 30) && (b > g + 20);
      mask[y * sw + x] = isBlue ? 255 : 0;
    }
  }
  return { mask, width: sw, height: sh };
}

/** resampleMaskTo - simple nearest-neighbor resample mask to target size */
async function resampleMaskTo(mask, w, h, tw, th) {
  const out = new Uint8ClampedArray(tw * th);
  for (let yy = 0; yy < th; yy++) {
    for (let xx = 0; xx < tw; xx++) {
      const sx = Math.floor(xx * w / tw);
      const sy = Math.floor(yy * h / th);
      out[yy * tw + xx] = mask[sy * w + sx];
    }
  }
  return out;
}

/**
 * maskToImageData - create ImageData (RGBA) from binary mask (0 or 255)
 * used to overlay mask
 */
export function maskToImageData(mask, width, height, overlayColor = [0, 120, 255, 150]) {
  const id = new ImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const v = mask[i];
    id.data[i * 4 + 0] = overlayColor[0]; // R
    id.data[i * 4 + 1] = overlayColor[1]; // G
    id.data[i * 4 + 2] = overlayColor[2]; // B
    id.data[i * 4 + 3] = v ? overlayColor[3] : 0; // alpha only where mask
  }
  return id;
}

/**
 * downloadMaskPNG - produce a PNG of mask at original cropped size
 */
export function downloadMaskPNG(mask, width, height, filename = 'mask.png') {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const id = maskToImageData(mask, width, height);
  ctx.putImageData(id, 0, 0);
  const a = document.createElement('a');
  a.href = canvas.toDataURL('image/png');
  a.download = filename;
  a.click();
}

/**
 * maskToGeoJSON - convert a binary mask (0/255) to a simple polygon GeoJSON (image pixel coordinates)
 * This uses a simple boundary-detection then a Moore-Neighbor tracing. It's simple but works for moderate shapes.
 */
export function maskToGeoJSON(mask, width, height, offsetX = 0, offsetY = 0) {
  // Build boolean mask
  const bin = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) bin[i] = mask[i] ? 1 : 0;

  // Find first foreground pixel
  let start = -1;
  for (let i = 0; i < bin.length; i++) {
    if (bin[i]) { start = i; break; }
  }
  if (start === -1) {
    return { type: 'FeatureCollection', features: [] };
  }

  // Moore-neighbor tracing (single outer contour)
  const coords = [];
  const directions = [
    [-1, 0], [-1, -1], [0, -1], [1, -1],
    [1, 0], [1, 1], [0, 1], [-1, 1]
  ];

  const idxToXY = (i) => [i % width, Math.floor(i / width)];

  let curr = start;
  let prevDir = 0;
  let done = false;
  let loopGuard = 0;
  while (!done && loopGuard++ < 1000000) {
    const [cx, cy] = idxToXY(curr);
    coords.push([cx + offsetX, cy + offsetY]);
    // find next boundary pixel by checking neighbors clockwise starting from prevDir-1
    let found = false;
    for (let d = 0; d < 8; d++) {
      const di = (prevDir + 7 + d) % 8;
      const nx = cx + directions[di][0];
      const ny = cy + directions[di][1];
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        const ni = ny * width + nx;
        if (bin[ni]) {
          // move to next pixel
          curr = ni;
          prevDir = di;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      // no neighbor found; stop
      done = true;
    }
    // stop when we return to starting pixel and have more than a few points
    if (curr === start && coords.length > 20) done = true;
  }

  // build GeoJSON polygon (single ring)
  const polygon = [
    coords.map(pt => [pt[0], pt[1]])
  ];

  return {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature',
      properties: {},
      geometry: {
        type: 'Polygon',
        coordinates: polygon
      }
    }]
  };
}
