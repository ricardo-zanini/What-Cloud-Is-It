import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ActivityIndicator,
  Alert,
  TouchableOpacity,
  Dimensions,
} from 'react-native';

import { CameraView, useCameraPermissions } from 'expo-camera';
import * as FileSystem from 'expo-file-system/legacy';
import { Asset } from 'expo-asset';
import * as ImageManipulator from 'expo-image-manipulator';
import jpegjs from 'jpeg-js';
import { loadTensorflowModel } from 'react-native-fast-tflite';

// ─────────────────────────────────────────────
//  Constantes
// ─────────────────────────────────────────────
const CAPTURE_INTERVAL_MS  = 100;
const BUFFER_SIZE          = 10;
const INPUT_WIDTH          = 224;
const INPUT_HEIGHT         = 224;
const CAPTURE_SQUARE_RATIO = 0.75;

const MODEL_ASSET  = require('./assets/models/model.tflite');
const LABELS_ASSET = require('./assets/models/labels.txt');

// ─────────────────────────────────────────────
//  Calcula dimensões do quadrado de captura
// ─────────────────────────────────────────────
function getCaptureSquareDimensions() {
  const { width: screenWidth, height: screenHeight } = Dimensions.get('window');
  const minScreenDim = Math.min(screenWidth, screenHeight);
  const squareSize   = minScreenDim * CAPTURE_SQUARE_RATIO;
  const squareX      = (screenWidth  - squareSize) / 2;
  const squareY      = (screenHeight - squareSize) / 2;
  return { squareSize, squareX, squareY, screenWidth, screenHeight };
}

// ─────────────────────────────────────────────
//  Resolve URI local de um asset, copiando para
//  documentDirectory se necessário.
//
//  Em builds standalone o asset.localUri pode ser
//  uma URI interna do APK (ex.: "asset:/…") que
//  APIs nativas de I/O não conseguem abrir.
//  Copiar para documentDirectory garante um path
//  de arquivo real acessível por qualquer lib nativa.
// ─────────────────────────────────────────────
async function resolveAssetToFileSystem(assetModule, destFilename) {
  // 1. Carrega o asset pelo sistema do Expo
  const [asset] = await Asset.loadAsync(assetModule);

  // 2. Garante que o arquivo local exista
  if (!asset.localUri) {
    await asset.downloadAsync();
  }

  const srcUri = asset.localUri ?? asset.uri;
  if (!srcUri) {
    throw new Error(`Não foi possível resolver a URI do asset: ${destFilename}`);
  }

  // 3. Destino final no documentDirectory (path nativo real)
  const destUri = FileSystem.documentDirectory + destFilename;

  // 4. Só copia se o arquivo ainda não existe (evita cópia desnecessária)
  const destInfo = await FileSystem.getInfoAsync(destUri);
  if (!destInfo.exists) {
    await FileSystem.copyAsync({ from: srcUri, to: destUri });
  }

  return destUri;
}

// ─────────────────────────────────────────────
//  Carrega labels do .txt
// ─────────────────────────────────────────────
async function loadLabels() {
  // Resolve para um path de arquivo real antes de ler
  const labelsUri = await resolveAssetToFileSystem(LABELS_ASSET, 'labels.txt');
  const content   = await FileSystem.readAsStringAsync(labelsUri);
  return content.split('\n').map(l => l.trim()).filter(Boolean);
}

// ─────────────────────────────────────────────
//  Carrega modelo TFLite via JSI
// ─────────────────────────────────────────────
async function loadModel() {
  // Resolve para um path de arquivo real antes de passar ao TFLite
  const modelUri = await resolveAssetToFileSystem(MODEL_ASSET, 'model.tflite');

  const model = await loadTensorflowModel({ url: modelUri }, []);

  console.log('[model] inputs:',  JSON.stringify(model.inputs));
  console.log('[model] outputs:', JSON.stringify(model.outputs));

  return model;
}

// ─────────────────────────────────────────────
//  Captura frame quadrado, redimensiona e retorna
//  Float32Array RGB [H*W*3]
// ─────────────────────────────────────────────
async function captureAndPreprocess(cameraRef) {
  const { squareSize, squareX, squareY, screenWidth, screenHeight } =
    getCaptureSquareDimensions();

  const photo = await cameraRef.current.takePictureAsync({
    base64:          false,
    quality:         0.8,
    skipProcessing:  true,
    shutterSound:    false,
  });

  const photoAspectRatio  = photo.width / photo.height;
  const screenAspectRatio = screenWidth  / screenHeight;

  let photoSquareX, photoSquareY, photoSquareSize;

  if (photoAspectRatio > screenAspectRatio) {
    const scale     = photo.height / screenHeight;
    photoSquareSize = squareSize * scale;
    photoSquareX    = (photo.width - squareSize * scale) / 2;
    photoSquareY    = squareY * scale;
  } else {
    const scale     = photo.width / screenWidth;
    photoSquareSize = squareSize * scale;
    photoSquareX    = squareX * scale;
    photoSquareY    = (photo.height - squareSize * scale) / 2;
  }

  const resized = await ImageManipulator.manipulateAsync(
    photo.uri,
    [
      {
        crop: {
          originX: Math.floor(photoSquareX),
          originY: Math.floor(photoSquareY),
          width:   Math.floor(photoSquareSize),
          height:  Math.floor(photoSquareSize),
        },
      },
      { resize: { width: INPUT_WIDTH, height: INPUT_HEIGHT } },
    ],
    { base64: true, format: ImageManipulator.SaveFormat.JPEG },
  );

  const binaryStr = atob(resized.base64);
  const jpegBytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    jpegBytes[i] = binaryStr.charCodeAt(i);
  }

  const decoded      = jpegjs.decode(jpegBytes, { useTArray: true });
  const pixelCount   = decoded.width * decoded.height;
  const float32Input = new Float32Array(pixelCount * 3);

  for (let i = 0; i < pixelCount; i++) {
    float32Input[i * 3]     = decoded.data[i * 4];      // R  [0–255]
    float32Input[i * 3 + 1] = decoded.data[i * 4 + 1]; // G  [0–255]
    float32Input[i * 3 + 2] = decoded.data[i * 4 + 2]; // B  [0–255]
  }

  return float32Input;
}

// ─────────────────────────────────────────────
//  Roda inferência e retorna label + confiança
// ─────────────────────────────────────────────
async function runInference(model, float32Input, labels) {
  const outputs = await model.run([float32Input.buffer]);
  const probs   = new Float32Array(outputs[0]);

  let maxIdx  = 0;
  let maxProb = probs[0];
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > maxProb) { maxProb = probs[i]; maxIdx = i; }
  }

  return {
    label:      labels[maxIdx] ?? `Classe ${maxIdx}`,
    confidence: (maxProb * 100).toFixed(1),
  };
}

// ─────────────────────────────────────────────
//  Moda do buffer de classificações
// ─────────────────────────────────────────────
function computeMajority(buffer) {
  if (!buffer || buffer.length === 0) return null;

  const counts = {};
  for (const item of buffer) {
    counts[item.label] = (counts[item.label] || 0) + 1;
  }

  let bestLabel = null;
  let bestCount = 0;
  for (const [label, count] of Object.entries(counts)) {
    if (count > bestCount) { bestCount = count; bestLabel = label; }
  }

  const matching       = buffer.filter(item => item.label === bestLabel);
  const avgConfidence  =
    matching.reduce((sum, item) => sum + parseFloat(item.confidence), 0) / matching.length;

  return {
    label:         bestLabel,
    count:         bestCount,
    total:         buffer.length,
    avgConfidence: avgConfidence.toFixed(1),
  };
}

// ─────────────────────────────────────────────
//  Overlay do quadrado de captura
// ─────────────────────────────────────────────
function CaptureSquareOverlay() {
  const { squareSize, squareX, squareY } = getCaptureSquareDimensions();

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      <View style={[styles.scrim, { height: squareY }]} />

      <View style={{ flexDirection: 'row', flex: 1 }}>
        <View style={[styles.scrim, { width: squareX }]} />

        <View style={[styles.captureSquare, { width: squareSize, height: squareSize }]}>
          <View style={[styles.corner, styles.topLeft]} />
          <View style={[styles.corner, styles.topRight]} />
          <View style={[styles.corner, styles.bottomLeft]} />
          <View style={[styles.corner, styles.bottomRight]} />
        </View>

        <View style={[styles.scrim, { flex: 1 }]} />
      </View>

      <View style={[styles.scrim, { flex: 1 }]} />
    </View>
  );
}

// ─────────────────────────────────────────────
//  Componente principal
// ─────────────────────────────────────────────
export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isReady, setIsReady]           = useState(false);
  const [error, setError]               = useState(null);
  const [zoom, setZoom]                 = useState(0);
  const [isStreaming, setIsStreaming]   = useState(false);
  const [majority, setMajority]         = useState(null);

  const cameraRef       = useRef(null);
  const modelRef        = useRef(null);
  const labelsRef       = useRef([]);
  const busyRef         = useRef(false);
  const isStreamingRef  = useRef(false);
  const bufferRef       = useRef([]);

  // ── Permissão de câmera ──────────────────────
  useEffect(() => {
    if (!permission) return;
    if (!permission.granted) {
      requestPermission().then(res => {
        if (!res.granted)
          Alert.alert('Permissão negada', 'Acesso à câmera é necessário.');
      });
    }
  }, [permission]);

  // ── Carrega modelo + labels ──────────────────
  // resolveAssetToFileSystem garante que os arquivos
  // estejam em um path nativo real antes de serem
  // consumidos por FileSystem.readAsStringAsync e
  // loadTensorflowModel — necessário em APKs standalone
  // onde asset.localUri pode ser uma URI interna do APK.
  useEffect(() => {
    (async () => {
      try {
        const [model, labels] = await Promise.all([loadModel(), loadLabels()]);
        modelRef.current  = model;
        labelsRef.current = labels;
        setIsReady(true);
      } catch (e) {
        setError(`Erro ao carregar modelo:\n${e.message}`);
      }
    })();
  }, []);

  // ── Garante que o loop pare se o componente desmontar ──
  useEffect(() => {
    return () => { isStreamingRef.current = false; };
  }, []);

  // ── Loop de captura contínua ─────────────────
  const captureLoop = useCallback(async () => {
    if (!isStreamingRef.current) return;

    if (busyRef.current || !cameraRef.current || !modelRef.current) {
      if (isStreamingRef.current) setTimeout(captureLoop, CAPTURE_INTERVAL_MS);
      return;
    }

    busyRef.current = true;
    try {
      const float32Input = await captureAndPreprocess(cameraRef);
      const result       = await runInference(modelRef.current, float32Input, labelsRef.current);

      bufferRef.current = [...bufferRef.current, result].slice(-BUFFER_SIZE);
      setMajority(computeMajority(bufferRef.current));
    } catch (e) {
      console.warn('[captureLoop]', e.message);
    } finally {
      busyRef.current = false;
      if (isStreamingRef.current) setTimeout(captureLoop, CAPTURE_INTERVAL_MS);
    }
  }, []);

  // ── Toggle captura ───────────────────────────
  const handleCapture = useCallback(() => {
    if (!isReady) return;

    if (isStreamingRef.current) {
      isStreamingRef.current = false;
      setIsStreaming(false);
    } else {
      isStreamingRef.current = true;
      bufferRef.current      = [];
      setMajority(null);
      setIsStreaming(true);
      captureLoop();
    }
  }, [isReady, captureLoop]);

  // ── Controles de zoom ────────────────────────
  const handleZoomIn    = useCallback(() => setZoom(prev => Math.min(prev + 0.1, 1)), []);
  const handleZoomOut   = useCallback(() => setZoom(prev => Math.max(prev - 0.1, 0)), []);
  const handleZoomReset = useCallback(() => setZoom(0), []);

  // ── Render ───────────────────────────────────
  if (error) {
    return (
      <View style={styles.centered}>
        <Text style={styles.errorText}>{error}</Text>
      </View>
    );
  }

  if (!permission?.granted) {
    return (
      <View style={styles.centered}>
        <Text style={styles.infoText}>Aguardando permissão de câmera…</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        facing="back"
        zoom={zoom}
        animateShutter={false}
      />

      <CaptureSquareOverlay />

      <View style={styles.overlay}>
        {/* Topo: carregando + Zoom */}
        <View style={styles.topControls}>
          {!isReady && (
            <View style={styles.badge}>
              <ActivityIndicator color="#fff" size="small" />
              <Text style={styles.badgeText}>Carregando modelo TFLite…</Text>
            </View>
          )}

          {isReady && (
            <View style={styles.zoomContainer}>
              <TouchableOpacity
                style={styles.zoomButton}
                onPress={handleZoomOut}
                disabled={zoom === 0}
              >
                <Text style={styles.zoomButtonText}>−</Text>
              </TouchableOpacity>

              <View style={styles.zoomDisplay}>
                <Text style={styles.zoomText}>
                  {((zoom * 100) + 100).toFixed(0)}%
                </Text>
              </View>

              <TouchableOpacity
                style={styles.zoomButton}
                onPress={handleZoomIn}
                disabled={zoom === 1}
              >
                <Text style={styles.zoomButtonText}>+</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.zoomButton, styles.resetButton]}
                onPress={handleZoomReset}
              >
                <Text style={styles.zoomButtonText}>⟲</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>

        {/* Resultado */}
        {isReady && (
          <View style={styles.resultCard}>
            {majority ? (
              <>
                <Text style={styles.cloudLabel}>{majority.label}</Text>
                <Text style={styles.confidence}>
                  {majority.count}/{majority.total} capturas · {majority.avgConfidence}% confiança média
                </Text>
                <Text style={styles.hint}>
                  {isStreaming ? 'Capturando continuamente…' : 'Toque para iniciar nova captura'}
                </Text>
              </>
            ) : (
              <Text style={styles.hint}>
                {isStreaming
                  ? 'Capturando… aguardando classificações'
                  : 'Toque no botão para iniciar a captura contínua'}
              </Text>
            )}
          </View>
        )}
      </View>

      {/* Rodapé: botão de captura */}
      <View style={styles.bottomSection}>
        <TouchableOpacity
          style={[
            styles.captureButton,
            isStreaming && styles.captureButtonActive,
            !isReady   && styles.captureButtonDisabled,
          ]}
          onPress={handleCapture}
          disabled={!isReady}
        >
          {isStreaming
            ? <View style={styles.stopIcon} />
            : <View style={styles.cameraIcon} />}
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ─────────────────────────────────────────────
//  Estilos
// ─────────────────────────────────────────────
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0a0a0a',
    padding: 24,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 60,
    paddingBottom: 48,
    paddingHorizontal: 20,
  },
  topControls: { width: '100%', alignItems: 'center', gap: 12 },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 24,
  },
  badgeText:    { color: '#e0e0e0', fontSize: 13, fontWeight: '500' },
  zoomContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 24,
  },
  zoomButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(59, 130, 246, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  resetButton:   { backgroundColor: 'rgba(168, 85, 247, 0.7)', marginLeft: 4 },
  zoomButtonText: { color: '#fff', fontSize: 18, fontWeight: '600' },
  zoomDisplay: {
    minWidth: 50,
    paddingHorizontal: 8,
    paddingVertical: 4,
    backgroundColor: 'rgba(59, 130, 246, 0.3)',
    borderRadius: 12,
  },
  zoomText: { color: '#3b82f6', fontSize: 12, fontWeight: '600', textAlign: 'center' },
  resultCard: {
    top: 10,
    width: '100%',
    backgroundColor: 'rgba(10,20,40,0.75)',
    borderRadius: 20,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.12)',
  },
  cloudLabel:  { color: '#ffffff', fontSize: 28, fontWeight: '700', letterSpacing: 0.4, textAlign: 'center' },
  confidence:  { color: '#7dd3fc', fontSize: 15, marginTop: 4, fontWeight: '500' },
  hint:        { color: 'rgba(255,255,255,0.4)', fontSize: 12, marginTop: 10 },
  infoText:    { color: '#aaa', fontSize: 15, textAlign: 'center' },
  errorText:   { color: '#f87171', fontSize: 14, textAlign: 'center', lineHeight: 22 },
  scrim:       { backgroundColor: 'rgba(0, 0, 0, 0.5)' },
  captureSquare: {
    borderWidth: 2,
    borderColor: '#3b82f6',
    backgroundColor: 'rgba(59, 130, 246, 0.05)',
    position: 'relative',
  },
  corner:      { position: 'absolute', width: 20, height: 20, borderColor: '#3b82f6', borderWidth: 3 },
  topLeft:     { top: -2, left: -2, borderRightWidth: 0, borderBottomWidth: 0 },
  topRight:    { top: -2, right: -2, borderLeftWidth: 0, borderBottomWidth: 0 },
  bottomLeft:  { bottom: -2, left: -2, borderRightWidth: 0, borderTopWidth: 0 },
  bottomRight: { bottom: -2, right: -2, borderLeftWidth: 0, borderTopWidth: 0 },
  resultSection: { width: '100%', alignItems: 'center', position: 'relative', gap: 16 },
  bottomSection: { width: '100%', alignItems: 'center', position: 'absolute', bottom: 100, gap: 16 },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(59, 130, 246, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#3b82f6',
  },
  captureButtonActive:   { backgroundColor: 'rgba(34, 197, 94, 0.8)', borderColor: '#22c55e' },
  captureButtonDisabled: { backgroundColor: 'rgba(107, 114, 128, 0.5)', borderColor: 'rgba(107, 114, 128, 0.5)' },
  cameraIcon: { width: 40, height: 35, borderRadius: 4, backgroundColor: '#fff', borderWidth: 2, borderColor: '#fff' },
  stopIcon:   { width: 28, height: 28, borderRadius: 4, backgroundColor: '#fff' },
});