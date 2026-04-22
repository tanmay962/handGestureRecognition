// build_bundle.js — Gesture Detection v1.0
// Run: node build_bundle.js
'use strict';
const fs   = require('fs');
const path = require('path');

const BASE = require('path').join(__dirname, 'frontend');
const OUT  = BASE + '/bundle.js';

const FILES = [
  'config/app.config.js',
  'utils/EventBus.js',
  'utils/WSClient.js',
  'utils/MathUtils.js',
  'utils/DOMHelper.js',
  'views/components/index.js',
  'models/NeuralNetwork.js',
  'models/GestureModel.js',
  'models/SentenceModel.js',
  'models/SensorModel.js',
  'services/StorageService.js',
  'services/CameraService.js',
  'services/MQTTService.js',
  'services/GeminiService.js',
  'services/NLPService.js',
  'services/TTSService.js',
  'views/AppView.js',
  'views/DetectView.js',
  'views/TrainView.js',
  'views/SequenceView.js',
  'views/SettingsView.js',
  'views/UserModeView.js',
  'controllers/RecognitionController.js',
  'controllers/SequenceController.js',
  'controllers/TrainingController.js',
  'controllers/AppController.js',
];

function strip(code) {
  code = code.replace(/^export\s+(default\s+)?/gm, '');
  code = code.replace(/^import\s+\{[^}]*\}\s+from\s+['"][^'"]+['"];?\s*\n?/gm, '');
  code = code.replace(/^import\s+\*\s+as\s+\w+\s+from\s+['"][^'"]+['"];?\s*\n?/gm, '');
  code = code.replace(/^import\s+\w+\s+from\s+['"][^'"]+['"];?\s*\n?/gm, '');
  code = code.replace(/^export\s*\{[^}]*\};?\s*\n?/gm, '');
  // Remove 'use strict' — already at top of bundle
  code = code.replace(/^'use strict';\s*\n?/gm, '');
  // Remove duplicate API declarations (both var and const)
  code = code.replace(/^(?:var|const|let) API = '\/api';\s*\n?/gm, '');
  return code;
}

const header = `// Gesture Detection v1.0 — Production Bundle
// Built: ${new Date().toISOString()}
// MediaPipe Holistic: hands + face + body = 41 features
// MLP static + LSTM dynamic + adaptive NLP + Gemini + PWA
// Optimised: rate limiting, confidence smoothing, time-based stability
'use strict';
var API = '/api';

`;

let bundle = header;
let errors = 0;

for (const f of FILES) {
  const fpath = path.join(BASE, f);
  if (!fs.existsSync(fpath)) {
    console.error('❌ MISSING FILE:', fpath);
    errors++;
    continue;
  }
  let code = fs.readFileSync(fpath, 'utf8');
  code = strip(code);
  bundle += `\n// ═══ ${f} ═══\n` + code + '\n';
}

if (errors > 0) {
  console.error(`BUILD FAILED: ${errors} missing file(s)`);
  process.exit(1);
}

fs.writeFileSync(OUT, bundle);
console.log(`Bundle: ${bundle.length} chars, ${bundle.split('\n').length} lines`);

// ── Verification checks ──────────────────────────────────────
const checks = [
  ['APP_CONFIG',              'APP_CONFIG'],
  ['STATIC_INPUT 41',         'STATIC_INPUT:        41'],
  ['Holistic CDN',            'holistic@0.5'],
  ['CameraService Holistic',  '_buildFeatureVector'],
  ['Face features',           '_extractFaceFeatures'],
  ['Pose features',           '_extractPoseFeatures'],
  ['SensorModel 41',         'features.length < 41'],
  ['Rate limiting',           '_predictEvery'],
  ['Ensemble voting',         '_ensembleVote'],
  ['Time-based stability',    '_stableStartTime'],
  ['Separate cooldowns',      'COOLDOWN_DIFF_LETTER'],
  ['Motion detection',        '_detectMotion'],
  ['NLP debounce',            '_scheduleNLP'],
  ['AppView detect tab',      "case 'detect'"],
  ['No VR tab',               true],
  ['No BLE service',          true],
  ['No Three.js',             true],
  ['RecognitionController',   'RecognitionController'],
  ['TrainingController',      'TrainingController'],
  ['AppController',           'AppController'],
  ['NeuralNetwork',           'NeuralNetwork'],
  ['GestureModel',            'GestureModel'],
];

let allOk = true;
for (const [label, check] of checks) {
  let ok;
  if (label === 'No VR tab')    ok = !bundle.includes('renderVRTab') && !bundle.includes('VRController');
  else if (label === 'No BLE service') ok = !bundle.includes('BLEService') && !bundle.includes('connectBLE');
  else if (label === 'No Three.js')    ok = !bundle.includes('THREE.') && !bundle.includes('three.js');
  else ok = typeof check === 'string' ? bundle.includes(check) : check;
  if (!ok) { console.error('❌ FAILED:', label); allOk = false; }
}

// Check no optional chaining
const optChain = bundle.split('\n').filter(function(l) {
  const s = l.replace(/\/\/.*/, '').replace(/'[^']*'/g, '').replace(/"[^"]*"/g, '');
  return s.includes('?.');
});
if (optChain.length > 0) {
  optChain.slice(0, 3).forEach(function(l) { console.error('❌ OPT CHAIN:', l.trim().slice(0, 80)); });
  allOk = false;
}

console.log(allOk ? '✅ All checks passed' : '❌ Some checks failed');
process.exit(allOk ? 0 : 1);
