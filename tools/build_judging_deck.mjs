/**
 * Rebuild the six-slide judging deck from preserved measured evidence.
 * Requires the Codex bundled @oai/artifact-tool runtime. No network access.
 * Usage: node tools/build_judging_deck.mjs [--runtime ROOT] [--skill-dir DIR]
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const arg = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const runtime = arg('--runtime', 'C:/Users/samee/.cache/codex-runtimes/codex-primary-runtime/dependencies');
const skillDir = arg('--skill-dir', 'C:/Users/samee/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations');
process.env.RUNTIME_NODE_MODULES = path.join(runtime, 'node/node_modules');
process.env.RUNTIME_NODE = path.join(runtime, 'node/bin/node.exe');
process.env.RUNTIME_PYTHON = path.join(runtime, 'python/python.exe');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const buildDir = path.join(ROOT, 'runs/presentation-build', stamp);
const outputDir = path.join(ROOT, 'presentation');
await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
// Chart workbook scratch files must stay out of the repository root.
process.chdir(buildDir);
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(path.join(runtime, 'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);

const sourcePaths = [
  'docs/judging-presentation.md',
  'research/results/encoder-v2/confirmation-result.json',
  'research/results/encoder-v2/confirmation-plan.json',
  'research/results/encoder-v2/final-policy.json',
  'research/results/encoder-v2/flower-head.json',
  'research/results/encoder-v2/README.md',
  'research/results/pilot-r2-20261103/README.md',
  'research/results/encoder-matrix-v1/README.md',
  'research/results/encoder-matrix-v1/summary.json',
  'research/results/remaining-work-v1/README.md',
  'research/results/remaining-work-v1/five-recovery.json',
  'research/results/remaining-work-v1/offline.json',
  'research/results/remaining-work-v1/comparison.png',
  'research/results/remaining-work-v1/comparison-provenance.json',
];
const sourceHashes = {};
for (const name of sourcePaths) sourceHashes[name] = createHash('sha256').update(await fs.readFile(path.join(ROOT, name))).digest('hex');
const readJson = async name => JSON.parse(await fs.readFile(path.join(ROOT, name), 'utf8'));
const confirmation = await readJson('research/results/encoder-v2/confirmation-result.json');
const matrix = await readJson('research/results/encoder-matrix-v1/summary.json');
const recovery = await readJson('research/results/remaining-work-v1/five-recovery.json');
const head = await readJson('research/results/encoder-v2/flower-head.json');
const stat = name => matrix.groups[name].final.accuracy;
const imbalance = ['sample', 'equal', 'equal_clipped'].map(method => stat(`imbalanced-heterogeneous-${method}`));
const included = stat('expanded-heterogeneous-equal');
const omitted = stat('expanded-heterogeneous-equal-omit4');
// Excel stores 15 significant decimal digits. Preserve full source precision in
// the manifest and round only the editable chart snapshot, below display precision.
const excelNumber = value => Number(value.toPrecision(15));
if (confirmation.correct !== 282 || confirmation.metrics.count !== 300 || matrix.completed !== 33 || recovery.accepted_counts.join(',') !== '5,4,5,2,5') throw new Error('Evidence identity or expected measurement changed. Review the deck before rebuilding.');
if (included.values.some((v, i) => Math.abs(v - omitted.values[i] - .18) > 1e-10)) throw new Error('Omission comparison changed.');

const C = { ink: '#152D38', teal: '#287B8E', orange: '#D77B37', slate: '#8C969E', muted: '#526A74', pale: '#E8F1F3', white: '#FFFFFF', grid: '#DCE4E7' };
const FONT = 'Arial';
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const noteIndex = [];
function text(slide, name, value, x, y, w, h, size = 28, color = C.ink, bold = false, align = 'left') {
  const s = slide.shapes.add({ geometry: 'textbox', name, position: { left: x, top: y, width: w, height: h }, fill: 'none', line: { fill: 'none', width: 0 } });
  s.text = value;
  s.text.style = { typeface: FONT, fontSize: size, color, bold, alignment: align, verticalAlignment: 'top', wrap: 'square', autoFit: 'none', insets: 0 };
  return s;
}
function slide(title, num, dark = false) {
  const s = deck.slides.add();
  s.background.fill = dark ? C.ink : C.white;
  if (title) text(s, 'slide-title', title, 64, 48, 1152, 100, 44, dark ? C.white : C.ink, true);
  text(s, 'page-number', String(num).padStart(2, '0'), 1160, 652, 56, 26, 17, dark ? '#B8CCD4' : C.muted, false, 'right');
  return s;
}
function notes(s, timing, speech, sources, identity = '') {
  const n = `Timing: ${timing}\n\n${speech}\n\nSources (repository-relative):\n${sources.map(x => `${x}\nSHA-256 ${sourceHashes[x]}`).join('\n')}\n${identity ? `\nMeasured identity: ${identity}\n` : ''}`;
  s.speakerNotes.text = n;
  noteIndex.push({ slide: noteIndex.length + 1, timing, speech, sources, identity });
}
function chart(s, type, name, config) {
  const c = s.charts.add(type, { ...config, chartFill: 'none', chartLine: { fill: 'none', width: 0 }, plotAreaFill: 'none', plotAreaLine: { fill: 'none', width: 0 } });
  applyPresentationChartFont(c, { fontFamily: FONT });
  return c;
}
const axis = { visible: true, min: 0, max: 1, majorUnit: .2, numberFormatCode: '0%', textStyle: { fontSize: 18, fill: C.muted }, line: { fill: 'none', width: 0 }, majorGridlines: { style: 'solid', fill: C.grid, width: 1 } };
const catAxis = { visible: true, textStyle: { fontSize: 20, fill: C.ink }, line: { fill: C.grid, width: 1 }, majorGridlines: null };

// 1: Problem and scope. Large scope numerals are editable text.
{
  const s = slide('', 1, true);
  text(s, 'project-name', 'FedLoRA-Edge', 64, 74, 1130, 92, 72, C.white, true);
  text(s, 'problem', 'Useful federation with\nunequal training budgets', 64, 211, 1020, 146, 46, C.white, true);
  text(s, 'question', 'Different data. Limited work. Clients that disappear.', 64, 391, 1080, 48, 28, '#C5D7DE');
  const facts = [ ['5', 'logical clients'], ['20', 'banking intents'], ['1', 'physical laptop today'] ];
  facts.forEach(([n, label], i) => {
    const x = 64 + i * 392;
    text(s, `scope-${i}`, n, x, 489, 300, 85, 72, i === 2 ? '#F0A768' : '#72BECC', true);
    text(s, `scope-label-${i}`, label, x, 582, 360, 42, 25, C.white);
  });
  notes(s, '0:00–0:25 (25 seconds)',
    'FedLoRA-Edge asks whether machines with unequal training budgets can contribute to one useful model, without a data-rich participant controlling aggregation or a disappearing client stopping the run. This banking-intent prototype has five logical clients, each covering different knowledge. It predicts one of twenty intents and displays an authored support response. All five clients currently share one physical laptop.',
    ['docs/judging-presentation.md', 'research/results/remaining-work-v1/README.md']);
}

// 2: Explicit architecture distinction, then common protocol.
{
  const s = slide('Model variants and federation protocol', 2);
  text(s, 'original-heading', 'Original LoRA benchmark', 64, 172, 534, 42, 30, C.muted, true);
  text(s, 'original-body', 'SmolLM2-135M-Instruct\nShared frozen A\nTrain and transmit LoRA B', 64, 230, 530, 162, 29);
  text(s, 'current-heading', 'Confirmed query model', 670, 172, 546, 42, 30, C.teal, true);
  text(s, 'current-body', 'Frozen MPNet encoder\nTrain and transmit a linear head\nHead-only training', 670, 230, 546, 162, 29);
  text(s, 'distinction', 'The original negative results remain preserved. The 94% result belongs to the MPNet-head variant.', 64, 420, 1120, 72, 25, C.muted);
  text(s, 'protocol-heading', 'Each round', 64, 529, 185, 42, 26, C.teal, true);
  const items = [ ['1', 'Clients train from\nthe same version'], ['2', 'Server validates replies\nwithin a deadline'], ['3', 'Quorum commits\nequal survivor weights'] ];
  items.forEach(([n, value], i) => {
    const x = 268 + i * 315;
    text(s, `step-number-${i}`, n, x, 525, 42, 60, 42, C.teal, true);
    text(s, `step-text-${i}`, value, x + 55, 529, 253, 102, 23);
  });
  notes(s, '0:25–0:55 (30 seconds)',
    'The original benchmark trains B-only LoRA on SmolLM2 with a shared frozen A. Its accuracy was weak. I implemented a separate frozen-MPNet encoder with a federated classifier head. The 94 percent result belongs to that head-only variant. Both use the same protocol: clients train from one version, the server validates replies within a deadline, and a quorum commits equal survivor weights. Local preparation embeds each participant’s own records.',
    ['docs/judging-presentation.md', 'research/results/pilot-r2-20261103/README.md', 'research/results/encoder-v2/flower-head.json', 'research/results/remaining-work-v1/README.md'],
    `Confirmed model run ${confirmation.run_id}; kind ${head.kind}; selected version 500. The original and head-only scores are not a matched architecture comparison.`);
}

// 3: Locked confirmation result and exact raw trainable payload.
{
  const s = slide('94% on locked independent confirmation', 3);
  text(s, 'accuracy', `${(confirmation.metrics.accuracy * 100).toFixed(0)}%`, 64, 178, 550, 184, 160, C.teal, true);
  text(s, 'confirmation-count', `${confirmation.correct} correct out of ${confirmation.metrics.count}`, 70, 380, 620, 52, 37, C.ink, true);
  text(s, 'confirmation-boundary', '20 intents, 18 errors\nModel and rules locked before scoring', 70, 453, 635, 95, 27, C.muted);
  text(s, 'payload', '61,520 B', 800, 218, 405, 80, 58, C.teal, true);
  text(s, 'payload-label', 'raw FP32 head\nper direction', 800, 320, 405, 90, 31);
  text(s, 'parameter-count', '15,380 trainable parameters\nTransport overhead excluded', 800, 453, 405, 90, 24, C.muted);
  text(s, 'accuracy-footer', '500 actual Flower training attempts. Reused validation: 95.5%.', 64, 605, 1088, 36, 24, C.muted);
  notes(s, '0:55–1:25 (30 seconds)',
    'The selected model completed five hundred actual Flower training attempts. It reached 95.5 percent on reused validation. I locked the model and rules before scoring separate confirmation queries: 282 of 300 correct, or 94 percent, across twenty intents. That includes eighteen errors. Its 15,380 trainable parameters require 61,520 raw bytes per direction. This excludes transport overhead and the frozen encoder’s storage, memory and computation.',
    ['research/results/encoder-v2/confirmation-result.json', 'research/results/encoder-v2/confirmation-plan.json', 'research/results/encoder-v2/final-policy.json', 'research/results/encoder-v2/flower-head.json'],
    `Run ${confirmation.run_id}, selected version 500; config ${confirmation.config_hash}; head ${confirmation.head_hash}; confirmation lock ${confirmation.lock_hash}; pre-scoring commit 7413e6b. Scored 3 November 2026. No retuning on this scored confirmation set.`);
}

// 4: Complete literal chart data stays native and receives embedded workbooks.
{
  const s = slide('Equal weighting and weak-client contribution', 4);
  text(s, 'study-subtitle', 'Three matched seeds, fixed attempt 500, reused development validation', 64, 139, 1150, 34, 23, C.muted);
  chart(s, 'bar', 'imbalance-chart', {
    position: { left: 64, top: 195, width: 554, height: 337 },
    title: '90% data imbalance', titleTextStyle: { fontSize: 27, bold: true, fill: C.ink },
    categories: ['Sample', 'Equal', 'Equal + clip'],
    series: [{ name: 'Final accuracy', values: imbalance.map(x => excelNumber(x.mean)), fill: C.teal, points: [{ idx: 0, fill: C.slate }, { idx: 1, fill: C.teal }, { idx: 2, fill: C.orange }], valuesFormatCode: '0.00%' }],
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 72, varyColors: true }, hasLegend: false,
    xAxis: catAxis, yAxis: axis,
    dataLabels: { showValue: true, position: 'outEnd', numberFormatCode: '0.0%', textStyle: { fontSize: 23, fill: C.ink, bold: true } },
  });
  chart(s, 'bar', 'omission-chart', {
    position: { left: 673, top: 195, width: 543, height: 337 },
    title: 'One-step verification client', titleTextStyle: { fontSize: 27, bold: true, fill: C.ink },
    categories: ['42', '43', '44'],
    series: [
      { name: 'All clients', values: included.values, fill: C.teal, valuesFormatCode: '0.0%' },
      { name: 'Omit client 4', values: omitted.values, fill: C.slate, valuesFormatCode: '0.0%' },
    ],
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 80 }, hasLegend: true,
    legend: { position: 'bottom', textStyle: { fontSize: 18, fill: C.ink } },
    xAxis: { ...catAxis, title: { text: 'Training seed', textStyle: { fontSize: 18, fill: C.muted } } }, yAxis: axis,
    dataLabels: { showValue: false, textStyle: { fontSize: 20, fill: C.ink } },
  });
  text(s, 'imbalance-caption', '400 training queries, unequal budgets\n62.5%, 76.5%, 78.17% means\nClipped sample SD: 0.29 percentage points', 64, 548, 572, 102, 22, C.muted);
  text(s, 'omission-caption', '2,760-query control, unequal budgets\n95.5% to 77.5% in every seed\n18-point benefit from its four unique intents', 673, 548, 543, 102, 22, C.muted);
  notes(s, '1:25–2:00 (35 seconds)',
    'The development matrix completed thirty-three numerical training runs, each with five hundred attempts across three matched seeds. With four hundred queries and ninety percent on one client, final accuracy averaged 62.5 percent for sample weighting, 76.5 for equal weighting and 78.17 with clipping. Most improvement came from weighting. Separately, omitting the one-step verification client reduced 95.5 to 77.5 percent in every seed. It owns the only examples for four intents. These reused-validation comparisons remain separate from the independent score.',
    ['research/results/encoder-matrix-v1/summary.json', 'research/results/encoder-matrix-v1/README.md', 'research/results/remaining-work-v1/comparison.png', 'research/results/remaining-work-v1/comparison-provenance.json'],
    `Study encoder-matrix-v1; plan ${matrix.plan_hash}; ${matrix.completed}/33 cells; seeds 42/43/44; all endpoint values from groups.*.final.accuracy. Sample and equal SD 0 pp; clipped SD ${(imbalance[2].sample_std * 100).toFixed(8)} pp. Three-seed development variability does not establish general significance. Methods share initial tensors, data, optimizer, budgets and client random streams.`);
}

// 5: Real same-host recovery, independent of classifier accuracy.
{
  const s = slide('Real process loss, quorum retention, fresh rejoin', 5);
  text(s, 'recovery-subtitle', 'Five logical clients on one laptop, Flower Deployment Runtime with TLS', 64, 140, 1152, 36, 24, C.muted);
  chart(s, 'bar', 'recovery-chart', {
    position: { left: 64, top: 212, width: 744, height: 369 },
    title: 'Accepted contributors by attempt', titleTextStyle: { fontSize: 28, bold: true, fill: C.ink },
    categories: ['1', '2', '3', '4', '5'],
    series: [{ name: 'Accepted contributors', values: recovery.accepted_counts, fill: C.teal, points: [{ idx: 3, fill: C.orange }], valuesFormatCode: '0' }],
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 100 }, hasLegend: false,
    xAxis: { ...catAxis, title: { text: 'Attempt', textStyle: { fontSize: 20, fill: C.muted } } },
    yAxis: { ...axis, min: 0, max: 6, majorUnit: 1, numberFormatCode: '0' },
    dataLabels: { showValue: true, position: 'outEnd', numberFormatCode: '0', textStyle: { fontSize: 27, bold: true, fill: C.ink } },
  });
  text(s, 'quorum-callout', 'Quorum: 3', 881, 221, 335, 56, 40, C.teal, true);
  text(s, 'attempt-2', 'Attempt 2\n4 survivors commit\n0.25 weight each', 881, 314, 335, 112, 25);
  text(s, 'attempt-4', 'Attempt 4\n2 replies miss quorum\nVersion 3 stays intact', 881, 464, 335, 112, 25, C.orange, true);
  text(s, 'rejoin-footer', 'Attempts 3 and 5 rejoin with the current head. Full rehearsal: 303 seconds.', 64, 608, 1080, 38, 24, C.muted);
  notes(s, '2:00–2:35 (35 seconds)',
    'This chart records real Flower processes on one laptop. The accepted counts are five, four, five, two and five. In attempt two, we kill a client’s process tree. Four survivors commit with a quarter weight each. Attempt three rejoins with fresh state. Attempt four misses the quorum of three and retains version three. Attempt five accepts everyone again. We verify each incoming head against current state. The full rehearsal took 303 seconds and started from the trained head.',
    ['research/results/remaining-work-v1/five-recovery.json', 'research/results/remaining-work-v1/README.md'],
    `Run ${recovery.run_id}; config ${recovery.config_hash}; initial head ${recovery.initial_head_hash}; committed versions 0,1,2,3,3,4; quorum 3; collection deadline 60 seconds. These later checkpoints do not replace the confirmed query model.`);
}

// 6: Live inference and bounded next claims.
{
  const s = slide('Live local query, then physical-laptop validation', 6, true);
  text(s, 'demo-query', '“My card payment\nis still pending.”', 64, 200, 724, 150, 52, C.white, true);
  text(s, 'demo-action', 'Local intent prediction\nwith an authored support response', 64, 399, 675, 104, 30, '#C5D7DE');
  text(s, 'next-label', 'Next evidence', 868, 210, 348, 44, 30, '#72BECC', true);
  text(s, 'next-devices', '3 physical laptops\nLAN dropout and rejoin', 868, 291, 348, 95, 28, C.white);
  text(s, 'next-data', 'Fresh held-out queries\nbefore any new accuracy claim', 868, 430, 348, 115, 28, C.white);
  text(s, 'contribution', 'Reproducible federation with established methods', 64, 605, 1040, 40, 28, '#72BECC', true);
  notes(s, '2:35–3:00 (25 seconds, including a brief live query)',
    'Stage action: switch to the selected-model tab, classify the displayed query, and show its answer or clarification. Say: “This prediction runs locally now. My contribution integrates and measures established federation methods. Next are three physical laptops, a LAN dropout and rejoin, and fresh held-out queries before any new accuracy claim.” If the page fails, identify this deck as recorded evidence.',
    ['docs/judging-presentation.md', 'research/results/remaining-work-v1/README.md', 'research/results/remaining-work-v1/offline.json', 'research/results/encoder-v2/final-policy.json'],
    'Live selected model: runs/encoder-flower-balanced-v2, version 500, research/results/encoder-v2/final-policy.json. Offline startup evidence is separate: encoder-isolated-offline-three-v1. Experimental scope detection is not deployed and fresh supported-query coverage remains untested.');
}

const candidate = path.join(buildDir, 'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
await fs.writeFile(path.join(buildDir, 'presentation-proto.json'), JSON.stringify(deck.toProto()));
const stageFinal = path.join(buildDir, 'final-output', 'validated.pptx');
await fs.mkdir(path.dirname(stageFinal), { recursive: true });
const finalizer = await finalizePresentation({
  workspaceDir: buildDir, candidatePath: candidate, finalPath: stageFinal,
  pythonExecutable: path.join(runtime, 'python/python.exe'),
  integrityValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit'],
  explicitTotalSlideCount: 6,
  requiredNativeTableOwnerSlides: [], requiredNativeChartOwnerSlides: [4, 5],
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: 'design', families: [FONT] }, verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, 'validation.json'),
});
const finalPath = path.join(outputDir, 'fedlora-edge-demo.pptx');
await fs.copyFile(stageFinal, finalPath);
await fs.unlink(stageFinal);
// Render the finalized package, including its native charts, rather than only the builder state.
const finalized = await PresentationFile.importPptx(await FileBlob.load(finalPath));
for (const [i, s] of finalized.slides.items.entries()) {
  const stem = `slide-${String(i + 1).padStart(2, '0')}`;
  const png = await finalized.export({ slide: s, format: 'png', scale: 1.5 });
  await fs.writeFile(path.join(buildDir, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  await fs.writeFile(path.join(buildDir, `${stem}.layout.json`), new Uint8Array(await (await s.export({ format: 'layout' })).arrayBuffer()));
}
const manifest = {
  deck: 'fedlora-edge-demo.pptx', generated_utc: new Date().toISOString(), slide_count: 6,
  generator: 'tools/build_judging_deck.mjs', generator_sha256: createHash('sha256').update(await fs.readFile(fileURLToPath(import.meta.url))).digest('hex'),
  package_version: '2.8.106', palette_source: 'research/results/remaining-work-v1/comparison.png',
  source_sha256: sourceHashes,
  identities: { confirmation_run: confirmation.run_id, confirmation_lock: confirmation.lock_hash, confirmation_head: confirmation.head_hash, pre_confirmation_commit: '7413e6b', matrix_plan: matrix.plan_hash, recovery_run: recovery.run_id, recovery_config: recovery.config_hash },
  chart_values: { imbalance_means: imbalance.map(x => x.mean), imbalance_sample_sd: imbalance.map(x => x.sample_std), included: included.values, omitted: omitted.values, recovery_accepted: recovery.accepted_counts },
  slides: noteIndex.map(({ slide, timing, sources, identity }) => ({ slide, timing, sources, identity })),
  limitations: ['33 matrix cells use sequential in-memory aggregation, not Flower runtime.', 'Development results use reused validation.', 'Recovery and offline gates used one physical laptop.', 'The frozen-MPNet head is a separately identified architecture variant, not LoRA.', 'No new evaluation, threshold tuning or model update occurs during deck generation.'],
};
await fs.writeFile(path.join(outputDir, 'source-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
await fs.writeFile(path.join(buildDir, 'notes.json'), JSON.stringify(noteIndex, null, 2));
await fs.writeFile(path.join(ROOT, 'runs/presentation-build/latest.json'), JSON.stringify({ build_dir: buildDir, final_path: finalPath, slides: 6, validation: finalizer }, null, 2));
console.log(JSON.stringify({ finalPath, buildDir, slides: finalized.slides.items.length, charts: 3 }, null, 2));
