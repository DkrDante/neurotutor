const SVG_NS = "http://www.w3.org/2000/svg";
const STATE_COLORS = {
  Focused: "#1d4ed8", Overloaded: "#b91c1c", Confused: "#c2410c",
  Fatigued: "#6d28d9", Engaged: "#15803d",
};

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

// Shared axis/frame drawing for the bar charts below — same convention as
// analytics.js's chartFrame, duplicated here since this page loads standalone.
function chartFrame(svg, width, height, padding) {
  svg.innerHTML = "";
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  svg.appendChild(svgEl("line", {
    x1: padding.left, y1: height - padding.bottom, x2: width - padding.right, y2: height - padding.bottom,
    stroke: "#c9c4bb", "stroke-width": 1,
  }));
  svg.appendChild(svgEl("line", {
    x1: padding.left, y1: padding.top, x2: padding.left, y2: height - padding.bottom,
    stroke: "#c9c4bb", "stroke-width": 1,
  }));
  return { plotWidth, plotHeight };
}

function linePath(points) {
  return points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
}

function renderHistoryChart(history) {
  const svg = document.getElementById("history-chart");
  const width = 640, height = 260, padding = { left: 44, right: 20, top: 20, bottom: 32 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);
  document.getElementById("history-status").textContent = history
    ? `${history.length} epochs — real per-epoch numbers from the run that produced this checkpoint.`
    : "No training history recorded for this checkpoint — retrain with `--save-history` to generate one.";
  if (!history || history.length === 0) return;

  const maxLoss = Math.max(...history.map((h) => h.train_loss), 1e-9);
  const xFor = (i) => padding.left + (i / Math.max(1, history.length - 1)) * plotWidth;

  // Val accuracy/F1 are already 0-1 and share a real y-axis; train loss has
  // no fixed range, so it's normalized against its own max PURELY for this
  // chart's y-position — its real value is what the point labels show.
  const accuracyPoints = history.map((h, i) => ({ x: xFor(i), y: padding.top + (1 - h.val_accuracy) * plotHeight }));
  const f1Points = history.map((h, i) => ({ x: xFor(i), y: padding.top + (1 - h.val_f1) * plotHeight }));
  const lossPoints = history.map((h, i) => ({ x: xFor(i), y: padding.top + (1 - h.train_loss / maxLoss) * plotHeight }));

  svg.appendChild(svgEl("path", { d: linePath(lossPoints), fill: "none", stroke: "#b58863", "stroke-width": 2, "stroke-dasharray": "4,3" }));
  svg.appendChild(svgEl("path", { d: linePath(f1Points), fill: "none", stroke: "#3f7fbf", "stroke-width": 1.5, "stroke-dasharray": "2,2" }));
  svg.appendChild(svgEl("path", { d: linePath(accuracyPoints), fill: "none", stroke: "#4a7c2f", "stroke-width": 2.5 }));

  svg.appendChild(svgEl("text", { x: padding.left, y: 12, class: "chart-legend", fill: "#4a7c2f" }))
    .textContent = `Val accuracy (final: ${(history[history.length - 1].val_accuracy * 100).toFixed(1)}%)`;
  svg.appendChild(svgEl("text", { x: padding.left + 220, y: 12, class: "chart-legend", fill: "#3f7fbf" }))
    .textContent = "Val F1";
  svg.appendChild(svgEl("text", { x: padding.left + 280, y: 12, class: "chart-legend", fill: "#b58863" }))
    .textContent = `Train loss (start: ${history[0].train_loss.toFixed(3)}, end: ${history[history.length - 1].train_loss.toFixed(3)})`;
  svg.appendChild(svgEl("text", { x: padding.left, y: height - 8, class: "chart-axis-label" })).textContent = `Epoch 1`;
  svg.appendChild(svgEl("text", {
    x: width - padding.right, y: height - 8, class: "chart-axis-label", "text-anchor": "end",
  })).textContent = `Epoch ${history.length}`;
}

function renderBaselinesChart(baselines, fusedAccuracy, fusedF1) {
  const svg = document.getElementById("baselines-chart");
  const width = 640, height = 200, padding = { left: 40, right: 20, top: 16, bottom: 40 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);

  const rows = [...baselines, { name: "fused (this checkpoint)", test_accuracy: fusedAccuracy, test_f1: fusedF1 }];
  const barWidth = plotWidth / rows.length;
  const colors = ["#b91c1c", "#b58863", "#4a7c2f"];

  rows.forEach((row, i) => {
    const barHeight = row.test_accuracy * plotHeight;
    const x = padding.left + i * barWidth + barWidth * 0.15;
    const y = padding.top + plotHeight - barHeight;
    svg.appendChild(svgEl("rect", { x, y, width: barWidth * 0.7, height: barHeight, fill: colors[i] || "#7a726a" }));
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: padding.top + plotHeight + 16, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = row.name.length > 14 ? row.name.slice(0, 13) + "…" : row.name;
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: y - 4, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = `${(row.test_accuracy * 100).toFixed(1)}%`;
  });

  const bodyEl = document.getElementById("baselines-body");
  bodyEl.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of [row.name, `${(row.test_accuracy * 100).toFixed(2)}%`, row.test_f1 !== undefined ? row.test_f1.toFixed(4) : "—"]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    bodyEl.appendChild(tr);
  }
}

const BEHAVIOR_FEATURE_LABELS = {
  correct: "correct", time_to_move: "time to move", eval_loss: "eval loss", puzzle_rating: "puzzle rating",
};

function renderImportanceBarChart(svgId, rows, labelFor) {
  const svg = document.getElementById(svgId);
  const width = 640, height = svgId === "eeg-importance-chart" ? 180 : 160;
  const padding = { left: 40, right: 20, top: 16, bottom: 40 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);
  if (rows.length === 0) return;

  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.importance)), 1e-9);
  const barWidth = plotWidth / rows.length;
  const zeroY = padding.top + plotHeight / 2;

  rows.forEach((row, i) => {
    const barHeight = (Math.abs(row.importance) / maxAbs) * (plotHeight / 2);
    const x = padding.left + i * barWidth + barWidth * 0.15;
    const y = row.importance >= 0 ? zeroY - barHeight : zeroY;
    svg.appendChild(svgEl("rect", {
      x, y, width: barWidth * 0.7, height: Math.max(1, barHeight),
      fill: row.importance >= 0 ? "#4a7c2f" : "#b3261e",
    }));
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: padding.top + plotHeight + 16, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = labelFor(row);
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: row.importance >= 0 ? y - 4 : y + barHeight + 12,
      class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = row.importance.toFixed(3);
  });
  svg.appendChild(svgEl("line", { x1: padding.left, y1: zeroY, x2: width - padding.right, y2: zeroY, stroke: "#c9c4bb", "stroke-width": 1 }));
}

function renderFeatureImportance(importance) {
  const grid = document.getElementById("importance-summary");
  grid.innerHTML = "";
  grid.append(statCard(`${(importance.baseline_accuracy * 100).toFixed(1)}%`, "Baseline accuracy (unshuffled)"));

  renderImportanceBarChart("behavior-importance-chart", importance.behavior_importance, (r) => BEHAVIOR_FEATURE_LABELS[r.feature] || r.feature);
  renderImportanceBarChart("eeg-importance-chart", importance.eeg_channel_importance, (r) => `Ch${r.channel + 1}`);
}

function statCard(value, label) {
  const card = document.createElement("div");
  card.className = "stat-card";
  const valueEl = document.createElement("div");
  valueEl.className = "stat-card-value";
  valueEl.textContent = value;
  const labelEl = document.createElement("div");
  labelEl.className = "stat-card-label";
  labelEl.textContent = label;
  card.append(valueEl, labelEl);
  return card;
}

function fmt(n, digits = 4) {
  return typeof n === "number" ? n.toFixed(digits) : "N/A";
}

function vectorBlock(values, digits = 4) {
  const pre = document.createElement("pre");
  pre.className = "trace-vector";
  pre.textContent = values.map((v) => fmt(v, digits)).join(", ");
  return pre;
}

function matrixBlock(rows, digits = 4) {
  const pre = document.createElement("pre");
  pre.className = "trace-vector";
  pre.textContent = rows.map((row) => "[" + row.map((v) => fmt(v, digits)).join(", ") + "]").join("\n");
  return pre;
}

// A vector rendered as a horizontal bar strip — one bar per dimension,
// green for positive, red for negative, height proportional to |value|
// relative to the largest magnitude in this specific vector. Lets a reader
// SEE that these are real, varied, signed activations rather than reading
// 32 comma-separated numbers.
function vectorBarChart(values, height = 90) {
  const width = Math.max(320, values.length * 18);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, class: "chart-svg", style: "height:" + height + "px;" });
  const maxAbs = Math.max(...values.map((v) => Math.abs(v)), 1e-9);
  const barWidth = width / values.length;
  const midY = height / 2;
  values.forEach((v, i) => {
    const barHeight = (Math.abs(v) / maxAbs) * (height / 2 - 6);
    const x = i * barWidth + barWidth * 0.15;
    const y = v >= 0 ? midY - barHeight : midY;
    svg.appendChild(svgEl("rect", {
      x, y, width: barWidth * 0.7, height: Math.max(1, barHeight),
      fill: v >= 0 ? "#4a7c2f" : "#b3261e",
    }));
  });
  svg.appendChild(svgEl("line", { x1: 0, y1: midY, x2: width, y2: midY, stroke: "#c9c4bb", "stroke-width": 1 }));
  return svg;
}

// Per-channel activation magnitude (L2 norm of each channel's hidden
// vector) for a GCN layer's output — bar chart, one bar per EEG channel.
function renderChannelActivationChart(svgId, rows) {
  const svg = document.getElementById(svgId);
  const width = 640, height = 180, padding = { left: 40, right: 20, top: 16, bottom: 28 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);
  const norms = rows.map((row) => Math.sqrt(row.reduce((sum, v) => sum + v * v, 0)));
  const maxNorm = Math.max(...norms, 1e-9);
  const barWidth = plotWidth / norms.length;
  norms.forEach((norm, i) => {
    const barHeight = (norm / maxNorm) * plotHeight;
    const x = padding.left + i * barWidth + barWidth * 0.15;
    const y = padding.top + plotHeight - barHeight;
    svg.appendChild(svgEl("rect", { x, y, width: barWidth * 0.7, height: barHeight, fill: "#3f7fbf" }));
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: padding.top + plotHeight + 16, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = `Ch${i}`;
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: y - 4, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = norm.toFixed(2);
  });
}

// The GCN's learned channel-adjacency as an actual graph diagram: EEG
// channels arranged on a circle, edges drawn between every pair with
// opacity/thickness scaled by how far the learned weight DEVIATES from the
// uniform baseline (1/n) — not by the matrix's own max. Normalizing by a
// matrix's own max would stretch even a perfectly flat/uniform matrix (e.g.
// gcn2's, softmax of an all-zero-initialized parameter) into what looks like
// a fully "strong" graph, hiding the exact thing this diagram exists to
// show. Against the shared 1/n baseline, a flat matrix renders as uniformly
// faint edges — an honest "no learned structure" — while a genuinely
// differentiated matrix visibly stands out against it.
function renderAdjacencyGraph(svgId, captionId, matrix) {
  const svg = document.getElementById(svgId);
  svg.innerHTML = "";
  const n = matrix.length;
  const baseline = 1 / n;
  const size = 300, center = size / 2, radius = size * 0.34, nodeRadius = 15;
  const positions = Array.from({ length: n }, (_, i) => {
    const angle = (2 * Math.PI * i) / n - Math.PI / 2;
    return { x: center + radius * Math.cos(angle), y: center + radius * Math.sin(angle) };
  });

  // Deviation from baseline, scaled against the largest possible deviation
  // (a weight of 1.0, i.e. all attention on one channel) — a FIXED scale
  // shared by every call, so gcn1's and gcn2's diagrams are directly
  // comparable rather than each auto-stretched to fill its own range.
  const maxPossibleDeviation = 1 - baseline;
  const deviationStrength = (w) => Math.min(1, Math.abs(w - baseline) / maxPossibleDeviation);

  let minWeight = Infinity, maxWeight = -Infinity;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const weight = (matrix[i][j] + matrix[j][i]) / 2;
      minWeight = Math.min(minWeight, weight);
      maxWeight = Math.max(maxWeight, weight);
      const strength = deviationStrength(weight);
      svg.appendChild(svgEl("line", {
        x1: positions[i].x, y1: positions[i].y, x2: positions[j].x, y2: positions[j].y,
        stroke: "#4a7c2f", "stroke-width": (0.5 + strength * 3).toFixed(2),
        "stroke-opacity": (0.06 + strength * 0.85).toFixed(2),
      }));
    }
  }

  positions.forEach((pos, i) => {
    const strength = deviationStrength(matrix[i][i]);
    svg.appendChild(svgEl("circle", {
      cx: pos.x, cy: pos.y, r: nodeRadius,
      fill: `rgba(181, 136, 99, ${(0.2 + strength * 0.7).toFixed(2)})`,
      stroke: "#8a6d4f", "stroke-width": 1.5,
    }));
    const label = svgEl("text", {
      x: pos.x, y: pos.y + 4, "text-anchor": "middle", class: "chart-axis-label", fill: "#2b2320",
    });
    label.textContent = `Ch${i}`;
    svg.appendChild(label);
  });

  const captionEl = document.getElementById(captionId);
  if (captionEl) {
    captionEl.textContent =
      `Edge weights range ${minWeight.toFixed(4)}–${maxWeight.toFixed(4)} ` +
      `(uniform/no-structure baseline = ${baseline.toFixed(4)}).`;
  }
}

function renderSoftmaxChart(probs) {
  const svg = document.getElementById("softmax-chart");
  const width = 640, height = 200, padding = { left: 40, right: 20, top: 16, bottom: 40 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);
  const entries = Object.entries(probs);
  const barWidth = plotWidth / entries.length;
  entries.forEach(([state, p], i) => {
    const barHeight = p * plotHeight;
    const x = padding.left + i * barWidth + barWidth * 0.15;
    const y = padding.top + plotHeight - barHeight;
    svg.appendChild(svgEl("rect", {
      x, y, width: barWidth * 0.7, height: barHeight, fill: STATE_COLORS[state] || "#7a726a",
    }));
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: padding.top + plotHeight + 16, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = state;
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: y - 4, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = `${(p * 100).toFixed(0)}%`;
  });
}

function renderWeightsChart(weights) {
  const svg = document.getElementById("weights-chart");
  const width = 640, height = 260, padding = { left: 44, right: 16, top: 16, bottom: 90 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);
  const meanAbs = weights.map((w) => Math.max(Math.abs(w.mean), w.std)); // std dominates for zero-mean weights
  const maxVal = Math.max(...meanAbs, 1e-9);
  const barWidth = plotWidth / weights.length;
  weights.forEach((w, i) => {
    const barHeight = (meanAbs[i] / maxVal) * plotHeight;
    const x = padding.left + i * barWidth + barWidth * 0.15;
    const y = padding.top + plotHeight - barHeight;
    svg.appendChild(svgEl("rect", { x, y, width: barWidth * 0.7, height: Math.max(1, barHeight), fill: "#b58863" }));
    const label = svgEl("text", {
      x: x + barWidth * 0.35, y: padding.top + plotHeight + 12, class: "chart-axis-label",
      "text-anchor": "end", transform: `rotate(-55, ${x + barWidth * 0.35}, ${padding.top + plotHeight + 12})`,
    });
    label.textContent = `${i + 1}. ${w.name.split(".").slice(-2).join(".")}`;
    svg.appendChild(label);
  });
}

function renderMetricsChart(metrics) {
  const svg = document.getElementById("metrics-chart");
  const width = 640, height = 240, padding = { left: 40, right: 20, top: 16, bottom: 56 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);
  const series = ["precision", "recall", "f1"];
  const seriesColors = { precision: "#3f7fbf", recall: "#b58863", f1: "#4a7c2f" };
  const groupWidth = plotWidth / metrics.state_order.length;
  const barWidth = (groupWidth * 0.75) / series.length;

  metrics.state_order.forEach((state, groupIndex) => {
    const m = metrics.per_class[state];
    series.forEach((key, seriesIndex) => {
      const value = m[key];
      const barHeight = value * plotHeight;
      const x = padding.left + groupIndex * groupWidth + groupWidth * 0.125 + seriesIndex * barWidth;
      const y = padding.top + plotHeight - barHeight;
      svg.appendChild(svgEl("rect", { x, y, width: barWidth * 0.9, height: barHeight, fill: seriesColors[key] }));
    });
    svg.appendChild(svgEl("text", {
      x: padding.left + groupIndex * groupWidth + groupWidth / 2, y: padding.top + plotHeight + 16,
      class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = state;
  });

  series.forEach((key, i) => {
    svg.appendChild(svgEl("rect", { x: padding.left + i * 90, y: height - 14, width: 10, height: 10, fill: seriesColors[key] }));
    svg.appendChild(svgEl("text", {
      x: padding.left + i * 90 + 14, y: height - 5, class: "chart-legend", fill: seriesColors[key],
    })).textContent = key;
  });
}

function renderWeights(weights) {
  document.getElementById("weights-status").textContent =
    `${weights.length} learnable parameter tensors, straight from the trained checkpoint's state_dict.`;

  const totalParams = weights.reduce((sum, w) => sum + w.num_params, 0);
  const summaryEl = document.getElementById("weights-summary");
  summaryEl.innerHTML = "";
  summaryEl.append(
    statCard(weights.length, "Parameter tensors"),
    statCard(totalParams.toLocaleString(), "Total learned scalars"),
  );

  renderWeightsChart(weights);

  const bodyEl = document.getElementById("weights-body");
  bodyEl.innerHTML = "";
  weights.forEach((w, i) => {
    const row = document.createElement("tr");
    const cells = [
      `${i + 1}. ${w.name}`, `[${w.shape.join(", ")}]`, w.num_params,
      fmt(w.mean), fmt(w.std), fmt(w.min), fmt(w.max),
    ];
    for (const value of cells) {
      const td = document.createElement("td");
      td.textContent = value;
      row.appendChild(td);
    }
    const sampleTd = document.createElement("td");
    sampleTd.className = "calc-mono";
    sampleTd.textContent = w.sample_values.map((v) => fmt(v)).join(", ");
    row.appendChild(sampleTd);
    bodyEl.appendChild(row);
  });
}

function renderMetrics(metrics) {
  document.getElementById("metrics-status").textContent =
    `Computed independently on the held-out "${metrics.split}" split (${metrics.num_samples} samples never seen during training) — a separate check against the checkpoint, not the number it was selected by.`;

  const summaryEl = document.getElementById("metrics-summary");
  summaryEl.innerHTML = "";
  summaryEl.append(
    statCard(`${(metrics.accuracy * 100).toFixed(1)}%`, "Accuracy"),
    statCard(fmt(metrics.macro_precision), "Macro precision"),
    statCard(fmt(metrics.macro_recall), "Macro recall"),
    statCard(fmt(metrics.macro_f1), "Macro F1"),
    statCard(fmt(metrics.weighted_f1), "Weighted F1"),
    statCard(metrics.roc_auc_macro !== null ? fmt(metrics.roc_auc_macro) : "N/A", "Macro ROC-AUC (OvR)"),
  );

  renderMetricsChart(metrics);

  const perClassBody = document.getElementById("per-class-body");
  perClassBody.innerHTML = "";
  for (const state of metrics.state_order) {
    const m = metrics.per_class[state];
    const row = document.createElement("tr");
    for (const value of [
      state, fmt(m.precision), fmt(m.recall), fmt(m.f1),
      m.roc_auc !== null ? fmt(m.roc_auc) : "N/A", m.support,
    ]) {
      const td = document.createElement("td");
      td.textContent = value;
      row.appendChild(td);
    }
    perClassBody.appendChild(row);
  }

  const confusionTable = document.getElementById("confusion-table");
  confusionTable.innerHTML = "";
  const headRow = document.createElement("tr");
  headRow.appendChild(document.createElement("th"));
  for (const state of metrics.state_order) {
    const th = document.createElement("th");
    th.textContent = state;
    headRow.appendChild(th);
  }
  confusionTable.appendChild(headRow);
  const maxCount = Math.max(...metrics.confusion_matrix.flat(), 1);
  metrics.confusion_matrix.forEach((row, i) => {
    const tr = document.createElement("tr");
    const rowHead = document.createElement("th");
    rowHead.textContent = metrics.state_order[i];
    tr.appendChild(rowHead);
    row.forEach((count, j) => {
      const td = document.createElement("td");
      td.textContent = count;
      // Heatmap shading: darker on the diagonal (correct predictions) means
      // a well-behaved classifier; darker off-diagonal cells flag a real
      // confusion pair, both visible at a glance instead of scanning numbers.
      const intensity = count / maxCount;
      const isDiagonal = i === j;
      td.style.backgroundColor = count === 0
        ? "transparent"
        : isDiagonal
          ? `rgba(74, 124, 47, ${0.15 + intensity * 0.55})`
          : `rgba(179, 38, 30, ${0.15 + intensity * 0.55})`;
      tr.appendChild(td);
    });
    confusionTable.appendChild(tr);
  });
}

function renderTrace(trace) {
  const outEl = document.getElementById("trace-output");
  outEl.innerHTML = "";

  renderAdjacencyGraph("gcn1-graph", "gcn1-graph-caption", trace.gcn1_adjacency_weights);
  renderAdjacencyGraph("gcn2-graph", "gcn2-graph-caption", trace.gcn2_adjacency_weights);
  renderChannelActivationChart("gcn1-activation-chart", trace.gcn1_output_last_timestep);
  renderChannelActivationChart("gcn2-activation-chart", trace.gcn2_output_last_timestep);
  renderSoftmaxChart(trace.softmax_probabilities);

  const header = document.createElement("p");
  header.className = "calc-result";
  const badge = document.createElement("span");
  badge.className = `calc-pill ${trace.correct ? "calc-pill-correct" : "calc-pill-wrong"}`;
  badge.textContent = trace.correct ? "Correct" : "Incorrect";
  header.append(
    `Sample #${trace.sample_index} (${trace.split} split) — true label `,
    document.createElement("strong"),
  );
  header.querySelector("strong").textContent = trace.true_label;
  header.append(", model predicted ");
  const predStrong = document.createElement("strong");
  predStrong.textContent = trace.predicted_label;
  header.append(predStrong, " ");
  header.append(badge);
  outEl.appendChild(header);

  const steps = [
    ["Input — last EEG timestep (channels × bands)", matrixBlock(trace.input_eeg_last_timestep), null],
    ["Input — last behavior timestep", vectorBlock(trace.input_behavior_last_timestep), null],
    ["GCN layer 1 output (per-channel, last timestep)", matrixBlock(trace.gcn1_output_last_timestep), null],
    ["GCN layer 2 output (per-channel, last timestep)", matrixBlock(trace.gcn2_output_last_timestep), null],
    ["GCN layer 1 learned channel-adjacency (softmax-normalized)", matrixBlock(trace.gcn1_adjacency_weights), null],
    ["GCN layer 2 learned channel-adjacency (softmax-normalized)", matrixBlock(trace.gcn2_adjacency_weights), null],
    ["EEG branch embedding (mean-pooled over channels)", vectorBlock(trace.eeg_embedding), trace.eeg_embedding],
    ["Behavior branch hidden layer", vectorBlock(trace.behavior_hidden), trace.behavior_hidden],
    ["Behavior branch embedding", vectorBlock(trace.behavior_embedding), trace.behavior_embedding],
    ["Fused vector (EEG embedding ++ behavior embedding)", vectorBlock(trace.fused_vector), trace.fused_vector],
    ["LSTM hidden state (last timestep)", vectorBlock(trace.lstm_hidden_state), trace.lstm_hidden_state],
    ["Final logits", vectorBlock(trace.logits), trace.logits],
  ];
  for (const [title, block, barValues] of steps) {
    const wrap = document.createElement("div");
    wrap.className = "trace-step";
    const titleEl = document.createElement("strong");
    titleEl.textContent = title;
    wrap.append(titleEl);
    if (barValues) wrap.appendChild(vectorBarChart(barValues));
    wrap.appendChild(block);
    outEl.appendChild(wrap);
  }

  const probsWrap = document.createElement("div");
  probsWrap.className = "trace-step";
  const probsTitle = document.createElement("strong");
  probsTitle.textContent = "Softmax probabilities (chart above)";
  const probsPre = document.createElement("pre");
  probsPre.className = "trace-vector";
  probsPre.textContent = Object.entries(trace.softmax_probabilities)
    .map(([state, p]) => `${state}: ${fmt(p)}`).join("\n");
  probsWrap.append(probsTitle, probsPre);
  outEl.appendChild(probsWrap);
}

async function loadEvidence(split, sampleIndex) {
  const statusEl = document.getElementById("trace-status");
  try {
    const response = await fetch(`/api/model-evidence?split=${encodeURIComponent(split)}&sample_index=${sampleIndex}`);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${response.status}`);
    }
    const data = await response.json();
    renderWeights(data.weights);
    renderHistoryChart(data.training_history);
    renderMetrics(data.metrics);
    renderBaselinesChart(data.baselines, data.metrics.accuracy, data.metrics.macro_f1);
    renderFeatureImportance(data.feature_importance);
    renderTrace(data.trace);
    statusEl.textContent = `Checkpoint: ${data.checkpoint_path}`;
  } catch (err) {
    statusEl.textContent = `Could not load: ${err.message}`;
    document.getElementById("weights-status").textContent = "Could not load weights.";
    document.getElementById("history-status").textContent = "Could not load training history.";
    document.getElementById("metrics-status").textContent = "Could not load metrics.";
  }
}

document.getElementById("trace-run").addEventListener("click", () => {
  const split = document.getElementById("trace-split").value;
  const sampleIndex = parseInt(document.getElementById("trace-index").value, 10) || 0;
  loadEvidence(split, sampleIndex);
});

loadEvidence("test", 0);
