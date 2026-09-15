const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

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

let allPuzzles = [];

async function loadPuzzles() {
  const statusEl = document.getElementById("puzzles-status");
  try {
    const response = await fetch("/api/puzzles");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    allPuzzles = data.puzzles;
    statusEl.textContent = `${data.total} puzzles, read live from chess_task/puzzle_data/sample_puzzles.csv.`;
    renderSummary(allPuzzles);
    renderRatingChart(allPuzzles);
    renderMotifChart(allPuzzles);
    populateMotifFilter(allPuzzles);
    applyFiltersAndRender();
  } catch (err) {
    statusEl.textContent = `Could not load the puzzle set: ${err.message}`;
  }
}

function renderSummary(puzzles) {
  const handBuilt = puzzles.filter((p) => p.source === "hand-built").length;
  const lichess = puzzles.filter((p) => p.source === "lichess").length;
  const ratings = puzzles.map((p) => p.rating);
  const grid = document.getElementById("puzzles-stat-grid");
  grid.innerHTML = "";
  grid.append(
    statCard(puzzles.length, "Total puzzles"),
    statCard(lichess, "From Lichess puzzle database"),
    statCard(handBuilt, "Hand-built mate patterns"),
    statCard(`${Math.min(...ratings)}–${Math.max(...ratings)}`, "Rating range"),
  );
}

function renderRatingChart(puzzles) {
  const svg = document.getElementById("puzzles-rating-chart");
  const width = 640, height = 220, padding = { left: 40, right: 20, top: 16, bottom: 40 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);
  if (puzzles.length === 0) return;

  const bucketSize = 200;
  const buckets = new Map();
  for (const p of puzzles) {
    const bucketStart = Math.floor(p.rating / bucketSize) * bucketSize;
    buckets.set(bucketStart, (buckets.get(bucketStart) || 0) + 1);
  }
  const bucketStarts = [...buckets.keys()].sort((a, b) => a - b);
  const maxCount = Math.max(...buckets.values());
  const barWidth = plotWidth / bucketStarts.length;

  bucketStarts.forEach((start, i) => {
    const count = buckets.get(start);
    const barHeight = (count / maxCount) * plotHeight;
    const x = padding.left + i * barWidth + barWidth * 0.1;
    const y = padding.top + plotHeight - barHeight;
    svg.appendChild(svgEl("rect", {
      x, y, width: barWidth * 0.8, height: barHeight, fill: "#3f7fbf",
    }));
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.4, y: padding.top + plotHeight + 16, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = start;
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.4, y: y - 4, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = count;
  });
}

const MOTIF_LABELS = {
  smothered: "Smothered", back_rank: "Back-rank", double_check: "Double check",
  discovered_check: "Discovered check", queen_mate: "Queen mate", rook_mate: "Rook mate",
  knight_mate: "Knight mate", bishop_mate: "Bishop mate", pawn_mate: "Pawn mate", other: "Other",
};

function renderMotifChart(puzzles) {
  const svg = document.getElementById("puzzles-motif-chart");
  const width = 640, height = 220, padding = { left: 40, right: 20, top: 16, bottom: 56 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);
  if (puzzles.length === 0) return;

  const counts = new Map();
  for (const p of puzzles) counts.set(p.motif, (counts.get(p.motif) || 0) + 1);
  const motifs = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a));
  const maxCount = Math.max(...counts.values());
  const barWidth = plotWidth / motifs.length;

  motifs.forEach((motif, i) => {
    const count = counts.get(motif);
    const barHeight = (count / maxCount) * plotHeight;
    const x = padding.left + i * barWidth + barWidth * 0.1;
    const y = padding.top + plotHeight - barHeight;
    svg.appendChild(svgEl("rect", { x, y, width: barWidth * 0.8, height: barHeight, fill: "#6d28d9" }));
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.4, y: padding.top + plotHeight + 14, class: "chart-axis-label", "text-anchor": "middle",
      transform: `rotate(-40, ${x + barWidth * 0.4}, ${padding.top + plotHeight + 14})`,
    })).textContent = MOTIF_LABELS[motif] || motif;
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.4, y: y - 4, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = count;
  });
}

function populateMotifFilter(puzzles) {
  const select = document.getElementById("puzzle-motif-filter");
  const motifs = [...new Set(puzzles.map((p) => p.motif))].sort();
  for (const motif of motifs) {
    const option = document.createElement("option");
    option.value = motif;
    option.textContent = MOTIF_LABELS[motif] || motif;
    select.appendChild(option);
  }
}

function applyFiltersAndRender() {
  const search = document.getElementById("puzzle-search").value.trim().toLowerCase();
  const sourceFilter = document.getElementById("puzzle-source-filter").value;
  const motifFilter = document.getElementById("puzzle-motif-filter").value;
  const minRating = parseFloat(document.getElementById("puzzle-min-rating").value);
  const maxRating = parseFloat(document.getElementById("puzzle-max-rating").value);
  const sort = document.getElementById("puzzle-sort").value;

  let filtered = allPuzzles.filter((p) => {
    if (sourceFilter !== "all" && p.source !== sourceFilter) return false;
    if (motifFilter !== "all" && p.motif !== motifFilter) return false;
    if (!Number.isNaN(minRating) && p.rating < minRating) return false;
    if (!Number.isNaN(maxRating) && p.rating > maxRating) return false;
    if (search) {
      const haystack = `${p.puzzle_id} ${p.solution_moves.join(" ")}`.toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    return true;
  });

  filtered = filtered.sort((a, b) => {
    if (sort === "rating-asc") return a.rating - b.rating;
    if (sort === "rating-desc") return b.rating - a.rating;
    return a.puzzle_id.localeCompare(b.puzzle_id);
  });

  document.getElementById("puzzle-filter-count").textContent =
    `Showing ${filtered.length} of ${allPuzzles.length} puzzles.`;
  renderTable(filtered);
}

// Puzzles can run into the hundreds — capped so this stays a page you can
// actually scroll and read rather than a 300-row unrendered wall the browser
// chokes on; the count line above always states the true filtered total.
const MAX_ROWS_RENDERED = 150;

function renderTable(puzzles) {
  const body = document.getElementById("puzzle-table-body");
  body.innerHTML = "";
  const shown = puzzles.slice(0, MAX_ROWS_RENDERED);
  for (const p of shown) {
    const tr = document.createElement("tr");

    const idTd = document.createElement("td");
    idTd.textContent = p.puzzle_id;
    idTd.className = "calc-mono";

    const sourceTd = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `calc-pill ${p.source === "lichess" ? "calc-pill-match" : "calc-pill-diverge"}`;
    badge.textContent = p.source;
    sourceTd.appendChild(badge);

    const motifTd = document.createElement("td");
    motifTd.textContent = MOTIF_LABELS[p.motif] || p.motif;

    const ratingTd = document.createElement("td");
    ratingTd.textContent = p.rating;

    const solutionTd = document.createElement("td");
    solutionTd.className = "calc-mono";
    solutionTd.textContent = p.solution_moves.join(" ");

    const fenTd = document.createElement("td");
    fenTd.className = "calc-mono puzzle-fen-cell";
    fenTd.textContent = p.fen;

    tr.append(idTd, sourceTd, motifTd, ratingTd, solutionTd, fenTd);
    body.appendChild(tr);
  }
  if (puzzles.length > MAX_ROWS_RENDERED) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.className = "calc-table-empty";
    td.textContent = `…and ${puzzles.length - MAX_ROWS_RENDERED} more — narrow your search to see them.`;
    tr.appendChild(td);
    body.appendChild(tr);
  }
}

for (const id of ["puzzle-search", "puzzle-source-filter", "puzzle-motif-filter", "puzzle-min-rating", "puzzle-max-rating", "puzzle-sort"]) {
  document.getElementById(id).addEventListener("input", applyFiltersAndRender);
}

loadPuzzles();
