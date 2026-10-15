// The data-collection protocol, as a pure function so it can be unit-tested
// (see tests/test_protocol_js.py) and previewed on the setup screen.
//
// The experimental logic, in one place:
//   * each condition induces its intended state through the TASK (puzzle
//     difficulty band, time pressure, a concurrent digit task), not through
//     anything the tutor decides — /collect runs the tutor with the no-op
//     "fixed" policy;
//   * condition order is shuffled per participant with no condition twice in
//     a row, so a condition is never confounded with its position;
//   * fatigue is measured by time-on-task: the last blocks come after a long
//     stretch of continuous play, and are only labelled Fatigued downstream if
//     the participant's own self-report agrees (data_gen/from_recordings.py).
(function (root) {
  "use strict";

  const PUZZLE_RATING_MIN = 604;   // chess_task/puzzle_data/sample_puzzles.csv
  const PUZZLE_RATING_MAX = 2499;

  function hashString(text) {
    let h = 2166136261;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Random order of `items` with no two equal neighbours (retries a bounded
  // number of times, then falls back to a greedy repair).
  function shuffleNoAdjacent(items, rng) {
    for (let attempt = 0; attempt < 200; attempt++) {
      const a = items.slice();
      for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
      }
      if (a.every((v, i) => i === 0 || v !== a[i - 1])) return a;
    }
    const sorted = items.slice().sort();
    const out = [];
    while (sorted.length) {
      const idx = sorted.findIndex((v) => v !== out[out.length - 1]);
      out.push(sorted.splice(idx === -1 ? 0 : idx, 1)[0]);
    }
    return out;
  }

  function clampBand(lo, hi) {
    lo = Math.max(PUZZLE_RATING_MIN, Math.round(lo));
    hi = Math.min(PUZZLE_RATING_MAX, Math.round(hi));
    if (hi - lo < 150) { hi = Math.min(PUZZLE_RATING_MAX, lo + 150); lo = Math.max(PUZZLE_RATING_MIN, hi - 150); }
    return [lo, hi];
  }

  const CONDITION_TEXT = {
    Focused: "Solve these at a comfortable, steady pace. They should feel challenging but fair.",
    Confused: "These are harder than usual and may feel unfamiliar. Take your time to work out what is going on — there is no time limit.",
    Overloaded: "Solve as fast as you can AND keep up with the digit task: press M whenever the digit repeats the previous one.",
    Late_Focused: "Keep going at a steady pace.",
  };

  // The after-block self-report: three plain questions answered with one tap each (no sliders).
  // data_gen/from_recordings.py reads exactly these keys and values for its manipulation checks;
  // tests/test_protocol_js.py keeps the two in step.
  const RATING_QUESTIONS = [
    {
      key: "difficulty", prompt: "How hard were the puzzles?",
      options: [
        { value: 1, label: "Easy" }, { value: 2, label: "About right" },
        { value: 3, label: "Hard" }, { value: 4, label: "Too hard" },
      ],
    },
    {
      key: "mind", prompt: "How did your mind feel?",
      options: [
        { value: "sharp", label: "Sharp and focused", hint: "in the zone" },
        { value: "pushing", label: "Working hard", hint: "but keeping up" },
        { value: "lost", label: "Lost", hint: "no idea what to try" },
        { value: "overwhelmed", label: "Overwhelmed", hint: "too much at once" },
        { value: "drifting", label: "Tired or drifting", hint: "drowsy, mind wandering" },
      ],
    },
    {
      key: "sleepiness", prompt: "How awake were you?",
      options: [
        { value: 1, label: "Wide awake" }, { value: 2, label: "Fairly awake" }, { value: 3, label: "Getting tired" },
        { value: 4, label: "Tired" }, { value: 5, label: "Very sleepy" },
      ],
    },
  ];

  function buildProtocol(opts) {
    const participant = String(opts.participant || "anon");
    const skill = Number(opts.skill) || 1000;
    const quick = !!opts.quick;
    const blockSeconds = quick ? 20 : Math.max(10, Number(opts.blockSeconds) || 240);
    const t = {
      eo: quick ? 10 : 120, ec: quick ? 8 : 60, endBaseline: quick ? 8 : 60,
      practice: quick ? 10 : 90, practiceDual: quick ? 8 : 45, rest: quick ? 3 : 30,
    };

    const bands = {
      Focused: clampBand(skill - 100, skill + 150),
      Confused: clampBand(skill + 300, skill + 600),
      Overloaded: clampBand(skill + 500, skill + 900),
    };
    bands.Late_Focused = bands.Focused;
    bands.Practice = bands.Focused;
    bands.Practice_Dual = bands.Overloaded;

    const steps = [];
    steps.push({
      id: "baseline-eo", type: "baseline", condition: "Baseline_EO", seconds: t.eo, rest: false,
      title: "Resting baseline — eyes open",
      text: "Sit comfortably and still. Look at the cross in the middle of the screen. Try not to blink hard, clench your jaw or move.",
    });
    steps.push({
      id: "baseline-ec", type: "baseline", condition: "Baseline_EC", seconds: t.ec, rest: false,
      title: "Resting baseline — eyes closed",
      text: "Close your eyes and relax. A beep will tell you when to open them.",
    });
    steps.push({
      id: "practice-1", type: "task", condition: "Practice", block: "practice-1", seconds: t.practice, band: bands.Practice,
      dual: false, rated: false, title: "Practice — puzzles",
      text: "A short warm-up so you know the interface. This block is not used for analysis.",
    });
    steps.push({
      id: "practice-2", type: "task", condition: "Practice_Dual", block: "practice-2", seconds: t.practiceDual, band: bands.Practice_Dual,
      dual: true, rated: false, title: "Practice — puzzles + digit task",
      text: CONDITION_TEXT.Overloaded + " This block is not used for analysis.",
    });

    const rng = mulberry32(hashString(participant + "|order"));
    const order = shuffleNoAdjacent(["Focused", "Focused", "Confused", "Confused", "Overloaded", "Overloaded"], rng);
    order.forEach((condition, i) => {
      steps.push({
        id: `b${i + 1}`, type: "task", condition, block: `b${i + 1}`, seconds: blockSeconds, band: bands[condition],
        dual: condition === "Overloaded", rated: true, title: `Block ${i + 1}`, text: CONDITION_TEXT[condition],
      });
    });
    ["Late_Focused", "Late_Focused"].forEach((condition, i) => {
      const n = order.length + i + 1;
      steps.push({
        id: `b${n}`, type: "task", condition, block: `b${n}`, seconds: blockSeconds, band: bands[condition],
        dual: false, rated: true, title: `Block ${n}`, text: CONDITION_TEXT[condition],
      });
    });
    steps.push({
      id: "baseline-eo-end", type: "baseline", condition: "Baseline_EO_end", seconds: t.endBaseline, rest: false,
      title: "Final resting baseline — eyes open",
      text: "Last one. Sit still and look at the cross.",
    });
    steps.forEach((s) => { s.restSeconds = t.rest; });
    return steps;
  }

  function totalSeconds(steps) {
    return steps.reduce((sum, s) => {
      const rated = s.type === "task" && s.rated;
      return sum + s.seconds + (rated ? 20 + s.restSeconds : 0);
    }, 0);
  }

  const api = { buildProtocol, totalSeconds, RATING_QUESTIONS, shuffleNoAdjacent, mulberry32, hashString, clampBand, PUZZLE_RATING_MIN, PUZZLE_RATING_MAX };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.NeuroProtocol = api;
})(typeof window !== "undefined" ? window : this);
