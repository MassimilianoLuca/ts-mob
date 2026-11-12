"use strict";

(() => {
  const tour = document.getElementById("conditioning");
  const canvas = document.getElementById("tour-canvas");
  const context = canvas.getContext("2d");
  const image = document.getElementById("tour-image");
  const playbackButton = document.getElementById("conditioning-play");
  const buttons = [...tour.querySelectorAll("[data-stage]")];
  const title = document.getElementById("tour-title");
  const description = document.getElementById("tour-description");
  const shares = document.getElementById("tour-shares");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const stages = [
    {
      title: "Start with places",
      description: "Population and 16 POI counts describe both ends of the trip. Haversine distance connects the origin and destination. These features enter every candidate branch.",
      boxes: [[20, 538, 670, 230]],
      paths: [[[134, 540], [134, 450]], [[354, 540], [354, 450]], [[565, 540], [565, 450]]],
    },
    {
      title: "One network, many pairs",
      description: "Each branch combines origin features, destination features and distance. The same feed-forward network scores every pair, sharing its learned parameters.",
      boxes: [[56, 303, 590, 144]],
      paths: [[[134, 395], [134, 350]], [[354, 395], [354, 350]], [[565, 395], [565, 350]], [[134, 302], [134, 263]], [[354, 302], [354, 263]], [[565, 302], [565, 263]]],
    },
    {
      title: "Turn scores into shares",
      description: "Softmax exponentiates each score and divides by the total across destinations. For the middle branch in this example, exp(1.0) ÷ 5.29 ≈ 51.4%.",
      boxes: [[57, 173, 590, 49], [220, 78, 306, 55]],
      paths: [[[134, 264], [134, 223]], [[354, 264], [354, 223]], [[565, 264], [565, 223]], [[354, 173], [354, 136]]],
    },
    {
      title: "Carry geography into time",
      description: "The selected pair’s attractiveness becomes a fixed covariate. TimesFM receives it with the latest 32 flow counts, precipitation, temperature and hour of day.",
      boxes: [[805, 90, 550, 150], [1438, 132, 136, 455]],
      paths: [[[538, 105], [785, 105]], [[1363, 182], [1426, 182]], [[1363, 355], [1426, 355]], [[1363, 552], [1426, 552]]],
    },
    {
      title: "Predict a count",
      description: "Fine-tuned TimesFM combines the temporal history with the geographic context to predict how many people move between this origin and destination in the next interval.",
      boxes: [[967, 751, 529, 191]],
      paths: [[[1498, 601], [1498, 740]], [[1380, 819], [1420, 867], [1460, 830]]],
    },
  ];
  let stage = 0;
  let frame;
  let playback = "idle";
  let observer;
  let startedAt = 0;
  let elapsed = 0;
  const stageDuration = 3600;

  function setStage(index) {
    stage = index;
    title.textContent = stages[index].title;
    description.textContent = stages[index].description;
    shares.hidden = index !== 2;
    buttons.forEach((button, position) => button.setAttribute("aria-pressed", String(position === index)));
  }

  function draw(progress) {
    if (!context) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    context.setTransform(canvas.width / 1600, 0, 0, canvas.height / 1005, 0, 0);
    context.clearRect(0, 0, 1600, 1005);
    const color = stage >= 3 ? "#b52325" : "#1c7180";
    context.lineWidth = 3;
    context.strokeStyle = color;
    context.fillStyle = stage >= 3 ? "#b5232509" : "#1c718009";
    for (const [x, y, w, h] of stages[stage].boxes) {
      context.beginPath();
      context.roundRect(x, y, w, h, 9);
      context.fill();
      context.stroke();
    }
    if (progress === null) return;
    for (const points of stages[stage].paths) {
      context.beginPath();
      points.forEach(([x, y], index) => index === 0 ? context.moveTo(x, y) : context.lineTo(x, y));
      context.strokeStyle = color + "70";
      context.lineWidth = 4;
      context.stroke();
      const position = (progress * 2) % 1;
      const segment = Math.min(Math.floor(position * (points.length - 1)), points.length - 2);
      const t = position * (points.length - 1) - segment;
      const x = points[segment][0] + (points[segment + 1][0] - points[segment][0]) * t;
      const y = points[segment][1] + (points[segment + 1][1] - points[segment][1]) * t;
      context.beginPath();
      context.arc(x, y, 7, 0, 2 * Math.PI);
      context.fillStyle = color;
      context.shadowColor = color;
      context.shadowBlur = 10;
      context.fill();
      context.shadowBlur = 0;
    }
  }

  function setPlayback(state, label) {
    playback = state;
    playbackButton.textContent = label;
  }

  function redraw() {
    draw(playback === "idle" ? null : (elapsed % stageDuration) / stageDuration);
  }

  function pause() {
    cancelAnimationFrame(frame);
    setPlayback("paused", "Resume animation");
  }

  function stop() {
    cancelAnimationFrame(frame);
    elapsed = 0;
    setPlayback("idle", "Replay animation");
  }

  function tick(now) {
    elapsed = Math.max(elapsed, now - startedAt);
    const nextStage = Math.min(Math.floor(elapsed / stageDuration), stages.length - 1);
    if (nextStage !== stage) setStage(nextStage);
    if (elapsed >= stages.length * stageDuration) {
      stop();
      redraw();
      return;
    }
    redraw();
    frame = requestAnimationFrame(tick);
  }

  function play() {
    observer?.disconnect();
    if (reducedMotion.matches) return;
    if (playback === "idle") setStage(0);
    setPlayback("playing", "Pause animation");
    startedAt = performance.now() - elapsed;
    frame = requestAnimationFrame(tick);
  }

  playbackButton.addEventListener("click", () => playback === "playing" ? pause() : play());
  buttons.forEach((button, index) => button.addEventListener("click", () => {
    observer?.disconnect();
    stop();
    setStage(index);
    draw(null);
  }));
  reducedMotion.addEventListener("change", () => {
    playbackButton.hidden = reducedMotion.matches;
    if (!reducedMotion.matches) return;
    observer?.disconnect();
    stop();
    redraw();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && playback === "playing") pause();
  });
  new ResizeObserver(redraw).observe(image);
  image.addEventListener("load", redraw);
  playbackButton.hidden = reducedMotion.matches;
  setStage(0);
  draw(null);
  if ("IntersectionObserver" in window && !reducedMotion.matches) {
    observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting && entry.intersectionRatio >= 0.5)) play();
    }, { threshold: 0.5, rootMargin: "-70px 0px 0px 0px" });
    observer.observe(image);
  }
})();
