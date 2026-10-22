const cases = {
  "cardiomegaly": {
    src: "../assets/videos/cardiomegaly_to_normal.mp4",
    fps: 10
  },
  "lung-opacity": {
    src: "../assets/videos/lung_opacity_to_normal.mp4",
    fps: 10
  },
  "oct": {
    src: "../assets/videos/oct_counterfactual.mp4",
    fps: 10
  }
};

const video = document.getElementById("trajectoryVideo");
const scrubber = document.getElementById("scrubber");
const frameLabel = document.getElementById("frameLabel");
const timeLabel = document.getElementById("timeLabel");
const playPause = document.getElementById("playPause");
const prevFrame = document.getElementById("prevFrame");
const nextFrame = document.getElementById("nextFrame");
const caseSelect = document.getElementById("caseSelect");

let fps = 10;
let userScrubbing = false;

function loadCase(key) {
  const selected = cases[key];
  fps = selected.fps;
  video.pause();
  playPause.textContent = "Play";
  video.src = selected.src;
  video.load();
}

function updateReadout() {
  if (!Number.isFinite(video.duration) || video.duration <= 0) return;
  if (!userScrubbing) {
    scrubber.value = Math.round((video.currentTime / video.duration) * 1000);
  }
  const frame = Math.round(video.currentTime * fps);
  frameLabel.textContent = `Frame ${frame}`;
  timeLabel.textContent = `${video.currentTime.toFixed(2)} s / ${video.duration.toFixed(2)} s`;
}

scrubber.addEventListener("input", () => {
  userScrubbing = true;
  if (Number.isFinite(video.duration) && video.duration > 0) {
    video.currentTime = (Number(scrubber.value) / 1000) * video.duration;
  }
});

scrubber.addEventListener("change", () => {
  userScrubbing = false;
  updateReadout();
});

video.addEventListener("loadedmetadata", updateReadout);
video.addEventListener("timeupdate", updateReadout);
video.addEventListener("seeked", updateReadout);
video.addEventListener("ended", () => {
  playPause.textContent = "Play";
});

playPause.addEventListener("click", async () => {
  if (video.paused) {
    await video.play();
    playPause.textContent = "Pause";
  } else {
    video.pause();
    playPause.textContent = "Play";
  }
});

function stepFrames(delta) {
  video.pause();
  playPause.textContent = "Play";
  const target = Math.min(
    Math.max(video.currentTime + delta / fps, 0),
    Number.isFinite(video.duration) ? video.duration : Infinity
  );
  video.currentTime = target;
}

prevFrame.addEventListener("click", () => stepFrames(-1));
nextFrame.addEventListener("click", () => stepFrames(1));

caseSelect.addEventListener("change", () => {
  loadCase(caseSelect.value);
  const params = new URLSearchParams(window.location.search);
  params.set("case", caseSelect.value);
  history.replaceState(null, "", `${location.pathname}?${params.toString()}`);
});

const params = new URLSearchParams(window.location.search);
const initialCase = cases[params.get("case")] ? params.get("case") : "cardiomegaly";
caseSelect.value = initialCase;
loadCase(initialCase);
