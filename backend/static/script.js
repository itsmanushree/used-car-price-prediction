const API = "/api";

const el = (id) => document.getElementById(id);

const brandSel = el("brand");
const modelSel = el("model");
const variantSel = el("variant");
const ownerSel = el("owner");
const citySel = el("city");
const colorSel = el("color");
const yearInput = el("year");
const kmInput = el("km");
const predictBtn = el("predictBtn");
const errorMsg = el("errorMsg");
const specStrip = el("specStrip");

function fillSelect(select, items, placeholder) {
  select.innerHTML = "";
  const ph = document.createElement("option");
  ph.textContent = placeholder;
  ph.value = "";
  ph.disabled = true;
  ph.selected = true;
  select.appendChild(ph);
  for (const item of items) {
    const opt = document.createElement("option");
    if (typeof item === "object") {
      opt.value = item.value;
      opt.textContent = item.label;
    } else {
      opt.value = item;
      opt.textContent = titleCase(item);
    }
    select.appendChild(opt);
  }
  select.disabled = false;
}

function titleCase(str) {
  return String(str).replace(/\w\S*/g, (t) => t[0].toUpperCase() + t.slice(1));
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error((await res.json()).detail || "Request failed");
  return res.json();
}

function checkFormReady() {
  const ready =
    variantSel.value &&
    ownerSel.value &&
    citySel.value &&
    colorSel.value &&
    yearInput.value &&
    kmInput.value !== "";
  predictBtn.disabled = !ready;
}

async function init() {
  try {
    const [{ brands }, { owner_types }, { cities }, { colors }] = await Promise.all([
      getJSON(`${API}/brands`),
      getJSON(`${API}/owner-types`),
      getJSON(`${API}/cities`),
      getJSON(`${API}/colors`),
    ]);
    fillSelect(brandSel, brands, "Select brand");
    fillSelect(ownerSel, owner_types, "Select ownership");
    fillSelect(citySel, cities, "Select city");
    fillSelect(colorSel, colors, "Select colour");
  } catch (e) {
    showError("Could not load the form. Please refresh the page.");
  }
}

brandSel.addEventListener("change", async () => {
  modelSel.disabled = true;
  variantSel.disabled = true;
  specStrip.hidden = true;
  fillSelect(modelSel, [], "Loading…");
  modelSel.disabled = true;
  try {
    const { models } = await getJSON(`${API}/models?brand=${encodeURIComponent(brandSel.value)}`);
    fillSelect(modelSel, models, "Select model");
  } catch (e) {
    showError("Could not load models for this brand.");
  }
  checkFormReady();
});

modelSel.addEventListener("change", async () => {
  variantSel.disabled = true;
  specStrip.hidden = true;
  fillSelect(variantSel, [], "Loading…");
  variantSel.disabled = true;
  try {
    const { variants } = await getJSON(`${API}/variants?model=${encodeURIComponent(modelSel.value)}`);
    fillSelect(variantSel, variants, "Select variant");
  } catch (e) {
    showError("Could not load variants for this model.");
  }
  checkFormReady();
});

variantSel.addEventListener("change", async () => {
  try {
    const d = await getJSON(`${API}/variant-details?variant_name=${encodeURIComponent(variantSel.value)}`);
    el("specFuel").textContent = titleCase(d.fuel_type);
    el("specTransmission").textContent = titleCase(d.transmission);
    el("specBody").textContent = titleCase(d.body_type);
    el("specEngine").textContent = `${Math.round(d.displacement_cc)} cc`;
    el("specPower").textContent = `${Math.round(d.max_power_bhp)} bhp`;
    el("specMileage").textContent = `${d.mileage_kmpl.toFixed(1)} km/l`;
    specStrip.hidden = false;
  } catch (e) {
    specStrip.hidden = true;
  }
  checkFormReady();
});

[ownerSel, citySel, colorSel, yearInput, kmInput].forEach((node) =>
  node.addEventListener("input", checkFormReady)
);

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = false;
}

function hideError() {
  errorMsg.hidden = true;
}

function animateOdometer(price) {
  const digitsEl = el("odometerDigits");
  const caption = el("odometerCaption");
  const detail = el("resultDetail");
  const gaugeFill = el("gaugeFill");

  const target = Math.round(price);
  const duration = 900;
  const start = performance.now();

  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const value = Math.round(target * eased);
    digitsEl.textContent = value.toLocaleString("en-IN");
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  const pct = Math.min(1, target / 5000000); // gauge maxes out around 50L
  const circumference = 327;
  gaugeFill.style.strokeDashoffset = String(circumference * (1 - pct));

  caption.textContent = "estimated resale price";
  detail.hidden = false;
  el("resultVariant").textContent = titleCase(variantSel.value);

  const lakh = target / 100000;
  el("resultLakh").textContent =
    lakh >= 1 ? `₹${lakh.toFixed(2)} Lakh` : `₹${target.toLocaleString("en-IN")}`;
}

predictBtn.addEventListener("click", async () => {
  hideError();
  predictBtn.disabled = true;
  predictBtn.textContent = "Calculating…";

  const payload = {
    variant_name: variantSel.value,
    km_driven: Number(kmInput.value),
    model_year: Number(yearInput.value),
    owner_type: ownerSel.value,
    city_name: citySel.value,
    color: colorSel.value,
  };

  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction failed");
    }
    const data = await res.json();
    animateOdometer(data.predicted_price);
  } catch (e) {
    showError(e.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Get valuation";
    checkFormReady();
  }
});

init();
