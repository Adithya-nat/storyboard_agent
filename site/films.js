(async () => {
  "use strict";

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fineHover = matchMedia("(hover: hover) and (pointer: fine)").matches;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const fmt = (t) => { const s = Math.max(0, Math.floor(t || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };

  const svg = (body, fill) => `<svg viewBox="0 0 24 24" aria-hidden="true" ${fill ? 'fill="currentColor"' : 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'}>${body}</svg>`;
  const I = {
    play: svg('<path d="M7 4.6v14.8a1 1 0 0 0 1.52.85l11.9-7.4a1 1 0 0 0 0-1.7L8.52 3.75A1 1 0 0 0 7 4.6Z"/>', true),
    pause: svg('<rect x="6" y="4" width="4" height="16" rx="1.2"/><rect x="14" y="4" width="4" height="16" rx="1.2"/>', true),
    info: svg('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>'),
    down: svg('<path d="m6 9 6 6 6-6"/>'),
    left: svg('<path d="m15 18-6-6 6-6"/>'),
    right: svg('<path d="m9 18 6-6-6-6"/>'),
    back: svg('<path d="M19 12H5M12 19l-7-7 7-7"/>'),
    close: svg('<path d="M18 6 6 18M6 6l12 12"/>'),
    on: svg('<path d="M11 4.7a.7.7 0 0 0-1.2-.5L6.4 7.6A1.4 1.4 0 0 1 5.4 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.4a1.4 1.4 0 0 1 1 .4l3.4 3.4a.7.7 0 0 0 1.2-.5Z"/><path d="M16 9a5 5 0 0 1 0 6M19.4 18.4a9 9 0 0 0 0-12.8"/>'),
    off: svg('<path d="M11 4.7a.7.7 0 0 0-1.2-.5L6.4 7.6A1.4 1.4 0 0 1 5.4 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.4a1.4 1.4 0 0 1 1 .4l3.4 3.4a.7.7 0 0 0 1.2-.5Z"/><path d="m22 9-6 6M16 9l6 6"/>'),
    rew: svg('<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><text x="12" y="15.5" font-size="7" font-weight="700" text-anchor="middle" fill="currentColor" stroke="none" font-family="system-ui">10</text>'),
    ff: svg('<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><text x="12" y="15.5" font-size="7" font-weight="700" text-anchor="middle" fill="currentColor" stroke="none" font-family="system-ui">10</text>'),
    full: svg('<path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/>'),
    exit: svg('<path d="M8 3v3a2 2 0 0 1-2 2H3M21 8h-3a2 2 0 0 1-2-2V3M3 16h3a2 2 0 0 1 2 2v3M16 21v-3a2 2 0 0 1 2-2h3"/>'),
  };

  // Real runs: cost is what Google billed our key; "made" is wall-clock render time.
  const FILMS = [
    {
      id: "grand-theft-auto-chennai",
      title: "Grand Theft Auto Chennai",
      native: "மீட்டர் போடு ராஜா",
      lang: "ta", langName: "Tamil",
      genre: "Crime comedy", style: "Dialogue", styleNote: "Characters speak on camera, lip-synced",
      length: "1:00", scenes: 5, made: 31, cost: 12.34,
      glow: "rgba(245, 165, 40, 0.5)",
      logline: "North Chennai, one rainy night. A cocky auto driver and his sharp-eyed partner lift a briefcase of gold from harbour boss Dass, splash his white silk shirt on the way out, and lead his black SUV and the city police on a chase through the metro pillars and the bazaar lanes.",
      prompt: "Grand Theft Auto Chennai Official Trailer",
    },
    {
      id: "irumbukai-mayavi",
      title: "Irumbukai Mayavi",
      native: "இரும்புக்கை மாயாவி",
      lang: "ta", langName: "Tamil",
      genre: "Spy action", style: "Mixed", styleNote: "A narrator and on-camera dialogue",
      length: "0:56", scenes: 12, made: 22, cost: 10.69,
      glow: "rgba(242, 140, 60, 0.5)",
      logline: "Chennai, 1972. Chola-era jewels vanish from a museum, a bank vault melts without a scratch, and every crime scene bears the same mark. Nizharpadai calls in its strangest agent: a man with a steel hand who turns invisible when electricity runs through him.",
      prompt: `Chennai, 1972. A priceless collection of Chola-era jewels is being displayed at the Government Museum in Egmore. One night, the electricity suddenly fails. Within minutes, masked thieves overpower the guards, break into the exhibition hall, and disappear with the treasures.

The next morning, an even stranger robbery takes place at a heavily guarded bank in Madurai. The thieves melt through a reinforced vault using mysterious beams projected from their fingertips. No explosives, no broken locks, and almost no evidence.

When a rare bronze sculpture disappears from Thanjavur, the intelligence agency Nizharpadai realizes these aren't ordinary robberies. They call upon their most unusual agent, Irumbukkai Mayavi.

Mayavi possesses a remarkable ability. When electricity passes through his body, he becomes invisible, leaving only his artificial steel hand visible.

Investigating the three robberies, Mayavi discovers a strange symbol at every crime scene. It belongs to ACHAM, a secret criminal network believed to operate internationally.

His next clue leads him to a private art auction in Mahabalipuram. Mayavi secretly guards the venue using his invisibility, but the criminals strike again, stealing a priceless painting right under his watch.

Determined to uncover their identity, Mayavi follows the thieves to an abandoned warehouse near Chennai harbour.

Inside, he discovers the stolen treasures, powerful experimental weapons, and plans for robberies across the world.

But suddenly, the warehouse doors slam shut. The criminals have spotted his floating steel hand!

As the guards surround him, Mayavi triggers an electrical panel, vanishes, and strikes from the darkness. He defeats the guards and recovers the stolen treasures.

But the mastermind has escaped, leaving behind a message:

"Mayavi, we've only just begun."

And beneath it, a photograph of their next target — a priceless diamond in London.

Create a proper 60s content in tamil please.`,
    },
    {
      id: "escape-the-anaconda",
      title: "Escape the Anaconda",
      lang: "en", langName: "English",
      genre: "Survival thriller", style: "Dialogue", styleNote: "Characters speak on camera, lip-synced",
      length: "1:00", scenes: 5, made: 26, cost: 9.67,
      glow: "rgba(150, 175, 80, 0.45)",
      logline: "Working alone on a flooded Amazon riverbank, a surveyor steps on what he takes for a root. It moves. Now a giant anaconda sits between him and dry ground, and every ripple gives him away.",
      prompt: "a tense scene depicting a man being chased by an anaconda in the dense forest of amazon among the river banks.",
    },
    {
      id: "battle-sativa",
      title: "Battle Sativa",
      lang: "en", langName: "English",
      genre: "Comedy", style: "Dialogue", styleNote: "Characters speak on camera, lip-synced",
      length: "1:00", scenes: 5, made: 24, cost: 12.35,
      glow: "rgba(90, 200, 150, 0.42)",
      logline: "California's mightiest superhero has laser eyes, a cape and a very relaxed attitude to emergencies. His handler has a city-wide alert on her tablet and a hero who would rather finish his snack.",
      prompt: "Comedy sketch on what if a Super Hero turns out to be a weed addict in California?",
    },
  ];
  // Films creators published from the studio, served by the hosted API. The curated films
  // above always show; published ones follow them when the API answers in time.
  const API = String((window.CREATOR_AGENT || {}).apiUrl || "").replace(/\/+$/, "");
  const STYLE_NOTES = {
    dialogue: ["Dialogue", "Characters speak on camera, lip-synced"],
    mixed: ["Mixed", "A narrator and on-camera dialogue"],
    narrated: ["Narrated", "A narrator tells it over moving scenes"],
  };
  const GLOWS = ["rgba(242, 181, 68, 0.45)", "rgba(110, 200, 230, 0.4)", "rgba(230, 110, 140, 0.4)", "rgba(150, 190, 90, 0.42)"];
  const MEDIA = { "poster.jpg": "poster", "frame.jpg": "frame", "preview.mp4": "preview", "film.mp4": "film" };
  async function addPublished(list) {
    if (!API) return;
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 2500);
    try {
      const res = await fetch(`${API}/library/public`, { signal: ctl.signal });
      if (!res.ok) return;
      const taken = new Set(list.map((f) => f.id));
      (await res.json()).forEach((p, i) => {
        const [style, styleNote] = STYLE_NOTES[p.style] || STYLE_NOTES.dialogue;
        const id = `p-${p.id}`;
        if (taken.has(id)) return;
        taken.add(id);
        list.push({
          id, title: p.title, native: p.native_title || undefined,
          lang: p.language === "ta" ? "ta" : "en", langName: p.language === "ta" ? "Tamil" : "English",
          genre: p.genre || "Short film", style, styleNote,
          length: fmt(p.length_seconds), scenes: p.scenes, made: p.made_minutes, cost: p.cost_usd,
          glow: GLOWS[i % GLOWS.length], logline: p.logline || p.tagline || "",
          prompt: p.prompt || "", media: p.media, published: true,
        });
      });
    } catch { /* offline or slow: the curated films still show */ } finally { clearTimeout(timer); }
  }
  await addPublished(FILMS);

  const byId = Object.fromEntries(FILMS.map((f) => [f.id, f]));
  const src = (f, file) => (f.media ? `${API}${f.media[MEDIA[file]]}` : `films/${f.id}/${file}`);
  const langTag = (f) => (f.lang === "ta" ? `<span class="ta" lang="ta">தமிழ்</span>` : f.langName);
  const metaHTML = (f) => `
    <ul class="meta">
      <li class="meta__ok">Made in ${f.made} min</li>
      <li>${f.length}</li>
      <li class="meta__box">HD</li>
      <li class="meta__lang${f.lang === "ta" ? " ta" : ""}"${f.lang === "ta" ? ' lang="ta"' : ""}>${f.lang === "ta" ? "தமிழ்" : "English"}</li>
      <li>${esc(f.genre)}</li>
    </ul>`;

  // ---------- ambient light: a tiny canvas the browser blurs into a glow ----------
  function ambient(canvas, getSource) {
    const ctx = canvas.getContext("2d", { alpha: false });
    canvas.width = 16;
    canvas.height = 28;
    let on = false, raf = 0, last = 0;
    const paint = (s) => {
      try { ctx.drawImage(s, 0, 0, 16, 28); canvas.classList.add("on"); } catch { /* not decodable yet */ }
    };
    const loop = (t) => {
      if (!on) return;
      const s = getSource();
      if (t - last > 90 && s && s.readyState >= 2) { last = t; paint(s); }
      raf = requestAnimationFrame(loop);
    };
    return {
      start() { if (!on) { on = true; raf = requestAnimationFrame(loop); } },
      stop() { on = false; cancelAnimationFrame(raf); },
      still(url) {
        const img = new Image();
        img.onload = () => paint(img);
        img.src = url;
      },
    };
  }

  // ---------- billboard ----------
  const bb = $("[data-billboard]");
  const bv = $("[data-bb-video]", bb), copy = $("[data-bb-copy]", bb), queue = $("[data-bb-queue]", bb);
  const bbMute = $("[data-bb-mute]", bb), screen = $(".bb__screen", bb);
  const bbAmb = ambient($("[data-ambient]", bb), () => bv);
  let cur = 0, muted = true, inView = true, held = false, progRaf = 0;

  queue.innerHTML = FILMS.map((f, i) => `
    <button class="q" type="button" role="tab" data-q="${i}" aria-selected="false" aria-label="${esc(f.title)}">
      <span class="q__bar"><i></i></span>
      <span class="q__name"><span>${esc(f.title)}</span><em>${f.length}</em></span>
    </button>`).join("");
  const qs = $$("[data-q]", queue);

  const syncMute = () => {
    bbMute.innerHTML = muted ? I.off : I.on;
    bbMute.setAttribute("aria-label", muted ? "Turn sound on" : "Turn sound off");
  };
  syncMute();

  const setBar = (k, p) => qs[k].style.setProperty("--p", String(p));
  function progress() {
    if (bv.duration) setBar(cur, Math.min(1, bv.currentTime / bv.duration));
    progRaf = requestAnimationFrame(progress);
  }

  function bbPlay() {
    if (reduce || !inView || held || document.hidden) return;
    bv.muted = muted;
    const p = bv.play();
    if (p) p.catch(() => {
      // Autoplay with sound was refused: fall back to muted.
      if (!bv.muted) { muted = true; syncMute(); bv.muted = true; bv.play().catch(() => {}); }
    });
  }
  function bbPause() { bv.pause(); }

  function feature(i) {
    cur = (i + FILMS.length) % FILMS.length;
    const f = FILMS[cur];
    bb.style.setProperty("--glow", f.glow);
    copy.innerHTML = `
      <p class="cue"><span class="cue__tri" aria-hidden="true"></span>Premiere ${String(cur + 1).padStart(2, "0")} · ${esc(f.genre)}</p>
      <h2 class="bb__title">${esc(f.title)}</h2>
      ${f.native ? `<p class="bb__native" lang="ta">${esc(f.native)}</p>` : ""}
      ${metaHTML(f)}
      <p class="bb__log">${esc(f.logline)}</p>
      <div class="bb__ctas">
        <button class="btn btn--light btn--icon" type="button" data-play="${f.id}">${I.play}Play</button>
        <button class="btn btn--glass btn--icon" type="button" data-more="${f.id}">${I.info}More info</button>
      </div>`;
    qs.forEach((q, k) => { q.setAttribute("aria-selected", String(k === cur)); setBar(k, k < cur ? 1 : 0); });
    bv.poster = src(f, "frame.jpg");
    bv.src = src(f, "preview.mp4");
    if (reduce) bbAmb.still(src(f, "frame.jpg"));
    else bbPlay();
  }

  bv.addEventListener("playing", () => { bbAmb.start(); cancelAnimationFrame(progRaf); progRaf = requestAnimationFrame(progress); });
  bv.addEventListener("pause", () => { bbAmb.stop(); cancelAnimationFrame(progRaf); });
  bv.addEventListener("ended", () => feature(cur + 1));
  queue.addEventListener("click", (e) => { const q = e.target.closest("[data-q]"); if (q) feature(Number(q.dataset.q)); });
  bbMute.addEventListener("click", () => { muted = !muted; syncMute(); bv.muted = muted; if (bv.paused) bbPlay(); });
  $("[data-bb-open]", bb).addEventListener("click", () => openTheater(FILMS[cur].id));

  new IntersectionObserver(([e]) => { inView = e.isIntersecting; inView ? bbPlay() : bbPause(); }, { threshold: 0.3 }).observe(bb);
  document.addEventListener("visibilitychange", () => (document.hidden ? bbPause() : bbPlay()));
  const hold = (on) => { held = on; on ? bbPause() : bbPlay(); };

  // Screen tilts toward the pointer.
  if (fineHover && !reduce) {
    let raf = 0;
    bb.addEventListener("pointermove", (e) => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const r = screen.getBoundingClientRect();
        const x = (e.clientX - (r.left + r.width / 2)) / innerWidth;
        const y = (e.clientY - (r.top + r.height / 2)) / innerHeight;
        screen.style.setProperty("--ry", `${(x * 14).toFixed(2)}deg`);
        screen.style.setProperty("--rx", `${(-y * 10).toFixed(2)}deg`);
      });
    });
    bb.addEventListener("pointerleave", () => { screen.style.setProperty("--rx", "0deg"); screen.style.setProperty("--ry", "0deg"); });
  }

  // ---------- premieres row ----------
  const row = $("[data-row]");
  row.style.scrollSnapType = "none"; // else the browser re-snaps to the "Your film" card after the insert
  row.insertAdjacentHTML("afterbegin", FILMS.map((f, i) => `
    <li class="slot" style="--glow:${f.glow}">
      <span class="slot__n" aria-hidden="true">${i + 1}</span>
      <article class="card" data-card="${f.id}">
        <img src="${src(f, "poster.jpg")}" alt="Poster for ${esc(f.title)}" loading="lazy" decoding="async" width="720" height="1290">
        <video muted loop playsinline preload="none" aria-hidden="true"></video>
        <button class="card__hit" type="button" data-more="${f.id}" aria-label="More info: ${esc(f.title)}"></button>
        <div class="card__info">
          <div class="card__btns">
            <button class="round round--light" type="button" data-play="${f.id}" aria-label="Play ${esc(f.title)}">${I.play}</button>
            <button class="round" type="button" data-more="${f.id}" aria-label="More info: ${esc(f.title)}">${I.down}</button>
          </div>
          <h3>${esc(f.title)}</h3>
          <p class="card__meta"><span class="meta__ok">${f.made} min render</span><span>${f.length}</span><span class="meta__box">HD</span></p>
          <p class="card__tags">${esc(f.genre)} · ${langTag(f)}</p>
        </div>
      </article>
    </li>`).join(""));
  row.scrollLeft = 0;
  requestAnimationFrame(() => (row.style.scrollSnapType = ""));
  $("[data-film-count]").textContent = `${FILMS.length} films · all made from one prompt`;

  // Hovering a poster plays its preview, like a streaming row.
  if (fineHover && !reduce) {
    $$("[data-card]", row).forEach((card) => {
      const v = $("video", card), f = byId[card.dataset.card];
      let t = 0;
      const start = () => {
        clearTimeout(t);
        t = setTimeout(() => {
          if (!v.src) v.src = src(f, "preview.mp4");
          v.play().then(() => card.classList.add("is-previewing")).catch(() => {});
        }, 450);
      };
      const stop = () => { clearTimeout(t); card.classList.remove("is-previewing"); v.pause(); };
      card.parentElement.addEventListener("pointerenter", start);
      card.parentElement.addEventListener("pointerleave", stop);
      card.addEventListener("focusin", start);
      card.addEventListener("focusout", (e) => { if (!card.contains(e.relatedTarget)) stop(); });
    });
  }

  const prev = $("[data-row-prev]"), next = $("[data-row-next]");
  prev.innerHTML = I.left;
  next.innerHTML = I.right;
  const page = (dir) => row.scrollBy({ left: dir * row.clientWidth * 0.75, behavior: reduce ? "auto" : "smooth" });
  prev.addEventListener("click", () => page(-1));
  next.addEventListener("click", () => page(1));
  const arrows = () => {
    prev.disabled = row.scrollLeft < 4;
    next.disabled = row.scrollLeft + row.clientWidth >= row.scrollWidth - 4;
  };
  row.addEventListener("scroll", arrows, { passive: true });
  addEventListener("resize", arrows);
  arrows();

  // ---------- prompts ----------
  const lines = $("[data-lines]");
  lines.innerHTML = FILMS.filter((f) => f.prompt).map((f, i) => {
    const words = f.prompt.trim().split(/\s+/).length;
    const long = words > 60;
    return `
      <article class="line" style="--d:${i};--glow:${f.glow}">
        <p class="line__label"><span class="dot" aria-hidden="true"></span>Prompt<em>${words} words</em></p>
        <blockquote class="line__text${long ? " is-long" : ""}">${esc(f.prompt)}${long ? "" : '<span class="caret" aria-hidden="true"></span>'}</blockquote>
        ${long ? `<button class="line__more" type="button" data-more="${f.id}">Read the full prompt</button>` : ""}
        <p class="line__became" aria-hidden="true">became</p>
        <button class="line__film" type="button" data-more="${f.id}">
          <img src="${src(f, "poster.jpg")}" alt="" loading="lazy" decoding="async">
          <span><b>${esc(f.title)}</b><i>${langTag(f)} · ${esc(f.style)} · ${f.length}</i></span>
        </button>
        <dl class="line__stats">
          <div><dt>Rendered in</dt><dd data-count="${f.made}" data-fmt="min">${f.made} min</dd></div>
          <div><dt>Scenes</dt><dd data-count="${f.scenes}">${f.scenes}</dd></div>
          <div><dt>Google bill</dt><dd data-count="${f.cost}" data-fmt="usd">$${f.cost.toFixed(2)}</dd></div>
        </dl>
      </article>`;
  }).join("");

  const countUp = (dd) => {
    const to = Number(dd.dataset.count), kind = dd.dataset.fmt, t0 = performance.now(), dur = 1100;
    const show = (v) => (dd.textContent = kind === "usd" ? `$${v.toFixed(2)}` : kind === "min" ? `${Math.round(v)} min` : String(Math.round(v)));
    const step = (t) => {
      const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      show(to * e);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  const lineIO = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (!e.isIntersecting) return;
    e.target.classList.add("on");
    if (!reduce) $$("[data-count]", e.target).forEach(countUp);
    lineIO.unobserve(e.target);
  }), { threshold: 0.2 });
  $$(".line", lines).forEach((l) => lineIO.observe(l));

  // ---------- more info ----------
  const info = $("[data-info]"), iv = $("[data-info-video]", info), ibody = $("[data-info-body]", info);
  const iMute = $("[data-info-mute]", info);
  let infoMuted = true;
  $("[data-info-close]", info).innerHTML = I.close;
  const syncInfoMute = () => {
    iMute.innerHTML = infoMuted ? I.off : I.on;
    iMute.setAttribute("aria-label", infoMuted ? "Turn sound on" : "Turn sound off");
  };

  function openInfo(id) {
    const f = byId[id];
    if (!f) return;
    info.style.setProperty("--glow", f.glow);
    infoMuted = true;
    syncInfoMute();
    iv.muted = true;
    iv.poster = src(f, "frame.jpg");
    iv.src = src(f, "preview.mp4");
    const others = FILMS.filter((o) => o.id !== id);
    ibody.innerHTML = `
      <p class="cue"><span class="cue__tri" aria-hidden="true"></span>${esc(f.genre)}</p>
      <h2 class="info__title" id="info-title">${esc(f.title)}</h2>
      ${f.native ? `<p class="info__native" lang="ta">${esc(f.native)}</p>` : ""}
      ${metaHTML(f)}
      <div class="info__ctas"><button class="btn btn--light btn--icon" type="button" data-play="${f.id}">${I.play}Play</button></div>
      <p class="info__log">${esc(f.logline)}</p>
      ${f.prompt ? `<div class="info__prompt"><p class="line__label">The prompt that made it</p><p>${esc(f.prompt)}</p></div>` : ""}
      <dl class="info__facts">
        <div><dt>Language</dt><dd>${f.lang === "ta" ? "Tamil, dialogue and narration" : "English"}</dd></div>
        <div><dt>Style</dt><dd>${esc(f.style)}: ${esc(f.styleNote.toLowerCase())}</dd></div>
        <div><dt>Scenes</dt><dd>${f.scenes}, every one filmed as AI video</dd></div>
        <div><dt>Rendered in</dt><dd>${f.made} minute${f.made === 1 ? "" : "s"}</dd></div>
        <div><dt>Google bill</dt><dd>$${f.cost.toFixed(2)} on ${f.published ? "the creator's own" : "our own"} Gemini key</dd></div>
      </dl>
      <div class="info__more">
        <p class="line__label">More films</p>
        <div class="info__thumbs">${others.map((o) => `
          <button type="button" data-more="${o.id}"><img src="${src(o, "poster.jpg")}" alt="" loading="lazy">
            <span><b>${esc(o.title)}</b><i>${esc(o.genre)} · ${o.length}</i></span></button>`).join("")}
        </div>
      </div>`;
    if (!info.open) { info.showModal(); document.documentElement.classList.add("no-scroll"); hold(true); }
    info.scrollTop = 0;
    $("[data-play]", ibody).focus({ preventScroll: true });
    if (!reduce) iv.play().catch(() => {});
  }
  iMute.addEventListener("click", () => { infoMuted = !infoMuted; iv.muted = infoMuted; syncInfoMute(); if (iv.paused) iv.play().catch(() => {}); });
  $("[data-info-close]", info).addEventListener("click", () => info.close());
  info.addEventListener("click", (e) => { if (e.target === info) info.close(); });
  info.addEventListener("close", () => {
    iv.pause();
    iv.removeAttribute("src");
    iv.load();
    if (!theater.open) { document.documentElement.classList.remove("no-scroll"); hold(false); }
  });

  // ---------- theater ----------
  const theater = $("[data-theater]"), tv = $("[data-theater-video]", theater);
  const thAmb = ambient($("[data-theater-ambient]", theater), () => tv);
  const ctl = Object.fromEntries($$("[data-ctl]", theater).map((b) => [b.dataset.ctl, b]));
  const fill = $("[data-scrub-fill]", theater), buf = $("[data-scrub-buf]", theater), knob = $("[data-scrub-knob]", theater);
  const range = $("[data-scrub-input]", theater), time = $("[data-theater-time]", theater), flash = $("[data-theater-flash]", theater);
  const nextBox = $("[data-next]", theater);
  let film = null, idleT = 0, nextT = 0, dragging = false, lastPointer = "mouse", tRaf = 0;

  $("[data-theater-close]", theater).innerHTML = I.back;
  ctl.back.innerHTML = I.rew;
  ctl.fwd.innerHTML = I.ff;
  const syncPlay = () => {
    ctl.play.innerHTML = tv.paused ? I.play : I.pause;
    ctl.play.setAttribute("aria-label", tv.paused ? "Play" : "Pause");
  };
  const syncSound = () => {
    ctl.mute.innerHTML = tv.muted ? I.off : I.on;
    ctl.mute.setAttribute("aria-label", tv.muted ? "Unmute" : "Mute");
  };
  const syncFull = () => {
    const fs = Boolean(document.fullscreenElement);
    ctl.full.innerHTML = fs ? I.exit : I.full;
    ctl.full.setAttribute("aria-label", fs ? "Exit full screen" : "Full screen");
  };
  syncFull();

  function paintTime() {
    const d = tv.duration || 0, p = d ? tv.currentTime / d : 0;
    fill.style.width = `${p * 100}%`;
    knob.style.left = `${p * 100}%`;
    if (!dragging) range.value = String(Math.round(p * 1000));
    range.setAttribute("aria-valuetext", `${fmt(tv.currentTime)} of ${fmt(d)}`);
    time.textContent = `${fmt(tv.currentTime)} / ${fmt(Math.round(d))}`;
    if (tv.buffered.length && d) buf.style.width = `${(tv.buffered.end(tv.buffered.length - 1) / d) * 100}%`;
  }
  const tick = () => { paintTime(); tRaf = requestAnimationFrame(tick); };

  function wake() {
    theater.classList.remove("is-idle");
    clearTimeout(idleT);
    if (!tv.paused) idleT = setTimeout(() => theater.classList.add("is-idle"), 2600);
  }
  function pulse(icon) {
    if (reduce) return;
    flash.innerHTML = icon;
    flash.classList.remove("go");
    void flash.offsetWidth;
    flash.classList.add("go");
  }
  function toggle() {
    if (tv.paused) { tv.play().catch(() => {}); pulse(I.play); } else { tv.pause(); pulse(I.pause); }
  }
  const seek = (dt) => { tv.currentTime = Math.min(Math.max(0, tv.currentTime + dt), tv.duration || 0); paintTime(); wake(); };

  function hideNext() { clearInterval(nextT); nextBox.hidden = true; }
  function showNext() {
    const n = FILMS[(FILMS.indexOf(film) + 1) % FILMS.length];
    $("[data-next-img]", nextBox).src = src(n, "poster.jpg");
    $("[data-next-title]", nextBox).textContent = n.title;
    $("[data-next-meta]", nextBox).textContent = `${n.genre} · ${n.length}`;
    let left = 6;
    const count = $("[data-next-count]", nextBox);
    count.textContent = String(left);
    nextBox.hidden = false;
    theater.classList.remove("is-idle");
    $("[data-next-go]", nextBox).onclick = () => openTheater(n.id);
    clearInterval(nextT);
    if (reduce) { $(".next__label", nextBox).textContent = "Up next"; return; }
    nextT = setInterval(() => {
      left -= 1;
      count.textContent = String(left);
      if (left <= 0) openTheater(n.id);
    }, 1000);
  }

  function openTheater(id) {
    const f = byId[id];
    if (!f) return;
    film = f;
    if (info.open) info.close();
    hideNext();
    theater.style.setProperty("--glow", f.glow);
    $("[data-theater-title]", theater).textContent = f.title;
    $("[data-theater-meta]", theater).innerHTML = `${langTag(f)} · ${esc(f.genre)} · ${f.length}`;
    tv.poster = src(f, "poster.jpg");
    tv.src = src(f, "film.mp4");
    tv.muted = false;
    syncSound();
    paintTime();
    if (!theater.open) {
      theater.showModal();
      document.documentElement.classList.add("no-scroll");
      hold(true);
    }
    ctl.play.focus({ preventScroll: true });
    tv.play().catch(() => { tv.muted = true; syncSound(); tv.play().catch(syncPlay); });
    if (reduce) thAmb.still(src(f, "frame.jpg"));
    wake();
  }

  tv.addEventListener("play", () => { syncPlay(); hideNext(); if (!reduce) thAmb.start(); cancelAnimationFrame(tRaf); tRaf = requestAnimationFrame(tick); wake(); });
  tv.addEventListener("pause", () => { syncPlay(); thAmb.stop(); cancelAnimationFrame(tRaf); paintTime(); wake(); });
  tv.addEventListener("loadedmetadata", paintTime);
  tv.addEventListener("progress", paintTime);
  tv.addEventListener("volumechange", syncSound);
  tv.addEventListener("ended", showNext);
  tv.addEventListener("pointerdown", (e) => (lastPointer = e.pointerType));
  tv.addEventListener("click", () => {
    if (lastPointer !== "mouse" && theater.classList.contains("is-idle")) { wake(); return; }
    toggle();
  });
  tv.addEventListener("dblclick", () => ctl.full.click());

  ctl.play.addEventListener("click", toggle);
  ctl.back.addEventListener("click", () => seek(-10));
  ctl.fwd.addEventListener("click", () => seek(10));
  ctl.mute.addEventListener("click", () => { tv.muted = !tv.muted; });
  ctl.full.addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else if (theater.requestFullscreen) theater.requestFullscreen().catch(() => {});
    else if (tv.webkitEnterFullscreen) tv.webkitEnterFullscreen();
  });
  document.addEventListener("fullscreenchange", syncFull);
  if (!document.fullscreenEnabled && !tv.webkitEnterFullscreen) ctl.full.hidden = true;

  range.addEventListener("input", () => {
    dragging = true;
    if (tv.duration) tv.currentTime = (Number(range.value) / 1000) * tv.duration;
    paintTime();
    wake();
  });
  range.addEventListener("change", () => { dragging = false; });

  $("[data-next-cancel]", theater).addEventListener("click", () => { hideNext(); ctl.play.focus(); });
  $("[data-theater-close]", theater).addEventListener("click", () => theater.close());
  ["pointermove", "pointerdown", "keydown"].forEach((ev) => theater.addEventListener(ev, wake));
  theater.addEventListener("keydown", (e) => {
    const inButton = e.target.closest("button") && (e.key === " " || e.key === "Enter");
    const inRange = e.target === range && (e.key === "ArrowLeft" || e.key === "ArrowRight");
    if (inButton || inRange || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    const act = { " ": toggle, k: toggle, arrowleft: () => seek(-5), arrowright: () => seek(5), j: () => seek(-10), l: () => seek(10), m: () => (tv.muted = !tv.muted), f: () => ctl.full.click() }[k];
    if (act) { e.preventDefault(); act(); }
  });
  theater.addEventListener("close", () => {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    tv.pause();
    tv.removeAttribute("src");
    tv.load();
    thAmb.stop();
    hideNext();
    clearTimeout(idleT);
    theater.classList.remove("is-idle");
    document.documentElement.classList.remove("no-scroll");
    hold(false);
  });

  // ---------- one delegate for every Play / More info button ----------
  document.addEventListener("click", (e) => {
    const p = e.target.closest("[data-play]");
    if (p) { openTheater(p.dataset.play); return; }
    const m = e.target.closest("[data-more]");
    if (m) openInfo(m.dataset.more);
  });

  // Deep links: films.html#battle-sativa opens that film's details.
  const hash = location.hash.slice(1);
  feature(byId[hash] ? FILMS.indexOf(byId[hash]) : 0);
  if (byId[hash]) openInfo(hash);
})();
