(() => {
  "use strict";

  const cfg = window.CREATOR_AGENT || {};
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // ---------- links: the app URL comes from the deploy workflow ----------
  const app = (cfg.appUrl || "").replace(/\/$/, "");
  // Log in / Sign up go to the hosted app; before it exists they point at the "opening soon" note.
  const soon = $("#beta-soon");
  $$("[data-app-link]").forEach((a) => {
    if (app) {
      a.href = app + a.dataset.appLink;
      return;
    }
    if (!soon) return;
    a.href = "#beta-soon";
    a.addEventListener("click", () => {
      soon.classList.remove("flash");
      void soon.offsetWidth; // restart the highlight
      soon.classList.add("flash");
    });
  });
  if (!app && soon) soon.hidden = false;

  // ---------- GitHub stars ----------
  fetch("https://api.github.com/repos/Adithya-nat/creator-agent")
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      if (!d || typeof d.stargazers_count !== "number") return;
      const n = d.stargazers_count;
      $("[data-star-count]").textContent = n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
    })
    .catch(() => {});

  // ---------- nav ----------
  const nav = $("[data-nav]");
  const onScroll = () => nav.classList.toggle("is-scrolled", scrollY > 8);
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ---------- run something only while it's on screen ----------
  function whenVisible(el, onEnter, onLeave, threshold = 0.25) {
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => (e.isIntersecting ? onEnter() : onLeave && onLeave())),
      { threshold },
    );
    io.observe(el);
  }

  // ---------- projector beam (WebGL) ----------
  const FRAG = `
    precision mediump float;
    uniform vec2 r; uniform float t; uniform vec2 m; uniform float soft;
    float h(float n){ return fract(sin(n*127.1)*43758.5453); }
    float n1(float x){ float i=floor(x), f=fract(x); return mix(h(i), h(i+1.), f*f*(3.-2.*f)); }
    void main(){
      vec2 uv = gl_FragCoord.xy / r;
      float x = uv.x + (m.x - .5) * .06 * (1. - uv.y);
      float cols = mix(150., 90., soft);
      float c = floor(x * cols);
      float w = fract(x * cols);
      float line = smoothstep(.0, .18, w) * smoothstep(.62, .3, w);
      float lane = n1(c * .37 + t * .05);
      float flick = n1(c * 1.7 + t * (1.2 + h(c) * 2.4));
      float band = n1(uv.y * 3. + c * .21 - t * (.25 + h(c + 3.) * .5));
      float on = step(.55, lane) * smoothstep(.35, .95, flick * band);
      float burst = step(.985, n1(c * 4.1 + floor(t * 2.)));
      float beam = exp(-pow((x - .62) * 2.6, 2.)) + .35;
      float v = line * (on * .75 + burst * 1.1) * beam;
      v *= smoothstep(1.05, .25, uv.y) * smoothstep(-.1, .45, uv.y + .1);
      float hue = h(c + 11.);
      vec3 amber = vec3(.95, .71, .27), cyan = vec3(.43, .9, .98), white = vec3(1., .96, .88);
      vec3 col = mix(amber, cyan, step(.72, hue));
      col = mix(col, white, step(.94, hue));
      float glow = exp(-pow((x - .62) * 1.4, 2.)) * .05 * smoothstep(1., .2, uv.y);
      gl_FragColor = vec4(col * v + amber * glow, 1.);
    }`;

  function beam(canvas) {
    const gl = canvas.getContext("webgl", { antialias: false, alpha: false, powerPreference: "low-power" });
    if (!gl) return;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, "attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}"));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const u = (n) => gl.getUniformLocation(prog, n);
    const ur = u("r"), ut = u("t"), um = u("m"), us = u("soft");
    gl.uniform1f(us, canvas.dataset.beam === "soft" ? 1 : 0);

    const scale = Math.min(1, devicePixelRatio) * 0.6; // soft light, cheap pixels
    const size = () => {
      canvas.width = Math.max(1, Math.round(canvas.clientWidth * scale));
      canvas.height = Math.max(1, Math.round(canvas.clientHeight * scale));
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(ur, canvas.width, canvas.height);
    };
    size();
    addEventListener("resize", size);

    let mx = 0.5, tx = 0.5, raf = 0, running = false, inView = false;
    const start = performance.now() - Math.random() * 20000;
    addEventListener("pointermove", (e) => (tx = e.clientX / innerWidth), { passive: true });
    const frame = (now) => {
      mx += (tx - mx) * 0.04;
      gl.uniform1f(ut, (now - start) / 1000);
      gl.uniform2f(um, mx, 0.5);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      raf = running ? requestAnimationFrame(frame) : 0;
    };
    const play = () => { if (!running && inView && !document.hidden) { running = true; raf = requestAnimationFrame(frame); } };
    const stop = () => { running = false; cancelAnimationFrame(raf); };
    if (reduce) { gl.uniform1f(ut, 12); gl.uniform2f(um, 0.5, 0.5); gl.drawArrays(gl.TRIANGLES, 0, 3); return; }
    whenVisible(canvas, () => { inView = true; play(); }, () => { inView = false; stop(); }, 0);
    document.addEventListener("visibilitychange", () => (document.hidden ? stop() : play()));
  }
  $$("[data-beam]").forEach(beam);

  // ---------- the gate: one line becomes a film ----------
  const IDEAS = [
    ["a lighthouse keeper befriends a storm", "The Keeper and the Storm", "<b>Keeper</b> You again? Every night, you knock."],
    ["two rival tea sellers share one umbrella", "One Umbrella", "<b>Meena</b> Your side is drier. Move over."],
    ["a retired astronaut teaches her grandson to fish", "Low Orbit", "<b>Asha</b> Patience. Up there, I waited months."],
  ];
  const gate = $("[data-gate]");
  if (gate) {
    const steps = Object.fromEntries($$("[data-step]", gate).map((s) => [s.dataset.step, s]));
    const typed = $("[data-typed]", gate), crew = $$("[data-crew]", gate), scene = $("[data-scene]", gate);
    const tc = $("[data-gate-tc]", gate);
    let alive = false, loopId = 0, frames = 0;
    const show = (name) => Object.entries(steps).forEach(([k, el]) => el.classList.toggle("is-on", k === name));
    const tick = setInterval(() => {
      if (!alive) return;
      frames += 1;
      const f = frames % 24, s = Math.floor(frames / 24) % 60, m = Math.floor(frames / 1440) % 60;
      tc.textContent = `00:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}:${String(f).padStart(2, "0")}`;
    }, 1000 / 24);

    async function run(id) {
      for (let i = 0; id === loopId; i = (i + 1) % IDEAS.length) {
        const [idea, title, line] = IDEAS[i];
        show("prompt");
        typed.textContent = "";
        await sleep(500);
        for (const ch of idea) { if (id !== loopId) return; typed.textContent += ch; await sleep(42 + Math.random() * 50); }
        await sleep(900);
        show("crew");
        crew.forEach((li) => li.classList.remove("is-run", "is-done"));
        for (const [k, li] of crew.entries()) {
          if (id !== loopId) return;
          li.classList.add("is-run");
          if (k === 3) for (let n = 1; n <= 3; n++) { scene.textContent = `scene ${n} of 3`; await sleep(650); }
          else await sleep(700);
          li.classList.replace("is-run", "is-done");
        }
        await sleep(350);
        $("[data-film-title]", gate).textContent = title;
        $(".gate__line", gate).innerHTML = line;
        show("ready");
        await sleep(3600);
      }
    }
    if (reduce) {
      typed.textContent = IDEAS[0][0];
      show("ready");
    } else {
      whenVisible(gate, () => { if (!alive) { alive = true; run(++loopId); } }, () => { alive = false; loopId++; });
    }
    void tick;
  }

  // ---------- ticker: duplicate once for a seamless loop ----------
  const track = $("[data-ticker]");
  if (track) track.innerHTML += track.innerHTML.replace(/<span/g, '<span aria-hidden="true"');

  // ---------- headlines light up word by word as they scroll in ----------
  $$("[data-lit]").forEach((h) => {
    const dim = h.querySelector("span");
    if (!dim || reduce) return;
    const words = dim.textContent.split(/(\s+)/);
    dim.innerHTML = words.map((w) => (w.trim() ? `<span class="w">${w}</span>` : w)).join("");
    const ws = $$(".w", dim);
    const update = () => {
      const r = h.getBoundingClientRect();
      const p = Math.min(1, Math.max(0, (innerHeight * 0.85 - r.top) / (innerHeight * 0.55)));
      const lit = Math.round(p * ws.length);
      ws.forEach((w, i) => w.classList.toggle("on", i < lit));
    };
    let on = false;
    whenVisible(h, () => { if (!on) { on = true; addEventListener("scroll", update, { passive: true }); update(); } },
      () => { on = false; removeEventListener("scroll", update); }, 0);
  });

  // ---------- shot cards ----------
  const shots = $("[data-shots]");
  if (shots) {
    const cells = $("[data-cells]", shots);
    cells.innerHTML = "<i></i>".repeat(48);
    const wave = $("[data-wave]", shots);
    wave.innerHTML = Array.from({ length: 36 }, (_, i) =>
      `<i style="--i:${i};--h:${(30 + Math.abs(Math.sin(i * 1.7)) * 70).toFixed(0)}%"></i>`).join("");
    const count = $("[data-shots-count]", shots);
    const fills = $$("[data-fill]", shots);
    const lines = $$(".ln", shots);
    let id = 0;

    async function loop(myId) {
      while (myId === id) {
        fills.forEach((d) => (d.textContent = " "));
        lines.forEach((l) => l.classList.remove("on"));
        $$("i", cells).forEach((c) => c.classList.remove("on"));
        for (const d of fills) {
          d.textContent = "";
          d.classList.add("typing");
          for (const ch of d.dataset.fill) { if (myId !== id) return; d.textContent += ch; await sleep(28); }
          d.classList.remove("typing");
        }
        for (const l of lines) { if (myId !== id) return; l.classList.add("on"); await sleep(650); }
        const order = [...Array(48).keys()];
        for (let k = 0; k < order.length; k++) {
          if (myId !== id) return;
          cells.children[order[k]].classList.add("on");
          count.textContent = String(k + 1);
          await sleep(55);
        }
        await sleep(3200);
      }
    }
    if (reduce) {
      fills.forEach((d) => (d.textContent = d.dataset.fill));
      $$("i", cells).forEach((c) => c.classList.add("on"));
      count.textContent = "48";
      shots.classList.add("is-live");
    } else {
      whenVisible(shots, () => { shots.classList.add("is-live"); loop(++id); }, () => { shots.classList.remove("is-live"); id++; }, 0.15);
    }
  }

  // ---------- key box ----------
  const kb = $("[data-keybox]");
  if (kb) {
    const field = $("[data-key-typed]", kb), lock = $("[data-lock]", kb), checks = $$("[data-check]", kb), ok = $("[data-key-ok]", kb);
    const KEY = "AIza••••••••••••••••••••Xq4w";
    let id = 0;
    async function loop(my) {
      while (my === id) {
        field.textContent = "";
        lock.classList.remove("shut");
        checks.forEach((c) => c.classList.remove("on"));
        ok.classList.remove("on");
        await sleep(600);
        field.textContent = KEY; // a paste, not typing
        await sleep(500);
        lock.classList.add("shut");
        for (const c of checks) { if (my !== id) return; await sleep(260); c.classList.add("on"); }
        ok.classList.add("on");
        await sleep(4200);
      }
    }
    if (reduce) {
      field.textContent = KEY;
      lock.classList.add("shut");
      checks.forEach((c) => c.classList.add("on"));
      ok.classList.add("on");
    } else whenVisible(kb, () => loop(++id), () => id++);
  }

  // ---------- 24-hour clock: a real countdown, sped up so you can see it move ----------
  const clock = $("[data-clock]");
  if (clock) {
    const arc = $("[data-clock-arc]", clock), out = $("[data-clock-time]", clock);
    const total = 24 * 3600, C = 2 * Math.PI * 88;
    arc.style.strokeDasharray = String(C);
    let left = total, timer = 0;
    const draw = () => {
      const h = Math.floor(left / 3600), m = Math.floor((left % 3600) / 60), s = left % 60;
      out.textContent = [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
      arc.style.strokeDashoffset = String(C * (1 - left / total));
    };
    draw();
    if (!reduce) {
      whenVisible(clock, () => {
        if (timer) return;
        timer = setInterval(() => { left = left > 60 ? left - 37 : total; draw(); }, 1000 / 12);
      }, () => { clearInterval(timer); timer = 0; });
    }
  }

  // ---------- terminal ----------
  const term = $("[data-term]");
  if (term) {
    const SCRIPT = [
      ["p", "$ ", "c", "git clone https://github.com/Adithya-nat/creator-agent"],
      ["p", "$ ", "c", "git clone https://github.com/Adithya-nat/creator-agent-web"],
      ["p", "$ ", "c", "cd creator-agent && make setup"],
      ["ok", "✓ ", "", "API, web app and database ready"],
      ["p", "$ ", "c", "echo GEMINI_API_KEY=your-key >> .env && make dev"],
      ["ok", "✓ ", "", "Studio running at http://localhost:3000"],
    ];
    const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    let id = 0;
    async function loop(my) {
      while (my === id) {
        term.innerHTML = "";
        for (const [pc, p, cc, text] of SCRIPT) {
          const line = document.createElement("div");
          line.innerHTML = `<span class="${pc}">${p}</span><span class="${cc}"></span>`;
          term.appendChild(line);
          const target = line.lastChild;
          if (pc === "p") for (const ch of text) { if (my !== id) return; target.textContent += ch; await sleep(18 + Math.random() * 30); }
          else { await sleep(500); target.innerHTML = esc(text); }
          await sleep(350);
        }
        await sleep(5000);
      }
    }
    if (reduce) term.innerHTML = SCRIPT.map(([pc, p, cc, t]) => `<div><span class="${pc}">${p}</span><span class="${cc}">${esc(t)}</span></div>`).join("");
    else whenVisible(term, () => loop(++id), () => id++);
  }
})();
