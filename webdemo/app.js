/* Hinglish Turn Detector: browser demo.
   Everything runs locally. Audio is decoded with Web Audio, resampled to
   16 kHz mono, right aligned into an 8 s window, then pushed through an
   int8 ONNX model with onnxruntime-web (WASM). */

(function () {
  'use strict';

  var CONFIG_URL = 'models/config.json';
  var STREAM_STEP_S = 0.24;
  var STREAM_MAX_POINTS = 40;
  var MAX_RECORD_MS = 30000;

  var cfg = null;
  var SR = 16000;
  var WIN = 128000;

  var models = {};          // id -> model config
  var order = [];           // model ids, largest file first
  var sessions = {};        // id -> InferenceSession
  var pending = {};         // id -> Promise
  var activeId = null;
  var threshold = 0.5;

  var clip = null;          // { pcm: Float32Array, duration, source }
  var clipFile = null;      // example filename, when the clip came from a chip
  var lastProb = null;
  var streamGen = 0;
  var curve = null;         // { xs, ys, total }
  var busy = false;

  var EXAMPLES = [
    'complete_hindi_voice.wav',
    'complete_enIN_voice.wav',
    'complete_with_midfiller.wav',
    'incomplete_trailing_aur.wav',
    'incomplete_trailing_filler.wav',
    'incomplete_midsentence_cut.wav'
  ];

  var el = {};
  ['modelSeg', 'thresh', 'threshVal', 'engine', 'engineText', 'recBtn', 'recState',
   'recHint', 'file', 'chips', 'msg', 'verdict', 'verdictText', 'verdictSrc',
   'probNum', 'meter', 'meterFill', 'meterNeedle', 'meterGate', 'gateFlag',
   'sLat', 'sSize', 'sWin', 'sDur', 'modelNote', 'wave', 'chart',
   'streamNote'].forEach(function (k) { el[k] = document.getElementById(k); });

  /* ------------------------------------------------------------- helpers */

  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  function fmtName(file) { return file.replace(/\.wav$/, '').replace(/_/g, ' '); }

  function labelOf(file) { return file.indexOf('incomplete') === 0 ? 'incomplete' : 'complete'; }

  function status(state, text) {
    el.engine.querySelector('.dot').dataset.state = state;
    el.engineText.textContent = text;
  }

  function note(text) { el.msg.textContent = text || ''; }

  // Yield to the event loop between curve points. setTimeout rather than
  // requestAnimationFrame, which stops firing when the tab is not visible and
  // would leave the curve half drawn.
  function yieldToUi() {
    return new Promise(function (r) { setTimeout(r, 0); });
  }

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // hex to rgba, so canvas never has to parse color-mix()
  function alpha(hex, a) {
    var h = hex.replace('#', '');
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
    if (isNaN(n)) return hex;
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }

  function span(cls, text) {
    var s = document.createElement('span');
    s.className = cls;
    s.textContent = text;
    return s;
  }

  /* ------------------------------------------------------------- audio */

  function toMono(buffer) {
    var n = buffer.length;
    var ch = buffer.numberOfChannels;
    if (ch === 1) return buffer.getChannelData(0).slice();
    var out = new Float32Array(n);
    for (var c = 0; c < ch; c++) {
      var d = buffer.getChannelData(c);
      for (var i = 0; i < n; i++) out[i] += d[i];
    }
    for (var j = 0; j < n; j++) out[j] /= ch;
    return out;
  }

  function resample(mono, srIn, srOut) {
    var n = Math.max(1, Math.round(mono.length * srOut / srIn));
    var oc = new OfflineAudioContext(1, n, srOut);
    var src = oc.createBufferSource();
    var b = oc.createBuffer(1, mono.length, srIn);
    b.copyToChannel(mono, 0);
    src.buffer = b;
    src.connect(oc.destination);
    src.start();
    return oc.startRendering().then(function (rendered) {
      return rendered.getChannelData(0).slice();
    });
  }

  // Decode straight into a 16 kHz OfflineAudioContext: one resampling step,
  // and an already 16 kHz wav comes through untouched. Falls back to a plain
  // AudioContext decode plus an OfflineAudioContext resample.
  function decodeToPcm(arrayBuffer) {
    var direct;
    try {
      direct = new OfflineAudioContext(1, 1, SR).decodeAudioData(arrayBuffer.slice(0));
    } catch (e) {
      direct = Promise.reject(e);
    }
    return Promise.resolve(direct).then(toMono).catch(function () {
      var AC = window.AudioContext || window.webkitAudioContext;
      var ac = new AC();
      return ac.decodeAudioData(arrayBuffer.slice(0)).then(function (buf) {
        var mono = toMono(buf);
        var sr = buf.sampleRate;
        try { ac.close(); } catch (e2) { /* already closed */ }
        return sr === SR ? mono : resample(mono, sr, SR);
      });
    });
  }

  // Last WIN samples of pcm[0..end), left padded with zeros when shorter.
  function windowAt(pcm, end) {
    var out = new Float32Array(WIN);
    var start = Math.max(0, end - WIN);
    var seg = pcm.subarray(start, end);
    out.set(seg, WIN - seg.length);
    return out;
  }

  /* ------------------------------------------------------------- onnx */

  function setSegBusy(on) {
    Array.prototype.forEach.call(el.modelSeg.querySelectorAll('input'), function (i) {
      i.disabled = on;
    });
  }

  function getSession(id) {
    if (sessions[id]) return Promise.resolve(sessions[id]);
    if (pending[id]) return pending[id];
    var m = models[id];
    status('load', 'Loading ' + m.tag.toLowerCase() + ' model, ' + m.size_mb.toFixed(1) + ' MB');
    setSegBusy(true);
    pending[id] = ort.InferenceSession.create('models/' + m.file, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    }).then(function (s) {
      sessions[id] = s;
      pending[id] = null;
      setSegBusy(false);
      status('ready', 'Ready');
      return s;
    }).catch(function (err) {
      pending[id] = null;
      setSegBusy(false);
      status('err', 'Model failed to load');
      throw err;
    });
    return pending[id];
  }

  function infer(id, win) {
    return getSession(id).then(function (sess) {
      var feeds = {};
      feeds[sess.inputNames[0]] = new ort.Tensor('float32', win, [1, WIN]);
      var t0 = performance.now();
      return sess.run(feeds).then(function (out) {
        var ms = performance.now() - t0;
        return { p: sigmoid(out[sess.outputNames[0]].data[0]), ms: ms };
      });
    });
  }

  /* ------------------------------------------------------------- readout */

  function renderMeter() {
    el.meterFill.style.width = (lastProb === null ? 0 : lastProb * 100) + '%';
    el.meterNeedle.style.left = (lastProb === null ? 0 : lastProb * 100) + '%';
    el.meterGate.style.left = (threshold * 100) + '%';
    el.meterGate.dataset.flip = threshold > 0.82 ? '1' : '0';
    el.gateFlag.textContent = threshold.toFixed(2);
    el.probNum.textContent = lastProb === null ? '0.000' : lastProb.toFixed(3);
    var v = lastProb === null ? 'none' : (lastProb >= threshold ? 'done' : 'still');
    el.meter.dataset.v = v;
    return v;
  }

  function renderVerdict(flash) {
    var v = renderMeter();
    if (v === 'none') {
      el.verdict.dataset.v = 'none';
      el.verdictText.textContent = 'No audio yet';
      el.verdictSrc.textContent = 'Pick an example, upload a file, or record';
      return;
    }
    var was = el.verdict.dataset.v;
    el.verdict.dataset.v = v;
    el.verdictText.textContent = v === 'done' ? 'Done speaking' : 'Still speaking';
    if (flash || was !== v) {
      el.verdict.classList.remove('flash');
      void el.verdict.offsetWidth;
      el.verdict.classList.add('flash');
    }
  }

  function renderModelStats() {
    var m = models[activeId];
    el.sSize.textContent = m.size_mb.toFixed(1) + ' MB';
    el.sWin.textContent = cfg.window_seconds + ' s @ ' + (SR / 1000) + ' kHz';
    el.modelNote.textContent = m.label + '. ' + (m.params / 1e6).toFixed(2) +
      ' M parameters, int8 quantised. Default threshold ' + m.threshold.toFixed(2) + '.';
  }

  function setThreshold(t, syncSlider) {
    threshold = t;
    el.threshVal.textContent = t.toFixed(2);
    if (syncSlider) el.thresh.value = String(t);
    renderVerdict(false);
    drawChart();
  }

  /* ------------------------------------------------------------- canvas */

  function prep(canvas) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = canvas.clientWidth || 600;
    // Latch the logical height on the first pass. Assigning canvas.height
    // rewrites the height content attribute, so reading that attribute back
    // on the next call would return the already scaled value and the canvas
    // would grow by one dpr factor on every redraw.
    if (!canvas.dataset.h) canvas.dataset.h = canvas.getAttribute('height') || '150';
    var h = parseInt(canvas.dataset.h, 10);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.height = h + 'px';
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }

  function palette() {
    return {
      ink: css('--ink'),
      ink2: css('--ink-2'),
      ink3: css('--ink-3'),
      rule: css('--rule'),
      ruleSoft: css('--rule-soft'),
      sig: css('--sig'),
      warn: css('--warn'),
      well: css('--well') || css('--paper-3')
    };
  }

  function emptyCanvas(canvas, text) {
    var p = prep(canvas);
    p.ctx.fillStyle = palette().ink3;
    p.ctx.font = '11px "Azeret Mono", monospace';
    p.ctx.textAlign = 'center';
    p.ctx.textBaseline = 'middle';
    p.ctx.fillText(text, p.w / 2, p.h / 2);
  }

  function drawWave() {
    if (!clip) { emptyCanvas(el.wave, 'no signal loaded'); return; }
    var p = prep(el.wave);
    var c = p.ctx, W = p.w, H = p.h;
    var col = palette();
    var padL = 10, padR = 10;
    var plotW = W - padL - padR;
    var dur = clip.duration;
    var winS = cfg.window_seconds;
    var t0 = Math.min(0, dur - winS);
    var span2 = dur - t0;
    var X = function (t) { return padL + (t - t0) / span2 * plotW; };
    var mid = (H - 14) / 2 + 2;

    // hatched zero pad, only when the clip is shorter than the window
    if (t0 < 0) {
      var px = X(t0), pw = X(0) - X(t0);
      c.save();
      c.beginPath();
      c.rect(px, 0, pw, H - 14);
      c.clip();
      c.strokeStyle = col.rule;
      c.globalAlpha = 0.55;
      c.lineWidth = 1;
      for (var d = -H; d < pw + H; d += 8) {
        c.beginPath();
        c.moveTo(px + d, H - 14);
        c.lineTo(px + d + H, 0);
        c.stroke();
      }
      c.restore();
      if (pw > 78) {
        c.fillStyle = col.ink3;
        c.font = '9.5px "Azeret Mono", monospace';
        c.textAlign = 'center';
        c.textBaseline = 'top';
        c.fillText('zero pad', px + pw / 2, 9);
      }
    }

    // centre line
    c.strokeStyle = col.ruleSoft;
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(padL, Math.round(mid) + 0.5);
    c.lineTo(W - padR, Math.round(mid) + 0.5);
    c.stroke();

    // waveform, min and max per pixel column
    var pcm = clip.pcm;
    var x0 = X(0);
    var cols = Math.max(1, Math.round(X(dur) - x0));
    var per = pcm.length / cols;
    var amp = mid - 14;
    c.fillStyle = col.sig;
    for (var i = 0; i < cols; i++) {
      var s = Math.floor(i * per), e = Math.min(pcm.length, Math.floor((i + 1) * per));
      var lo = 0, hi = 0;
      for (var k = s; k < e; k++) {
        var v = pcm[k];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      c.fillRect(x0 + i, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
    }

    // analysis window brackets
    var wx0 = X(Math.max(t0, dur - winS)), wx1 = X(dur);
    c.strokeStyle = col.sig;
    c.globalAlpha = 0.55;
    c.beginPath();
    c.moveTo(Math.round(wx0) + 0.5, 3); c.lineTo(Math.round(wx0) + 0.5, H - 16);
    c.moveTo(Math.round(wx1) - 0.5, 3); c.lineTo(Math.round(wx1) - 0.5, H - 16);
    c.moveTo(Math.round(wx0) + 0.5, 3.5); c.lineTo(Math.round(wx0) + 8, 3.5);
    c.moveTo(Math.round(wx1) - 0.5, 3.5); c.lineTo(Math.round(wx1) - 8, 3.5);
    c.stroke();
    c.globalAlpha = 1;

    // second ticks
    c.strokeStyle = col.rule;
    c.fillStyle = col.ink3;
    c.font = '9px "Azeret Mono", monospace';
    c.textAlign = 'center';
    c.textBaseline = 'bottom';
    var wEvery = Math.max(1, Math.ceil(30 / (plotW / span2)));
    for (var t = Math.ceil(t0); t <= Math.floor(dur); t++) {
      var tx = Math.round(X(t)) + 0.5;
      c.globalAlpha = 0.7;
      c.beginPath();
      c.moveTo(tx, H - 14); c.lineTo(tx, H - 10);
      c.stroke();
      c.globalAlpha = 1;
      if (t % wEvery === 0) c.fillText(t + 's', X(t), H - 1);
    }
  }

  function drawChart() {
    if (!curve || !curve.xs.length) {
      emptyCanvas(el.chart, clip ? 'computing curve' : 'no curve yet');
      return;
    }
    var p = prep(el.chart);
    var c = p.ctx, W = p.w, H = p.h;
    var col = palette();
    var padL = 42, padR = 16, padT = 16, padB = 26;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var maxT = Math.max(curve.total, 0.001);
    var X = function (t) { return padL + t / maxT * plotW; };
    var Y = function (v) { return padT + (1 - v) * plotH; };

    c.strokeStyle = col.ruleSoft;
    c.lineWidth = 1;
    c.fillStyle = col.ink3;
    c.font = '9px "Azeret Mono", monospace';
    c.textAlign = 'right';
    c.textBaseline = 'middle';
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      var y = Math.round(Y(v)) + 0.5;
      c.beginPath(); c.moveTo(padL, y); c.lineTo(W - padR, y); c.stroke();
      c.fillText(v.toFixed(2), padL - 8, Y(v));
    });
    c.textAlign = 'center';
    c.textBaseline = 'top';
    var every = Math.max(1, Math.ceil(34 / (plotW / maxT)));
    for (var t = 0; t <= Math.floor(maxT); t++) {
      var gx = Math.round(X(t)) + 0.5;
      c.globalAlpha = 0.6;
      c.beginPath(); c.moveTo(gx, padT); c.lineTo(gx, padT + plotH); c.stroke();
      c.globalAlpha = 1;
      if (t % every === 0) c.fillText(t + 's', X(t), padT + plotH + 8);
    }

    var xs = curve.xs, ys = curve.ys;

    // area under the trace
    var grad = c.createLinearGradient(0, padT, 0, padT + plotH);
    grad.addColorStop(0, alpha(col.sig, 0.26));
    grad.addColorStop(1, alpha(col.sig, 0));
    c.beginPath();
    c.moveTo(X(xs[0]), Y(0));
    for (var i = 0; i < xs.length; i++) c.lineTo(X(xs[i]), Y(ys[i]));
    c.lineTo(X(xs[xs.length - 1]), Y(0));
    c.closePath();
    c.fillStyle = grad;
    c.fill();

    // threshold line, over the fill and under the trace
    c.save();
    c.strokeStyle = col.ink2;
    c.setLineDash([4, 4]);
    c.beginPath();
    c.moveTo(padL, Math.round(Y(threshold)) + 0.5);
    c.lineTo(W - padR, Math.round(Y(threshold)) + 0.5);
    c.stroke();
    c.restore();
    c.fillStyle = col.ink2;
    c.textAlign = 'left';
    c.textBaseline = 'bottom';
    c.fillText('threshold ' + threshold.toFixed(2), padL + 6, Y(threshold) - 4);

    // trace
    c.strokeStyle = col.sig;
    c.lineWidth = 1.75;
    c.lineJoin = 'round';
    c.beginPath();
    for (var j = 0; j < xs.length; j++) {
      if (j === 0) c.moveTo(X(xs[j]), Y(ys[j]));
      else c.lineTo(X(xs[j]), Y(ys[j]));
    }
    c.stroke();

    // points, coloured against the live threshold
    for (var k = 0; k < xs.length; k++) {
      c.beginPath();
      c.arc(X(xs[k]), Y(ys[k]), 2.6, 0, Math.PI * 2);
      c.fillStyle = ys[k] >= threshold ? col.sig : col.warn;
      c.fill();
      c.strokeStyle = col.well;
      c.lineWidth = 1;
      c.stroke();
    }
  }

  function redraw() { drawWave(); drawChart(); }

  /* ------------------------------------------------------------- pipeline */

  function streamEnds() {
    var step = Math.round(STREAM_STEP_S * SR);
    var total = clip.pcm.length;
    var ends = [];
    for (var e = step; e < total; e += step) ends.push(e);
    if (!ends.length || ends[ends.length - 1] !== total) ends.push(total);
    if (ends.length > STREAM_MAX_POINTS) {
      var stride = Math.ceil(ends.length / STREAM_MAX_POINTS);
      var keep = [];
      for (var i = 0; i < ends.length; i += stride) keep.push(ends[i]);
      if (keep[keep.length - 1] !== total) keep.push(total);
      while (keep.length > STREAM_MAX_POINTS) keep.splice(keep.length - 2, 1);
      ends = keep;
    }
    return ends;
  }

  function runStream(gen) {
    var ends = streamEnds();
    curve = { xs: [], ys: [], total: clip.duration };
    var i = 0;

    function step() {
      if (gen !== streamGen) return Promise.resolve();
      if (i >= ends.length) {
        var dt = ends.length > 1 ? (ends[1] - ends[0]) / SR : STREAM_STEP_S;
        el.streamNote.textContent = ends.length + ' points, ' + dt.toFixed(2) + ' s apart';
        status('ready', 'Ready');
        return Promise.resolve();
      }
      el.streamNote.textContent = 'point ' + (i + 1) + ' of ' + ends.length;
      return infer(activeId, windowAt(clip.pcm, ends[i])).then(function (r) {
        if (gen !== streamGen) return;
        curve.xs.push(ends[i] / SR);
        curve.ys.push(r.p);
        drawChart();
        i++;
        return yieldToUi().then(step);
      });
    }

    status('busy', 'Streaming curve');
    return step();
  }

  function analyse(source) {
    if (!clip) return Promise.resolve();
    var gen = ++streamGen;
    busy = true;
    curve = null;
    el.streamNote.textContent = '';
    drawChart();
    status('busy', 'Running');
    el.verdictSrc.textContent = source;

    return infer(activeId, windowAt(clip.pcm, clip.pcm.length)).then(function (r) {
      if (gen !== streamGen) return;
      lastProb = r.p;
      el.sLat.textContent = r.ms.toFixed(0) + ' ms';
      el.sDur.textContent = clip.duration.toFixed(2) + ' s';
      renderVerdict(true);
      return runStream(gen);
    }).catch(function (err) {
      if (gen !== streamGen) return;
      status('err', 'Inference failed');
      note(String(err && err.message ? err.message : err));
    }).then(function () {
      busy = false;
    });
  }

  function loadBuffer(arrayBuffer, source) {
    note('');
    status('busy', 'Decoding audio');
    return decodeToPcm(arrayBuffer).then(function (pcm) {
      if (!pcm.length) throw new Error('Decoded to zero samples.');
      clip = { pcm: pcm, duration: pcm.length / SR, source: source };
      drawWave();
      return analyse(source);
    }).catch(function (err) {
      status('err', 'Could not decode');
      note('Could not decode that audio. ' + (err && err.message ? err.message : ''));
    });
  }

  /* ------------------------------------------------------------- inputs */

  function markChip(file) {
    clipFile = file;
    Array.prototype.forEach.call(el.chips.querySelectorAll('.chip'), function (b) {
      b.setAttribute('aria-pressed', String(b.dataset.file === file));
    });
  }

  function loadExample(file) {
    markChip(file);
    return fetch('examples/' + file).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.arrayBuffer();
    }).then(function (buf) {
      return loadBuffer(buf, fmtName(file));
    }).catch(function (err) {
      note('Could not load example. ' + err.message);
      status('err', 'Example missing');
    });
  }

  function buildChips() {
    EXAMPLES.forEach(function (file) {
      var lab = labelOf(file);
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.dataset.file = file;
      b.setAttribute('aria-pressed', 'false');
      b.appendChild(span('cn', fmtName(file)));
      var cl = span('cl', lab);
      cl.dataset.l = lab;
      b.appendChild(cl);
      // a new pick preempts any curve still being computed
      b.addEventListener('click', function () { loadExample(file); });
      el.chips.appendChild(b);
    });
  }

  function buildSeg() {
    order.forEach(function (id, idx) {
      var m = models[id];
      var input = document.createElement('input');
      input.type = 'radio';
      input.name = 'model';
      input.id = 'm_' + id;
      input.value = id;
      input.checked = idx === 0;

      var label = document.createElement('label');
      label.setAttribute('for', 'm_' + id);
      label.appendChild(span('t', m.tag));
      label.appendChild(span('s', m.size_mb.toFixed(1) + ' MB'));

      input.addEventListener('change', function () {
        if (!input.checked) return;
        activeId = id;
        lastProb = null;
        curve = null;
        el.sLat.textContent = '--';
        renderModelStats();
        setThreshold(models[id].threshold, true);
        renderVerdict(false);
        if (clip) analyse(clip.source);
        else getSession(id).catch(function () { /* surfaced by status */ });
      });

      el.modelSeg.appendChild(input);
      el.modelSeg.appendChild(label);
    });
  }

  /* ------------------------------------------------------------- mic */

  var recorder = null, recStream = null, recChunks = [], recStart = 0;
  var recTick = null, recCap = null;
  // 'idle' | 'opening' | 'recording'. Guards against a second session being
  // opened while getUserMedia is still pending, which used to leave an
  // orphaned interval that nothing held a handle to any more.
  var recPhase = 'idle';
  var recGen = 0;

  function pickMime() {
    var opts = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
    for (var i = 0; i < opts.length; i++) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(opts[i])) return opts[i];
    }
    return '';
  }

  // Single teardown path for the timer and the button. Safe to call twice.
  function clearRecTimers() {
    if (recTick !== null) { clearInterval(recTick); recTick = null; }
    if (recCap !== null) { clearTimeout(recCap); recCap = null; }
  }

  function resetRecUi() {
    clearRecTimers();
    recPhase = 'idle';
    el.recBtn.dataset.rec = '0';
    el.recState.textContent = 'Record';
    el.recHint.textContent = 'Speak, then stop. Max 30 s.';
  }

  function releaseStream() {
    if (recStream) {
      recStream.getTracks().forEach(function (t) { t.stop(); });
      recStream = null;
    }
  }

  function stopRecording() {
    if (recPhase === 'idle') return;
    // Stop the clock now. MediaRecorder.onstop is asynchronous, so leaving
    // teardown to the event let the counter run on past the click.
    recGen++;
    resetRecUi();
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    else releaseStream();
  }

  function startRecording() {
    if (recPhase !== 'idle') return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
      note('This browser does not expose a microphone recording API.');
      return;
    }
    recPhase = 'opening';
    var gen = ++recGen;
    el.recState.textContent = 'Waiting for mic';
    navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
    }).then(function (stream) {
      // The user cancelled while the permission prompt was up.
      if (gen !== recGen) {
        stream.getTracks().forEach(function (t) { t.stop(); });
        return;
      }
      recStream = stream;
      recChunks = [];
      var mime = pickMime();
      recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
      recorder.ondataavailable = function (e) { if (e.data && e.data.size) recChunks.push(e.data); };
      recorder.onstop = function () {
        resetRecUi();
        releaseStream();
        var blob = new Blob(recChunks, { type: recorder.mimeType || 'audio/webm' });
        if (!blob.size) { note('Nothing was captured.'); return; }
        markChip(null);
        blob.arrayBuffer().then(function (buf) { return loadBuffer(buf, 'Microphone'); });
      };
      recorder.start();
      recPhase = 'recording';
      recStart = performance.now();
      el.recBtn.dataset.rec = '1';
      el.recState.textContent = 'Recording 0.0 s';
      el.recHint.textContent = 'Click again to stop.';
      note('');
      clearRecTimers();
      recTick = setInterval(function () {
        if (recPhase !== 'recording') { clearRecTimers(); return; }
        el.recState.textContent = 'Recording ' +
          ((performance.now() - recStart) / 1000).toFixed(1) + ' s';
      }, 100);
      recCap = setTimeout(stopRecording, MAX_RECORD_MS);
    }).catch(function (err) {
      if (gen !== recGen) return;
      resetRecUi();
      note('Microphone blocked or unavailable. ' + (err && err.name ? err.name : ''));
    });
  }

  /* ------------------------------------------------------------- boot */

  function boot() {
    if (typeof ort === 'undefined') {
      status('err', 'Runtime missing');
      note('onnxruntime-web did not load from vendor/.');
      return;
    }
    // Point onnxruntime at the vendored copies. Absolute URLs because ORT
    // dynamically imports the loader, and a .js extension so no host has to
    // know the .mjs media type.
    var vendor = new URL('vendor/', document.baseURI);
    ort.env.wasm.wasmPaths = {
      mjs: new URL('ort-wasm-simd-threaded.js', vendor).href,
      wasm: new URL('ort-wasm-simd-threaded.wasm', vendor).href
    };
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = false;
    ort.env.logLevel = 'error';

    el.thresh.addEventListener('input', function () {
      setThreshold(parseFloat(el.thresh.value), false);
    });

    el.recBtn.addEventListener('click', function () {
      // Keyed off the real phase, not the button attribute, so a click while
      // the permission prompt is open cancels instead of opening a second one.
      if (recPhase === 'idle') startRecording();
      else stopRecording();
    });

    el.file.addEventListener('change', function () {
      var f = el.file.files && el.file.files[0];
      if (!f) return;
      markChip(null);
      f.arrayBuffer().then(function (buf) { return loadBuffer(buf, f.name); });
      el.file.value = '';
    });

    var rt = null;
    window.addEventListener('resize', function () {
      clearTimeout(rt);
      rt = setTimeout(redraw, 120);
    });

    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', redraw);
    else if (mq.addListener) mq.addListener(redraw);

    if (document.fonts && document.fonts.ready) document.fonts.ready.then(redraw);

    fetch(CONFIG_URL).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (c) {
      cfg = c;
      SR = c.sample_rate;
      WIN = Math.round(c.sample_rate * c.window_seconds);

      c.models.slice().sort(function (a, b) { return b.size_mb - a.size_mb; })
        .forEach(function (m, i, arr) {
          m.tag = i === 0 ? 'Accurate' : (i === arr.length - 1 ? 'Fast' : 'Mid');
          models[m.id] = m;
          order.push(m.id);
        });
      activeId = order[0];

      buildSeg();
      buildChips();
      renderModelStats();
      setThreshold(models[activeId].threshold, true);
      redraw();

      // warm the default session so the first click is not the slow one
      getSession(activeId).catch(function () { /* surfaced by status */ });
    }).catch(function (err) {
      status('err', 'Config failed');
      note('Could not read models/config.json. ' + err.message);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
