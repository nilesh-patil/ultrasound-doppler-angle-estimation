/* demo.js — INTERACTIVE #3: precomputed prediction explorer (Clinical page).
 *
 * Drives the asset-contract DOM ids inside <section id="prediction-demo">:
 *   demo-image-select   <select>  (study images, grouped by patient-proxy)
 *   demo-thumb          <img>     (B-mode thumbnail from assets/images/bmode/)
 *   demo-rotation-scrub <input type=range>  (-60..60 step 5)
 *   demo-result         container for theta_true / theta_pred / signed residual
 *   demo-band           inline-SVG number line with the +/-20.50 conformal band
 *   demo-enable-live    disabled "coming soon" affordance (live ONNX is a follow-up)
 *
 * Honesty: these are PRECOMPUTED out-of-fold predictions from the tuned
 * DenseNet201, not live inference. The conformal half-width (+/-20.50 deg) is a
 * fixed constant from a single seed-42 patient-disjoint split, supplied by the
 * data file. The only number this script computes is the signed residual
 * (theta_pred - theta_true), which is already precomputed in the JSON and only
 * re-derived defensively.
 *
 * Progressive enhancement: the figure ships a static SVG <img class="fallback">
 * that stays visible until the first successful paint. On any throw we leave the
 * fallback in place and never hide it. No CSV is parsed in the browser.
 */
(function () {
  'use strict';

  var DEFAULT_HALFWIDTH = 20.50; // ledger constant; overridden by data file if present
  var ROOT_ID = 'prediction-demo';

  // The layout emits this script as
  //   <script defer src=".../assets/js/demo.js" data-base="<baseurl>/assets/data/">
  // so the baseurl-aware data directory travels on our own <script> tag. Capture
  // it at initial execution time, while document.currentScript is still valid
  // (it is null inside deferred callbacks like DOMContentLoaded). This is the
  // canonical attribute (data-base) — NOT data-asset-base — and it is absolute
  // under the Jekyll baseurl, so it never 404s the way a relative "../" hop did.
  var SCRIPT_DATA_BASE = (function () {
    var s = document.currentScript;
    var v = s && s.getAttribute('data-base');
    return v ? v.replace(/\/+$/, '') + '/' : '';
  })();

  function $(id) { return document.getElementById(id); }

  function num(x) { return Math.round(x * 100) / 100; }

  function fmtSigned(x) {
    var v = num(x);
    return (v > 0 ? '+' : (v < 0 ? '−' : '')) + Math.abs(v).toFixed(2);
  }

  function fmtDeg(x) {
    return num(x).toFixed(2) + '°';
  }

  // ---- data normalisation ---------------------------------------------------
  // Accept either the canonical grouped schema (demo_predictions.json:
  //   { conformal_halfwidth, images:[{id,patient,img,base:{...},sweep:[{rot,true,pred,err}]}] })
  // or the flat per-row schema this builder emits (predictions.json:
  //   { conformal_halfwidth, records:[{image_id,patient_id,rotation_deg,theta_true,theta_pred,error,img,base}] }).
  function normalise(data) {
    var halfwidth = (data && typeof data.conformal_halfwidth === 'number')
      ? data.conformal_halfwidth : DEFAULT_HALFWIDTH;

    var byId = {};
    var order = [];

    function ensure(id, patient, img) {
      if (!byId[id]) {
        byId[id] = { id: id, patient: patient, img: img, sweep: {}, base: null };
        order.push(id);
      }
      return byId[id];
    }

    if (data && Array.isArray(data.images)) {
      data.images.forEach(function (im) {
        var rec = ensure(im.id, im.patient, im.img);
        var sweep = im.sweep || [];
        sweep.forEach(function (s) {
          var rot = +s.rot;
          var t = +s.true, p = +s.pred;
          var e = (typeof s.err === 'number') ? s.err : (p - t);
          rec.sweep[rot] = { rot: rot, theta_true: t, theta_pred: p, error: e };
        });
        if (im.base) {
          var bt = +im.base.true, bp = +im.base.pred;
          var be = (typeof im.base.err === 'number') ? im.base.err : (bp - bt);
          rec.base = { rot: 0, theta_true: bt, theta_pred: bp, error: be };
          if (!rec.sweep[0]) rec.sweep[0] = rec.base;
        }
      });
    } else if (data && Array.isArray(data.records)) {
      data.records.forEach(function (r) {
        var rec = ensure(r.image_id, r.patient_id, r.img || (r.image_id + '.jpg'));
        var rot = +r.rotation_deg;
        var t = +r.theta_true, p = +r.theta_pred;
        var e = (typeof r.error === 'number') ? r.error : (p - t);
        var entry = { rot: rot, theta_true: t, theta_pred: p, error: e };
        rec.sweep[rot] = entry;
        if (rot === 0 || r.base === true) rec.base = entry;
      });
    } else {
      throw new Error('demo: unrecognised prediction data shape');
    }

    var images = order.map(function (id) {
      var rec = byId[id];
      var rots = Object.keys(rec.sweep).map(Number).sort(function (a, b) { return a - b; });
      if (!rec.base) rec.base = rec.sweep[rots[0]];
      return { id: rec.id, patient: rec.patient, img: rec.img, base: rec.base, sweepMap: rec.sweep, rots: rots };
    });

    if (!images.length) throw new Error('demo: no images in prediction data');

    // group by patient-proxy for the <optgroup> ordering, preserving id order
    images.sort(function (a, b) {
      if (a.patient !== b.patient) return a.patient - b.patient;
      return a.id < b.id ? -1 : (a.id > b.id ? 1 : 0);
    });

    return { halfwidth: halfwidth, images: images };
  }

  // ---- DOM building ---------------------------------------------------------
  function populateSelect(sel, images) {
    sel.innerHTML = '';
    var groups = {};
    var groupOrder = [];
    images.forEach(function (im) {
      var key = im.patient;
      if (!groups[key]) { groups[key] = []; groupOrder.push(key); }
      groups[key].push(im);
    });
    groupOrder.forEach(function (p) {
      var og = document.createElement('optgroup');
      og.label = 'Patient group ' + p;
      groups[p].forEach(function (im) {
        var opt = document.createElement('option');
        opt.value = im.id;
        opt.textContent = im.id;
        og.appendChild(opt);
      });
      sel.appendChild(og);
    });
  }

  function nearestRot(image, want) {
    if (image.sweepMap[want]) return want;
    var best = image.rots[0], bestD = Infinity;
    for (var i = 0; i < image.rots.length; i++) {
      var d = Math.abs(image.rots[i] - want);
      if (d < bestD) { bestD = d; best = image.rots[i]; }
    }
    return best;
  }

  // ---- conformal number line (inline SVG) -----------------------------------
  var SVGNS = 'http://www.w3.org/2000/svg';
  function svg(tag, attrs) {
    var el = document.createElementNS(SVGNS, tag);
    for (var k in attrs) if (attrs.hasOwnProperty(k)) el.setAttribute(k, attrs[k]);
    return el;
  }

  function renderBand(host, entry, halfwidth) {
    var W = 520, H = 96, PADX = 28, axisY = 58;
    // domain: pad around true/pred and the band so everything is visible
    var lo = Math.min(entry.theta_true, entry.theta_pred - halfwidth);
    var hi = Math.max(entry.theta_true, entry.theta_pred + halfwidth);
    var span = hi - lo;
    var dlo = lo - span * 0.12 - 2;
    var dhi = hi + span * 0.12 + 2;
    function x(v) { return PADX + (v - dlo) / (dhi - dlo) * (W - 2 * PADX); }

    var inside = Math.abs(entry.error) <= halfwidth;

    var s = svg('svg', {
      viewBox: '0 0 ' + W + ' ' + H,
      width: '100%', preserveAspectRatio: 'xMidYMid meet',
      'class': 'demo__numberline'
    });

    // baseline axis
    s.appendChild(svg('line', {
      x1: PADX, y1: axisY, x2: W - PADX, y2: axisY,
      stroke: 'var(--hairline)', 'stroke-width': '1'
    }));

    // conformal band rectangle around the prediction
    var bx1 = x(entry.theta_pred - halfwidth);
    var bx2 = x(entry.theta_pred + halfwidth);
    var band = svg('rect', {
      x: Math.min(bx1, bx2), y: axisY - 16,
      width: Math.abs(bx2 - bx1), height: 32, rx: 3,
      fill: inside ? 'var(--accent-wash)' : 'rgba(181,64,58,0.12)',
      stroke: inside ? 'var(--accent)' : 'var(--series-2)',
      'stroke-width': '1'
    });
    s.appendChild(band);

    // band edge ticks + labels
    [entry.theta_pred - halfwidth, entry.theta_pred + halfwidth].forEach(function (v) {
      s.appendChild(svg('line', {
        x1: x(v), y1: axisY - 16, x2: x(v), y2: axisY + 16,
        stroke: inside ? 'var(--accent)' : 'var(--series-2)', 'stroke-width': '1',
        'stroke-dasharray': '2 2'
      }));
      var lbl = svg('text', {
        x: x(v), y: axisY + 30, 'text-anchor': 'middle',
        'class': 'axis-text', fill: 'var(--ink-mute)', 'font-size': '10'
      });
      lbl.textContent = num(v).toFixed(0) + '°';
      s.appendChild(lbl);
    });

    // predicted marker (brick-red diamond — lead/residual emphasis colour)
    var px = x(entry.theta_pred);
    s.appendChild(svg('path', {
      d: 'M ' + px + ' ' + (axisY - 9) + ' l 7 9 l -7 9 l -7 -9 z',
      fill: 'var(--series-2)', stroke: 'var(--bg-raised)', 'stroke-width': '1'
    }));
    var predLbl = svg('text', {
      x: px, y: axisY - 16, 'text-anchor': 'middle',
      'class': 'axis-text', fill: 'var(--series-2)', 'font-size': '11', 'font-weight': '600'
    });
    predLbl.textContent = 'θ̂ ' + num(entry.theta_pred).toFixed(1) + '°';
    s.appendChild(predLbl);

    // reference (true) marker (ink circle)
    var tx = x(entry.theta_true);
    s.appendChild(svg('circle', {
      cx: tx, cy: axisY, r: '5', fill: 'var(--ink)', stroke: 'var(--bg-raised)', 'stroke-width': '1.5'
    }));
    var trueLbl = svg('text', {
      x: tx, y: axisY + 30, 'text-anchor': 'middle',
      'class': 'axis-text', fill: 'var(--ink)', 'font-size': '11', 'font-weight': '600'
    });
    trueLbl.textContent = 'θ ' + num(entry.theta_true).toFixed(1) + '°';
    s.appendChild(trueLbl);

    host.setAttribute('aria-label',
      'Reference angle ' + fmtDeg(entry.theta_true) +
      '; predicted ' + fmtDeg(entry.theta_pred) +
      ' with a 90% conformal band of plus or minus ' + halfwidth.toFixed(2) + ' degrees from ' +
      fmtDeg(entry.theta_pred - halfwidth) + ' to ' + fmtDeg(entry.theta_pred + halfwidth) +
      '. The reference reading is ' + (inside ? 'inside' : 'outside') + ' the band.');
    host.innerHTML = '';
    host.appendChild(s);
  }

  function renderResult(host, entry, halfwidth) {
    var inside = Math.abs(entry.error) <= halfwidth;
    host.innerHTML = '';
    function row(term, val, cls) {
      var dt = document.createElement('dt'); dt.textContent = term;
      var dd = document.createElement('dd'); dd.textContent = val;
      if (cls) dd.className = cls;
      host.appendChild(dt); host.appendChild(dd);
    }
    row('Reference angle θ', fmtDeg(entry.theta_true));
    row('Predicted θ̂', fmtDeg(entry.theta_pred));
    row('Signed residual (θ̂ − θ)', fmtSigned(entry.error) + '°', 'demo__residual');
    row('90% conformal band',
      fmtDeg(entry.theta_pred - halfwidth) + ' to ' + fmtDeg(entry.theta_pred + halfwidth) +
      ' (±' + halfwidth.toFixed(2) + '°)');
    row('Reference within band', inside ? 'yes' : 'no');
  }

  // ---- wiring ---------------------------------------------------------------
  function init(data, baseFor) {
    var model = normalise(data);
    var root = $(ROOT_ID);
    var fig = root ? root.closest('.fig') : null;
    var sel = $('demo-image-select');
    var thumb = $('demo-thumb');
    var scrub = $('demo-rotation-scrub');
    var result = $('demo-result');
    var band = $('demo-band');
    var live = $('demo-enable-live');
    var rotValue = $('demo-rotation-value');

    if (!sel || !result || !band) throw new Error('demo: required mount ids missing');

    populateSelect(sel, model.images);

    var imgBase = baseFor('images/bmode/');
    var current = model.images[0];

    var reduceMotion = window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function setImage(id) {
      current = null;
      for (var i = 0; i < model.images.length; i++) {
        if (model.images[i].id === id) { current = model.images[i]; break; }
      }
      if (!current) current = model.images[0];
      sel.value = current.id;
      if (thumb) {
        thumb.src = imgBase + current.img;
        thumb.alt = 'Longitudinal B-mode carotid image ' + current.id +
          ' (patient group ' + current.patient + ').';
      }
      // clamp scrub to a rotation we actually have for this image
      if (scrub) {
        var want = +scrub.value;
        var r = nearestRot(current, want);
        if (r !== want) scrub.value = String(r);
      }
      paint();
    }

    function paint() {
      var want = scrub ? +scrub.value : 0;
      var rot = nearestRot(current, want);
      var entry = current.sweepMap[rot];
      if (scrub) {
        scrub.setAttribute('aria-valuetext',
          rot + ' degrees rotation; predicted ' + fmtDeg(entry.theta_pred) +
          ', reference ' + fmtDeg(entry.theta_true) +
          ', residual ' + fmtSigned(entry.error) + ' degrees.');
        if (thumb) {
          if (reduceMotion) thumb.style.transition = 'none';
          thumb.style.transform = 'rotate(' + (-rot) + 'deg)';
        }
        if (rotValue) rotValue.textContent = rot + '°';
      }
      renderResult(result, entry, model.halfwidth);
      renderBand(band, entry, model.halfwidth);
    }

    sel.addEventListener('change', function () { setImage(sel.value); });
    if (scrub) {
      scrub.addEventListener('input', paint);
      scrub.addEventListener('change', paint);
    }

    // live-ONNX affordance stays disabled in v-00.
    if (live) {
      live.disabled = true;
      live.setAttribute('aria-disabled', 'true');
      live.title = 'Live in-browser inference is a documented follow-up; v-00 ships precomputed predictions.';
    }

    setImage(current.id);

    // first successful paint complete -> reveal the live widget, retire fallback
    if (fig) {
      fig.classList.add('js-ready');
      var fb = fig.querySelector('img.fallback');
      if (fb) fb.hidden = true;
    }
    if (root) root.dataset.ready = '1';
  }

  function boot() {
    var root = $(ROOT_ID);
    if (!root) return; // not on this page
    var fig = root.closest('.fig');
    var mount = fig || root;

    // Resolve a URL under /assets/<rel> honouring the Jekyll baseurl. The single
    // source of truth is the absolute data base on our own <script data-base>
    // tag (e.g. "/ultrasound-doppler-angle-estimation/assets/data/"); the assets
    // root is its parent, so we strip the trailing "data/" segment. If data.js
    // exposed UDA.assetBase we honour that; otherwise we fall back to the
    // injected data base. Only as a last resort do we use a relative hop. This
    // builds an ABSOLUTE base so it cannot 404 against /clinical/ under baseurl.
    function assetsBase() {
      if (window.UDA && typeof UDA.assetBase === 'string' && UDA.assetBase) {
        return UDA.assetBase.replace(/\/+$/, '') + '/';
      }
      if (SCRIPT_DATA_BASE) {
        // ".../assets/data/" -> ".../assets/"
        return SCRIPT_DATA_BASE.replace(/data\/+$/, '');
      }
      return '../'; // last resort: demo.js lives at /assets/js/, assets root is up one
    }
    function baseFor(rel) { return assetsBase() + rel; }

    function fail(err) {
      // leave the static fallback visible; surface the failure for diagnostics.
      mount.dataset.failed = '1';
      if (window.console && console.warn) console.warn('demo: falling back to static figure —', err && err.message);
    }

    function tryShapes(loader) {
      // The flat predictions.json ships in v-00; the grouped exporter file
      // (demo_predictions.json) is an optional upgrade, tried only if the flat
      // file is absent, so the common path never fires a spurious 404.
      return loader('predictions.json').catch(function () {
        return loader('demo_predictions.json');
      });
    }

    try {
      if (window.UDA && UDA.data && typeof UDA.data.load === 'function') {
        tryShapes(function (name) { return UDA.data.load(name); })
          .then(function (data) { init(data, baseFor); })
          .catch(fail);
      } else {
        // Standalone fetch fallback if the shared UDA.data helper is absent.
        // Prefer the absolute data dir injected on our <script data-base>; only
        // synthesise it from the assets base if the attribute was missing.
        var dataDir = SCRIPT_DATA_BASE || baseFor('data/');
        function fetchJson(name) {
          return fetch(dataDir + name, { credentials: 'same-origin' })
            .then(function (r) { if (!r.ok) throw new Error(name + ' ' + r.status); return r.json(); });
        }
        tryShapes(fetchJson)
          .then(function (data) { init(data, baseFor); })
          .catch(fail);
      }
    } catch (err) {
      fail(err);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
