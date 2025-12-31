/*
 * doppler-explainer.js — INTERACTIVE #1 (Overview)
 *
 * The conceptual hook: sweep the ultrasound BEAM across a steady real B-mode
 * carotid image with a slider and watch the insonation angle theta and the
 * resulting spectral-Doppler velocity error change live. Pure client-side,
 * zero-dependency, progressive enhancement.
 *
 * Model (authoritative, from the math spec):
 *   Spectral Doppler reports v from a measured Doppler shift assuming an angle
 *   theta. If the TRUE beam-to-vessel angle is theta0 but the operator (or the
 *   model) uses theta, the reported velocity is scaled by cos(theta0)/cos(theta).
 *   Signed fractional velocity error:
 *
 *       epsilon = cos(theta0) / cos(theta) - 1
 *
 *   Sanity values (verified): eps(60,65)=+18.31%, eps(60,55)=-12.83%,
 *   eps(80,75)=-32.91%, eps(t0,t0)=0.
 *   Secondary readout: the raw angle-correction multiplier 1 / cos(theta).
 *
 * This beam overlay is SYNTHETIC / ILLUSTRATIVE: it does not map 1:1 to the
 * rotation-augmentation grid used to train the estimator. It is a beam-vs-vessel
 * angle teaching tool only.
 *
 * Progressive-enhancement contract:
 *   The explainer <figure class="fig widget"> ships a static SVG fallback img
 *   (figure2_augmentation.svg) plus a server-rendered theta-vs-multiplier table,
 *   both visible by default. On successful first paint we add `.js-ready` to the
 *   figure (CSS hides .fallback, shows .live) and hide the static table. On any
 *   throw we leave the fallback in place and never touch img.hidden.
 */
(function () {
  'use strict';

  // -- Guard-rails / constants (from the math spec) ------------------------
  var ANGLE_MIN = 0;        // slider lower bound (degrees)
  var ANGLE_MAX = 89;       // slider upper bound; 90 is the singularity
  var DEFAULT_THETA = 60;   // default insonation angle
  var DEFAULT_THETA0 = 60;  // reference (true) angle -> initial error exactly 0
  var EPS = 1e-6;           // numerical guard near cos(theta) -> 0
  var DISPLAY_CAP = 200;    // |error %| cap; beyond this we render "diverges"
  var DEG = Math.PI / 180;

  // -- Pure math -----------------------------------------------------------

  // Clamp an angle into the legal [ANGLE_MIN, ANGLE_MAX] range.
  function clampAngle(a) {
    if (!isFinite(a)) return DEFAULT_THETA;
    if (a < ANGLE_MIN) return ANGLE_MIN;
    if (a > ANGLE_MAX) return ANGLE_MAX;
    return a;
  }

  // cos with a floor so we never divide by (or report) zero near 90 deg.
  function safeCos(deg) {
    var c = Math.cos(deg * DEG);
    if (Math.abs(c) < EPS) return c < 0 ? -EPS : EPS;
    return c;
  }

  // Signed fractional velocity error epsilon = cos(theta0)/cos(theta) - 1.
  // Returned as a fraction (multiply by 100 for percent).
  function velocityError(theta0, theta) {
    return safeCos(theta0) / safeCos(theta) - 1;
  }

  // Raw angle-correction multiplier 1 / cos(theta).
  function velocityMultiplier(theta) {
    return 1 / safeCos(theta);
  }

  // Format a signed percentage, applying the divergence cap near 90 deg.
  function fmtSignedPct(fraction) {
    var pct = fraction * 100;
    if (!isFinite(pct) || Math.abs(pct) > DISPLAY_CAP) {
      return { text: pct < 0 ? '−diverges' : '+diverges', diverges: true };
    }
    var sign = pct >= 0 ? '+' : '−'; // U+2212 minus
    return { text: sign + Math.abs(pct).toFixed(1) + '%', diverges: false };
  }

  // -- Small DOM helpers ---------------------------------------------------
  function $(id) { return document.getElementById(id); }

  function prefersReducedMotion() {
    return window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  // -- 1/cos(theta) curve (inline SVG) -------------------------------------
  // Drawn once into the explainer-cos-curve mount; a marker is repositioned
  // on each input. Self-contained: no external chart library.
  var SVG_NS = 'http://www.w3.org/2000/svg';
  var CURVE = {
    w: 320, h: 180,
    padL: 38, padR: 12, padT: 12, padB: 28,
    xMax: ANGLE_MAX,   // degrees
    yMax: 6            // cap the multiplier axis (1/cos goes to inf at 90)
  };

  function curveX(theta) {
    var inner = CURVE.w - CURVE.padL - CURVE.padR;
    return CURVE.padL + (theta / CURVE.xMax) * inner;
  }
  function curveY(mult) {
    var inner = CURVE.h - CURVE.padT - CURVE.padB;
    var clipped = Math.min(mult, CURVE.yMax);
    return CURVE.padT + (1 - (clipped - 1) / (CURVE.yMax - 1)) * inner;
  }

  function el(name, attrs, text) {
    var node = document.createElementNS(SVG_NS, name);
    if (attrs) {
      for (var k in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, k)) {
          node.setAttribute(k, attrs[k]);
        }
      }
    }
    if (text != null) node.textContent = text;
    return node;
  }

  // Build the static parts of the curve once; return the marker handles.
  function buildCurve(mount) {
    // Clear any prior content (idempotent re-init).
    while (mount.firstChild) mount.removeChild(mount.firstChild);

    // The outer #explainer-cos-curve mount carries role=img + aria-label
    // (index.md); this inner SVG's label would be pruned, so we omit it to
    // avoid dead, redundant accessibility markup.
    var svg = el('svg', {
      viewBox: '0 0 ' + CURVE.w + ' ' + CURVE.h,
      class: 'explainer__cos-svg'
    });

    // Axes (hairlines).
    svg.appendChild(el('line', {
      x1: CURVE.padL, y1: CURVE.padT,
      x2: CURVE.padL, y2: CURVE.h - CURVE.padB,
      class: 'grid-line'
    }));
    svg.appendChild(el('line', {
      x1: CURVE.padL, y1: CURVE.h - CURVE.padB,
      x2: CURVE.w - CURVE.padR, y2: CURVE.h - CURVE.padB,
      class: 'grid-line'
    }));

    // y ticks at multiplier = 1, 2, 4, 6.
    [1, 2, 4, 6].forEach(function (m) {
      if (m > CURVE.yMax) return;
      var y = curveY(m);
      svg.appendChild(el('text', {
        x: CURVE.padL - 5, y: y + 3, 'text-anchor': 'end',
        class: 'axis-text'
      }, '×' + m));
    });
    // x ticks at 0, 30, 60, 89.
    [0, 30, 60, 89].forEach(function (a) {
      var x = curveX(a);
      svg.appendChild(el('text', {
        x: x, y: CURVE.h - CURVE.padB + 16, 'text-anchor': 'middle',
        class: 'axis-text'
      }, a + '°'));
    });

    // The 1/cos(theta) curve itself, sampled to the y cap.
    var d = '';
    for (var a = 0; a <= ANGLE_MAX; a += 1) {
      var m = velocityMultiplier(a);
      if (m > CURVE.yMax) { // stop drawing once it leaves the frame
        d += (d ? ' L' : 'M') + curveX(a) + ' ' + curveY(CURVE.yMax);
        break;
      }
      d += (d ? ' L' : 'M') + curveX(a) + ' ' + curveY(m);
    }
    svg.appendChild(el('path', { d: d, class: 'series-1', fill: 'none' }));

    // Moving marker: a vertical guide + dot, repositioned on input.
    var guide = el('line', { class: 'explainer__cos-guide' });
    var dot = el('circle', { r: 4, class: 'series-2' });
    svg.appendChild(guide);
    svg.appendChild(dot);

    mount.appendChild(svg);
    return { guide: guide, dot: dot };
  }

  function moveMarker(handles, theta) {
    if (!handles) return;
    var x = curveX(clampAngle(theta));
    var mult = velocityMultiplier(theta);
    var y = curveY(mult);
    handles.guide.setAttribute('x1', x);
    handles.guide.setAttribute('x2', x);
    handles.guide.setAttribute('y1', CURVE.padT);
    handles.guide.setAttribute('y2', CURVE.h - CURVE.padB);
    handles.dot.setAttribute('cx', x);
    handles.dot.setAttribute('cy', y);
  }

  // -- Beam / vessel / angle overlay (inline SVG over the STATIC B-mode) ----
  // The image no longer rotates (that left the 4:3 frame mostly empty). Instead
  // the slider sweeps the ultrasound BEAM across the steady image: we draw the
  // vessel (flow) axis, the beam at angle theta to it, and the angle arc between
  // them. Coordinates are in the overlay's 0..100 viewBox; colour + fill:none
  // come from the .vessel / .beam / .arc CSS, stroke-width is set inline.
  function buildOverlay(mount) {
    while (mount.firstChild) mount.removeChild(mount.firstChild);
    mount.appendChild(el('line', { x1: 10, y1: 54, x2: 90, y2: 54, 'class': 'vessel', 'stroke-width': 1.2 }));
    var beam = el('line', { 'class': 'beam', 'stroke-width': 1.2 });
    var arc = el('path', { 'class': 'arc', 'stroke-width': 1 });
    var label = el('text', { 'text-anchor': 'middle', fill: '#fff', 'font-size': 6, 'font-weight': '600' });
    mount.appendChild(beam);
    mount.appendChild(arc);
    mount.appendChild(label);
    return { beam: beam, arc: arc, label: label };
  }

  function updateOverlay(h, theta) {
    if (!h) return;
    var t = clampAngle(theta) * DEG;
    var cx = 50, cy = 54, c = Math.cos(t), s = Math.sin(t);
    // Beam: through the intersection at angle theta above the (rightward) vessel.
    h.beam.setAttribute('x1', (cx - 9 * c).toFixed(1));
    h.beam.setAttribute('y1', (cy + 9 * s).toFixed(1));
    h.beam.setAttribute('x2', (cx + 44 * c).toFixed(1));
    h.beam.setAttribute('y2', (cy - 44 * s).toFixed(1));
    // Angle arc from the vessel (+x) up to the beam, radius 15.
    var r = 15;
    h.arc.setAttribute('d', 'M ' + (cx + r) + ' ' + cy + ' A ' + r + ' ' + r + ' 0 0 0 ' +
      (cx + r * c).toFixed(1) + ' ' + (cy - r * s).toFixed(1));
    // "theta" label on the arc bisector.
    h.label.setAttribute('x', (cx + 24 * Math.cos(t / 2)).toFixed(1));
    h.label.setAttribute('y', (cy - 24 * Math.sin(t / 2) + 2).toFixed(1));
    h.label.textContent = 'θ';
  }

  // -- Wiring --------------------------------------------------------------

  function init() {
    var slider = $('explainer-rotation');

    // The slider is the irreducible minimum; without it there is nothing to
    // enhance, so we bail and leave the static fallback untouched.
    if (!slider) return;

    // The enclosing <figure class="fig widget" id="chart-explainer"> is what
    // the CSS swap keys on (.widget.js-ready hides .fallback, shows .live).
    // The inner <section id="doppler-explainer"> is NOT the widget, so we must
    // climb to the figure; fall back to the section only if no figure exists.
    var figure = slider.closest('.fig.widget, .widget') ||
      document.getElementById('chart-explainer') ||
      $('doppler-explainer');
    var bmode = $('explainer-bmode');
    var overlay = $('explainer-overlay');
    var thetaOut = $('explainer-theta-readout');
    var gauge = $('explainer-velocity-gauge');
    var curveMount = $('explainer-cos-curve');

    // The readout div holds three sibling leaf spans (theta, error, multiplier).
    // We must write into the leaf spans, never the parent's textContent, or we
    // would delete the gauge span and the live error display on first input.
    var thetaSpan = thetaOut ? thetaOut.querySelector('.readout__theta') : null;
    var errSpan = thetaOut ? thetaOut.querySelector('.readout__err') : null;

    var reduce = prefersReducedMotion();
    var curveHandles = null;
    if (curveMount) {
      try { curveHandles = buildCurve(curveMount); } catch (e) { curveHandles = null; }
    }
    var overlayHandles = null;
    if (overlay) {
      try { overlayHandles = buildOverlay(overlay); } catch (e) { overlayHandles = null; }
    }

    // theta0 is the fixed reference (true) angle; the slider drives theta.
    // If the page exposes a data-theta0, honor it; else default to 60.
    var theta0 = DEFAULT_THETA0;
    if (figure && figure.dataset && figure.dataset.theta0) {
      var parsed = parseFloat(figure.dataset.theta0);
      if (isFinite(parsed)) theta0 = clampAngle(parsed);
    }

    function render(theta) {
      theta = clampAngle(theta);

      // The B-mode image stays STEADY; the slider sweeps the beam across it.
      // (Rotating the photo left the 4:3 frame mostly empty.) Redraw the
      // beam / vessel / angle overlay for the current theta.
      if (overlayHandles) updateOverlay(overlayHandles, theta);

      var errFrac = velocityError(theta0, theta);
      var err = fmtSignedPct(errFrac);
      var mult = velocityMultiplier(theta);
      var multText = (mult > 50) ? 'diverges' : '×' + mult.toFixed(2);

      // theta readout — write into the leaf span only, never the parent div
      // (the parent also holds the error and multiplier spans).
      if (thetaSpan) {
        thetaSpan.textContent = 'θ = ' + Math.round(theta) + '°';
      }

      // Primary signed velocity error. The data-sign attribute drives the
      // colour language: 0 renders in neutral ink (zero is not an error),
      // ±1 reserve the brick-red error/emphasis tone for nonzero deviations.
      if (errSpan) {
        errSpan.textContent = 'velocity error ' + err.text;
        errSpan.setAttribute(
          'data-sign',
          errFrac === 0 ? '0' : (errFrac > 0 ? '1' : '-1')
        );
      }

      // Secondary angle-correction multiplier 1 / cos(theta).
      if (gauge) {
        gauge.textContent = 'velocity multiplier ' + multText;
        gauge.classList.toggle('is-diverging', err.diverges);
      }

      moveMarker(curveHandles, theta);

      // Accessible live value on the slider itself.
      slider.setAttribute(
        'aria-valuetext',
        Math.round(theta) + ' degrees, velocity error ' + err.text
      );
    }

    // Normalize slider attributes to the spec (defensive; page should match).
    slider.min = String(ANGLE_MIN);
    slider.max = String(ANGLE_MAX);
    slider.step = '1';
    var startVal = parseFloat(slider.value);
    if (!isFinite(startVal)) {
      startVal = DEFAULT_THETA;
      slider.value = String(DEFAULT_THETA);
    }

    slider.addEventListener('input', function () {
      render(parseFloat(slider.value));
    });

    // First paint. No auto-animation under reduced motion (we never animate
    // automatically regardless; the CSS transition is what reduced motion
    // disables, and rendering is purely input-driven).
    render(startVal);

    // Mark ready so CSS swaps fallback -> live. Hide the static fallback img
    // (and any server-rendered table) ONLY after the first successful paint.
    if (figure) {
      figure.classList.add('js-ready');
      var fallbackImg = figure.querySelector('img.fallback');
      if (fallbackImg) fallbackImg.hidden = true;
      var staticTable = figure.querySelector('.explainer__static-table');
      if (staticTable) staticTable.hidden = true;
    }
    if (reduce && bmode) {
      // Belt-and-braces: ensure no transition jank if CSS gating is absent.
      bmode.style.transition = 'none';
      if (overlay) overlay.style.transition = 'none';
    }
  }

  function boot() {
    try {
      init();
    } catch (e) {
      // Any failure leaves the static SVG fallback + table visible.
      if (window.console && window.console.warn) {
        window.console.warn('doppler-explainer: enhancement skipped', e);
      }
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  // Expose the pure model for tests / other widgets (read-only namespace).
  window.UDA = window.UDA || {};
  window.UDA.explainer = {
    velocityError: velocityError,
    velocityMultiplier: velocityMultiplier,
    clampAngle: clampAngle,
    ANGLE_MIN: ANGLE_MIN,
    ANGLE_MAX: ANGLE_MAX,
    EPS: EPS,
    DISPLAY_CAP: DISPLAY_CAP
  };
})();
