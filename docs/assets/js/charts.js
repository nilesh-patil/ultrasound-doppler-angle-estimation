/*
 * charts.js — interactive inline-SVG charts for the Doppler-angle site.
 *
 * Public API: UDA.charts.render(mountId, spec)
 *   spec.type ∈ { scatter, groupedBar, slopegraph, errorScatter,
 *                 blandAltman, calibrationStep }
 *
 * Progressive enhancement contract (asset contract §progressive-enhancement):
 *   - Each chart lives in <figure class="fig widget"> with a default-visible
 *     <img class="fallback"> and a <div class="widget__canvas" id=...>.
 *   - On success we render into the canvas, add `.js-ready` to the figure
 *     (CSS hides .fallback / shows .live), and set the fallback img.hidden=true
 *     ONLY after the first successful paint.
 *   - On any throw we set mount.dataset.failed='1' and leave the fallback SVG
 *     visible (never destroy it).
 *
 * Data: fetched through UDA.data (data.js), which is baseurl-aware. No CSV is
 * ever parsed in the browser. The only computation here is presentational
 * scaling/binning of values the exporter already computed; published
 * statistics (bias, LoA, conformal half-widths, headline metrics) are read as
 * constants and never recomputed.
 *
 * Colours come from CSS custom properties:
 *   --accent  (#2f5d8a steel-blue)  — chrome + non-lead series
 *   --series-2 (#b5403a brick-red)  — chart lead series + residual/error only
 * Read at render time via getComputedStyle so a single stylesheet governs them.
 */
(function (global) {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';

  // ---- shared geometry -----------------------------------------------------
  var VIEW_W = 720;          // SVG user-space width (scales responsively)
  var VIEW_H = 440;          // SVG user-space height
  var PAD = { t: 28, r: 24, b: 56, l: 64 };

  // ---- colour tokens (resolved from CSS at render time) --------------------
  function tokens(el) {
    var cs = getComputedStyle(el);
    function pick(name, fallback) {
      var v = cs.getPropertyValue(name);
      return (v && v.trim()) || fallback;
    }
    return {
      accent: pick('--accent', '#2f5d8a'),
      accentDeep: pick('--accent-deep', '#244a6e'),
      accentWash: pick('--accent-wash', '#eaf0f6'),
      lead: pick('--series-2', '#b5403a'),
      ink: pick('--ink', '#1a1a1a'),
      inkSoft: pick('--ink-soft', '#4a4a46'),
      inkMute: pick('--ink-mute', '#6f6f68'),
      hairline: pick('--hairline', '#dcdad2'),
      bgRaised: pick('--bg-raised', '#ffffff'),
      bgSunk: pick('--bg-sunk', '#f3f2ec')
    };
  }

  function reducedMotion() {
    return global.matchMedia &&
      global.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  // ---- tiny SVG helpers ----------------------------------------------------
  function el(name, attrs, text) {
    var n = document.createElementNS(SVGNS, name);
    if (attrs) {
      for (var k in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, k) && attrs[k] != null) {
          n.setAttribute(k, attrs[k]);
        }
      }
    }
    if (text != null) n.appendChild(document.createTextNode(text));
    return n;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  // Linear scale factory: domain [d0,d1] -> range [r0,r1].
  function scale(d0, d1, r0, r1) {
    var span = (d1 - d0) || 1;
    return function (v) { return r0 + (v - d0) / span * (r1 - r0); };
  }

  function fmt(v, dp) {
    if (v == null || isNaN(v)) return '';
    var s = (dp == null) ? v : Number(v).toFixed(dp);
    return s;
  }

  // "Nice" tick step for a span (1/2/5 * 10^n).
  function niceStep(span, target) {
    var raw = span / (target || 5);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    var step;
    if (norm < 1.5) step = 1;
    else if (norm < 3) step = 2;
    else if (norm < 7) step = 5;
    else step = 10;
    return step * mag;
  }

  function ticks(d0, d1, target) {
    var step = niceStep(d1 - d0, target);
    var start = Math.ceil(d0 / step) * step;
    var out = [];
    for (var v = start; v <= d1 + step * 1e-6; v += step) {
      out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    return out;
  }

  // ---- chart frame: axes, grid, labels -------------------------------------
  function frame(svg, T, opts) {
    var x0 = PAD.l, x1 = VIEW_W - PAD.r;
    var y0 = VIEW_H - PAD.b, y1 = PAD.t;
    var sx = scale(opts.xDomain[0], opts.xDomain[1], x0, x1);
    var sy = scale(opts.yDomain[0], opts.yDomain[1], y0, y1);

    var g = el('g');
    svg.appendChild(g);

    // horizontal grid + y ticks
    var yt = opts.yTicks || ticks(opts.yDomain[0], opts.yDomain[1], 5);
    yt.forEach(function (v) {
      var y = sy(v);
      g.appendChild(el('line', {
        x1: x0, y1: y, x2: x1, y2: y,
        stroke: T.hairline, 'stroke-width': 1, class: 'grid-line'
      }));
      g.appendChild(el('text', {
        x: x0 - 8, y: y + 4, 'text-anchor': 'end',
        class: 'axis-text', fill: T.inkMute, 'font-size': 12
      }, opts.yFmt ? opts.yFmt(v) : fmt(v)));
    });

    // x ticks (categorical or numeric)
    if (opts.xCats) {
      opts.xCats.forEach(function (c, i) {
        var x = c.x != null ? c.x : sx(i);
        g.appendChild(el('text', {
          x: x, y: y0 + 22, 'text-anchor': 'middle',
          class: 'axis-text', fill: T.inkSoft, 'font-size': 12
        }, c.label));
      });
    } else {
      var xt = opts.xTicks || ticks(opts.xDomain[0], opts.xDomain[1], 6);
      xt.forEach(function (v) {
        var x = sx(v);
        if (opts.xGrid) {
          g.appendChild(el('line', {
            x1: x, y1: y0, x2: x, y2: y1,
            stroke: T.hairline, 'stroke-width': 1, class: 'grid-line'
          }));
        }
        g.appendChild(el('text', {
          x: x, y: y0 + 22, 'text-anchor': 'middle',
          class: 'axis-text', fill: T.inkMute, 'font-size': 12
        }, opts.xFmt ? opts.xFmt(v) : fmt(v)));
      });
    }

    // axis baselines (hairline)
    g.appendChild(el('line', { x1: x0, y1: y0, x2: x1, y2: y0, stroke: T.ink, 'stroke-width': 1.25 }));
    g.appendChild(el('line', { x1: x0, y1: y0, x2: x0, y2: y1, stroke: T.ink, 'stroke-width': 1.25 }));

    // axis titles
    if (opts.xLabel) {
      g.appendChild(el('text', {
        x: (x0 + x1) / 2, y: VIEW_H - 10, 'text-anchor': 'middle',
        class: 'axis-text', fill: T.inkSoft, 'font-size': 13, 'font-weight': 500
      }, opts.xLabel));
    }
    if (opts.yLabel) {
      var ty = (y0 + y1) / 2;
      g.appendChild(el('text', {
        x: 16, y: ty, 'text-anchor': 'middle',
        transform: 'rotate(-90 16 ' + ty + ')',
        class: 'axis-text', fill: T.inkSoft, 'font-size': 13, 'font-weight': 500
      }, opts.yLabel));
    }

    return { sx: sx, sy: sy, x0: x0, x1: x1, y0: y0, y1: y1, layer: g };
  }

  function makeSvg(label) {
    var svg = el('svg', {
      viewBox: '0 0 ' + VIEW_W + ' ' + VIEW_H,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      width: '100%'
    });
    if (label) svg.setAttribute('aria-label', label);
    return svg;
  }

  // ---- hover readout (delegated, single tooltip per chart) -----------------
  function attachHover(svg, T) {
    var tip = el('g', { class: 'chart-tip', 'pointer-events': 'none' });
    tip.style.opacity = '0';
    var rect = el('rect', {
      rx: 4, ry: 4, fill: T.bgRaised, stroke: T.hairline, 'stroke-width': 1
    });
    var line1 = el('text', { 'font-size': 12, 'font-weight': 600, fill: T.ink, x: 8, y: 16 });
    var line2 = el('text', { 'font-size': 12, fill: T.inkSoft, x: 8, y: 32, class: 'chart-tip__sub' });
    tip.appendChild(rect);
    tip.appendChild(line1);
    tip.appendChild(line2);
    svg.appendChild(tip);

    return {
      show: function (px, py, title, sub) {
        line1.textContent = title;
        line2.textContent = sub || '';
        var w = Math.max(line1.getComputedTextLength
          ? line1.getComputedTextLength() : title.length * 7,
          sub ? (line2.getComputedTextLength ? line2.getComputedTextLength() : sub.length * 6) : 0) + 16;
        var h = sub ? 42 : 26;
        rect.setAttribute('width', w);
        rect.setAttribute('height', h);
        var tx = px + 12;
        if (tx + w > VIEW_W) tx = px - w - 12;
        var ty = py - h - 8;
        if (ty < PAD.t) ty = py + 12;
        tip.setAttribute('transform', 'translate(' + tx + ',' + ty + ')');
        tip.style.opacity = '1';
      },
      hide: function () { tip.style.opacity = '0'; }
    };
  }

  // ---- visually-hidden data table (accessibility fallback) -----------------
  function dataTable(rows, head, caption) {
    var wrap = document.createElement('div');
    wrap.className = 'visually-hidden';
    var t = document.createElement('table');
    if (caption) {
      var cap = document.createElement('caption');
      cap.textContent = caption;
      t.appendChild(cap);
    }
    if (head) {
      var thead = document.createElement('thead');
      var htr = document.createElement('tr');
      head.forEach(function (h) {
        var th = document.createElement('th');
        th.setAttribute('scope', 'col');
        th.textContent = h;
        htr.appendChild(th);
      });
      thead.appendChild(htr);
      t.appendChild(thead);
    }
    var tb = document.createElement('tbody');
    rows.forEach(function (r) {
      var tr = document.createElement('tr');
      r.forEach(function (c) {
        var td = document.createElement('td');
        td.textContent = c;
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    wrap.appendChild(t);
    return wrap;
  }

  // ==========================================================================
  // CHART RENDERERS
  // Each returns { svg, label, table, controls? } and is wrapped by render().
  // ==========================================================================

  // -- groupedBar: protocol comparison & architecture bake-off ---------------
  // Spec drives two modes:
  //   protocol mode  (hasProtocol): toggle image<->patient, metric is mape.
  //   bakeoff mode   (single series): one MAPE per backbone, lead highlighted.
  function renderGroupedBar(data, spec, T, mount) {
    var metric = (spec.metric || data.metric_default || 'mape');
    var rows, lead, isProtocol, protocol;

    if (data.backbones) {              // protocol_comparison.json
      isProtocol = true;
      protocol = spec.protocol || 'patient';   // default to the harder, headline-relevant view
      rows = data.backbones.slice();
    } else {                           // architecture_bakeoff.json
      isProtocol = false;
      rows = data.rows.slice();
    }

    var label, controls = null;

    function valueOf(r) {
      if (isProtocol) return r[protocol][metric];
      return r[metric];
    }
    function stdOf(r) {
      if (isProtocol) return r[protocol][metric + '_std'];
      return r[metric + '_std'];
    }

    var svg = makeSvg('');
    var host = mount;

    function draw() {
      clear(svg);
      // re-add base structure each redraw
      var ordered = rows.slice().sort(function (a, b) { return valueOf(a) - valueOf(b); });
      var maxV = 0;
      ordered.forEach(function (r) {
        var v = valueOf(r) + (stdOf(r) || 0);
        if (v > maxV) maxV = v;
      });
      maxV = Math.ceil(maxV / 5) * 5;

      var f = frame(svg, T, {
        xDomain: [0, ordered.length],
        yDomain: [0, maxV],
        yLabel: metric.toUpperCase() + (metric === 'mape' ? ' (%)' : ' (°)'),
        xCats: [],
        yFmt: function (v) { return fmt(v, 0); }
      });

      var band = (f.x1 - f.x0) / ordered.length;
      var bw = Math.min(band * 0.6, 64);
      var hov = attachHover(svg, T);

      ordered.forEach(function (r, i) {
        var cx = f.x0 + band * (i + 0.5);
        var v = valueOf(r);
        var y = f.sy(v);
        var sd = stdOf(r);
        var isLead = !!r.lead;
        var fill = isLead ? T.lead : T.accent;

        // error bar (std)
        if (sd != null) {
          var yTop = f.sy(v + sd), yBot = f.sy(Math.max(0, v - sd));
          f.layer.appendChild(el('line', {
            x1: cx, y1: yTop, x2: cx, y2: yBot,
            stroke: T.inkSoft, 'stroke-width': 1.25, 'pointer-events': 'none'
          }));
          f.layer.appendChild(el('line', {
            x1: cx - 5, y1: yTop, x2: cx + 5, y2: yTop,
            stroke: T.inkSoft, 'stroke-width': 1.25, 'pointer-events': 'none'
          }));
        }

        var bar = el('rect', {
          x: cx - bw / 2, y: y, width: bw, height: Math.max(0, f.y0 - y),
          fill: fill, 'fill-opacity': isLead ? 0.92 : 0.78,
          stroke: isLead ? T.lead : T.accentDeep, 'stroke-width': isLead ? 1.5 : 0.75,
          tabindex: 0, role: 'img',
          'aria-label': (r.label || r.backbone) + ': ' + fmt(v, 2) + (metric === 'mape' ? '%' : '°')
        });
        bar.style.cursor = 'pointer';
        bar.dataset.backbone = r.backbone;

        var subtxt = (sd != null ? '± ' + fmt(sd, 2) : '') +
          (isProtocol ? '  ·  ' + protocol + '-level' : (r.tier ? '  ·  ' + r.tier : ''));
        function over() {
          bar.setAttribute('fill-opacity', '1');
          hov.show(cx, y, (r.label || r.backbone) + '  ' + fmt(v, 2) + (metric === 'mape' ? '%' : '°'), subtxt);
        }
        function out() {
          bar.setAttribute('fill-opacity', isLead ? 0.92 : 0.78);
          hov.hide();
        }
        bar.addEventListener('mouseenter', over);
        bar.addEventListener('mouseleave', out);
        bar.addEventListener('focus', over);
        bar.addEventListener('blur', out);
        f.layer.appendChild(bar);

        // value label
        f.layer.appendChild(el('text', {
          x: cx, y: y - 6, 'text-anchor': 'middle',
          'font-size': 11, fill: isLead ? T.lead : T.inkSoft, 'font-weight': isLead ? 600 : 400
        }, fmt(v, 2)));

        // category label (rotated for readability)
        f.layer.appendChild(el('text', {
          x: cx, y: f.y0 + 16, 'text-anchor': 'end',
          transform: 'rotate(-30 ' + cx + ' ' + (f.y0 + 16) + ')',
          'font-size': 11, fill: isLead ? T.ink : T.inkMute, 'font-weight': isLead ? 600 : 400
        }, r.label || r.backbone));
      });
    }

    draw();

    if (isProtocol) {
      label = 'Bar chart of ' + metric.toUpperCase() +
        ' for each backbone; toggle between image-level and patient-level protocols. ' +
        'DenseNet201 (brick-red) leads.';
      controls = buildToggle([
        { key: 'image', text: 'Image-level' },
        { key: 'patient', text: 'Patient-level' }
      ], protocol, function (k) { protocol = k; draw(); }, 'Protocol');
    } else {
      label = 'Bar chart of patient-level cross-validation ' + metric.toUpperCase() +
        ' by ImageNet backbone; DenseNet201 (brick-red) is lowest. Newer architectures are not better.';
    }

    var tblRows = rows.slice().sort(function (a, b) { return valueOf(a) - valueOf(b); })
      .map(function (r) {
        return [r.label || r.backbone, fmt(valueOf(r), 2), stdOf(r) != null ? fmt(stdOf(r), 2) : '—'];
      });
    var table = dataTable(tblRows,
      ['Backbone', metric.toUpperCase(), '± std'],
      label);

    svg.setAttribute('aria-label', label);
    return { svg: svg, label: label, table: table, controls: controls };
  }

  // -- scatter: predicted vs actual (identity line + conformal band) ---------
  function renderScatter(data, spec, T) {
    var pts = data.points || [];
    if (!pts.length) throw new Error('empty dataset');   // keep static fallback (PE contract)
    var hw = data.conformal_halfwidth;
    var allT = pts.map(function (p) { return p.true; });
    var allP = pts.map(function (p) { return p.pred; });
    // Domain follows the data (absolute insonation angle, ~18-164°); no artificial
    // ±60-65° padding, which would otherwise leave a large empty band below ~18°.
    var lo = Math.min.apply(null, allT.concat(allP));
    var hi = Math.max.apply(null, allT.concat(allP));
    lo = Math.floor(lo / 10) * 10;
    hi = Math.ceil(hi / 10) * 10;

    var svg = makeSvg('');
    var f = frame(svg, T, {
      xDomain: [lo, hi], yDomain: [lo, hi], xGrid: true,
      xLabel: 'Reference angle θ (°)', yLabel: 'Predicted angle θ̂ (°)',
      xFmt: function (v) { return fmt(v, 0); }, yFmt: function (v) { return fmt(v, 0); }
    });

    // conformal band around identity (±halfwidth)
    if (hw != null) {
      var bandPts = [
        [f.sx(lo), f.sy(lo + hw)], [f.sx(hi), f.sy(hi + hw)],
        [f.sx(hi), f.sy(hi - hw)], [f.sx(lo), f.sy(lo - hw)]
      ];
      f.layer.appendChild(el('polygon', {
        points: bandPts.map(function (p) { return p[0] + ',' + p[1]; }).join(' '),
        fill: T.accentWash, 'fill-opacity': 0.7, stroke: 'none'
      }));
    }
    // identity line
    f.layer.appendChild(el('line', {
      x1: f.sx(lo), y1: f.sy(lo), x2: f.sx(hi), y2: f.sy(hi),
      stroke: T.ink, 'stroke-width': 1.25, 'stroke-dasharray': '5 4', 'pointer-events': 'none'
    }));

    var hov = attachHover(svg, T);
    var dots = el('g');
    f.layer.appendChild(dots);
    pts.forEach(function (p) {
      var isBase = !!p.base;
      var c = el('circle', {
        cx: f.sx(p.true), cy: f.sy(p.pred),
        r: isBase ? 3.2 : 2,
        fill: isBase ? T.lead : T.accent,
        'fill-opacity': isBase ? 0.85 : 0.32,
        stroke: isBase ? T.lead : 'none', 'stroke-width': isBase ? 0.5 : 0
      });
      c.dataset.id = p.id;
      c.dataset.rot = p.rot;
      c.style.cursor = 'pointer';
      c.addEventListener('mouseenter', function () {
        c.setAttribute('r', isBase ? 4.5 : 3.5);
        c.setAttribute('fill-opacity', '1');
        hov.show(f.sx(p.true), f.sy(p.pred),
          'θ ' + fmt(p.true, 0) + '°  →  θ̂ ' + fmt(p.pred, 1) + '°',
          'residual ' + (p.err >= 0 ? '+' : '') + fmt(p.err, 1) + '°' +
          (p.rot != null ? '  ·  rot ' + fmt(p.rot, 0) + '°' : ''));
      });
      c.addEventListener('mouseleave', function () {
        c.setAttribute('r', isBase ? 3.2 : 2);
        c.setAttribute('fill-opacity', isBase ? 0.85 : 0.32);
        hov.hide();
      });
      // cross-link to demo on base-image click (clinical explorer)
      if (isBase && p.id != null) {
        c.addEventListener('click', function () {
          var root = (global.UDA && typeof UDA.assetBase === 'string' && UDA.assetBase)
            ? UDA.assetBase.replace(/assets\/+$/, '')   // ".../assets/" -> site root
            : '/';
          try { global.location.href = root + 'clinical/#prediction-demo'; }
          catch (e) { /* ignore */ }
        });
      }
      dots.appendChild(c);
    });

    var label = 'Scatter of predicted versus reference Doppler angle for all ' + pts.length +
      ' samples; points cluster on the dashed identity line within a ' +
      (hw != null ? '±' + fmt(hw, 2) + '° conformal band' : 'tight band') +
      '. Base (un-rotated) images are emphasised in brick-red.';
    svg.setAttribute('aria-label', label);

    var baseRows = pts.filter(function (p) { return p.base; })
      .map(function (p) { return [p.id, fmt(p.true, 0), fmt(p.pred, 1), (p.err >= 0 ? '+' : '') + fmt(p.err, 1)]; });
    var table = dataTable(baseRows,
      ['Image', 'Reference θ (°)', 'Predicted θ̂ (°)', 'Residual (°)'],
      'Predicted versus reference angle, base images.');

    return { svg: svg, label: label, table: table };
  }

  // -- errorScatter: signed error vs reference angle (with binned means) -----
  function renderErrorScatter(data, spec, T) {
    var pts = data.points || [];
    if (!pts.length) throw new Error('empty dataset');   // keep static fallback (PE contract)
    var bins = data.error_bins || [];
    var bias = data.bias_line;
    var xs = pts.map(function (p) { return p.true; });
    var es = pts.map(function (p) { return p.err; });
    // Domain follows the reference-angle data (~18-164°), not a fixed ±60° range.
    var xlo = Math.floor(Math.min.apply(null, xs) / 10) * 10;
    var xhi = Math.ceil(Math.max.apply(null, xs) / 10) * 10;
    var emax = Math.max(30, Math.ceil(Math.max.apply(null, es.map(Math.abs)) / 5) * 5);

    var svg = makeSvg('');
    var f = frame(svg, T, {
      xDomain: [xlo, xhi], yDomain: [-emax, emax], xGrid: true,
      xLabel: 'Reference angle θ (°)', yLabel: 'Signed error θ̂ − θ (°)',
      xFmt: function (v) { return fmt(v, 0); }, yFmt: function (v) { return fmt(v, 0); }
    });

    // zero line
    f.layer.appendChild(el('line', {
      x1: f.x0, y1: f.sy(0), x2: f.x1, y2: f.sy(0),
      stroke: T.ink, 'stroke-width': 1, 'pointer-events': 'none'
    }));
    // mean-bias line (brick-red)
    if (bias != null) {
      f.layer.appendChild(el('line', {
        x1: f.x0, y1: f.sy(bias), x2: f.x1, y2: f.sy(bias),
        stroke: T.lead, 'stroke-width': 1.25, 'stroke-dasharray': '6 4', 'pointer-events': 'none'
      }));
      f.layer.appendChild(el('text', {
        x: f.x1 - 4, y: f.sy(bias) - 5, 'text-anchor': 'end',
        'font-size': 11, fill: T.lead, 'font-weight': 600
      }, 'mean ' + (bias >= 0 ? '+' : '') + fmt(bias, 2) + '°'));
    }

    var hov = attachHover(svg, T);
    var dots = el('g');
    f.layer.appendChild(dots);
    pts.forEach(function (p) {
      var isBase = !!p.base;
      var c = el('circle', {
        cx: f.sx(p.true), cy: f.sy(p.err),
        r: isBase ? 3 : 1.8,
        fill: isBase ? T.lead : T.accent,
        'fill-opacity': isBase ? 0.8 : 0.28
      });
      c.addEventListener('mouseenter', function () {
        c.setAttribute('fill-opacity', '1');
        hov.show(f.sx(p.true), f.sy(p.err),
          'θ ' + fmt(p.true, 0) + '°  error ' + (p.err >= 0 ? '+' : '') + fmt(p.err, 1) + '°',
          p.id != null ? String(p.id) : '');
      });
      c.addEventListener('mouseleave', function () {
        c.setAttribute('fill-opacity', isBase ? 0.8 : 0.28);
        hov.hide();
      });
      dots.appendChild(c);
    });

    // binned mean error (overlaid step markers)
    bins.forEach(function (b) {
      var cx = f.sx((b.lo + b.hi) / 2);
      f.layer.appendChild(el('circle', {
        cx: cx, cy: f.sy(b.mean_err), r: 4,
        fill: T.bgRaised, stroke: T.ink, 'stroke-width': 1.5, 'pointer-events': 'none'
      }));
    });

    var label = 'Signed prediction error against reference angle for all ' + pts.length +
      ' samples; the brick-red dashed line marks the ' +
      (bias != null ? (bias >= 0 ? '+' : '') + fmt(bias, 2) + '° mean bias' : 'mean bias') +
      '. Hollow markers are the per-bin mean error.';
    svg.setAttribute('aria-label', label);

    var binRows = bins.map(function (b) {
      return [fmt(b.lo, 0) + ' to ' + fmt(b.hi, 0), fmt(b.mean_err, 2), b.n];
    });
    var table = dataTable(binRows,
      ['Angle bin (°)', 'Mean error (°)', 'n'],
      'Mean signed error by 10° reference-angle bin.');

    return { svg: svg, label: label, table: table };
  }

  // -- blandAltman: agreement (per-sample <-> per-patient toggle) ------------
  function renderBlandAltman(data, spec, T, mount) {
    var view = spec.view || 'per_sample';

    var svg = makeSvg('');

    function draw() {
      clear(svg);
      var block = data[view];
      var pts = block.points || [];
      if (!pts.length) throw new Error('empty dataset');   // keep static fallback (PE contract)
      var means = pts.map(function (p) { return p.mean; });
      var diffs = pts.map(function (p) { return p.diff; });
      var xlo = Math.floor(Math.min.apply(null, means) / 10) * 10;
      var xhi = Math.ceil(Math.max.apply(null, means) / 10) * 10;
      var dabs = Math.max(
        Math.abs(block.loa_lo), Math.abs(block.loa_hi),
        Math.max.apply(null, diffs.map(Math.abs))
      );
      var dmax = Math.ceil(dabs / 5) * 5;

      var f = frame(svg, T, {
        xDomain: [xlo, xhi], yDomain: [-dmax, dmax], xGrid: true,
        xLabel: 'Mean of model and reference (°)',
        yLabel: 'Model − reference (°)',
        xFmt: function (v) { return fmt(v, 0); }, yFmt: function (v) { return fmt(v, 0); }
      });

      // zero
      f.layer.appendChild(el('line', {
        x1: f.x0, y1: f.sy(0), x2: f.x1, y2: f.sy(0),
        stroke: T.hairline, 'stroke-width': 1
      }));
      // bias line (brick-red) + LoA (dashed accent)
      function hline(v, color, dash, lbl) {
        f.layer.appendChild(el('line', {
          x1: f.x0, y1: f.sy(v), x2: f.x1, y2: f.sy(v),
          stroke: color, 'stroke-width': 1.25, 'stroke-dasharray': dash || null
        }));
        f.layer.appendChild(el('text', {
          x: f.x1 - 4, y: f.sy(v) - 5, 'text-anchor': 'end',
          'font-size': 11, fill: color, 'font-weight': 600
        }, lbl + ' ' + (v >= 0 ? '+' : '') + fmt(v, 2) + '°'));
      }
      hline(block.bias, T.lead, null, 'bias');
      hline(block.loa_hi, T.accent, '5 4', '+1.96 SD');
      hline(block.loa_lo, T.accent, '5 4', '−1.96 SD');

      var hov = attachHover(svg, T);
      var dots = el('g');
      f.layer.appendChild(dots);
      pts.forEach(function (p) {
        var c = el('circle', {
          cx: f.sx(p.mean), cy: f.sy(p.diff),
          r: view === 'per_patient' ? 4 : 2.2,
          fill: T.accent, 'fill-opacity': view === 'per_patient' ? 0.8 : 0.3
        });
        c.addEventListener('mouseenter', function () {
          c.setAttribute('fill-opacity', '1');
          hov.show(f.sx(p.mean), f.sy(p.diff),
            'diff ' + (p.diff >= 0 ? '+' : '') + fmt(p.diff, 1) + '°',
            (p.patient != null ? 'patient ' + p.patient : (p.id != null ? String(p.id) : '')));
        });
        c.addEventListener('mouseleave', function () {
          c.setAttribute('fill-opacity', view === 'per_patient' ? 0.8 : 0.3);
          hov.hide();
        });
        dots.appendChild(c);
      });
    }

    draw();

    var controls = buildToggle([
      { key: 'per_sample', text: 'Per sample' },
      { key: 'per_patient', text: 'Per patient' }
    ], view, function (k) { view = k; draw(); }, 'Aggregation');

    var b = data[view];
    var label = 'Bland–Altman agreement plot, model minus reference; brick-red mean-bias line at ' +
      fmt(data.per_sample.bias, 2) + '° (per sample) with ±1.96 SD limits of agreement. ' +
      'This is method-versus-reference, not inter-observer, agreement.';
    svg.setAttribute('aria-label', label);

    var rows = [
      ['Per sample (n=' + data.per_sample.n + ')', fmt(data.per_sample.bias, 2),
        fmt(data.per_sample.loa_lo, 2) + ' to ' + fmt(data.per_sample.loa_hi, 2)],
      ['Per patient (n=' + data.per_patient.n + ')', fmt(data.per_patient.bias, 2),
        fmt(data.per_patient.loa_lo, 2) + ' to ' + fmt(data.per_patient.loa_hi, 2)]
    ];
    var table = dataTable(rows,
      ['Aggregation', 'Bias (°)', 'Limits of agreement (°)'],
      'Bland–Altman bias and limits of agreement.');

    return { svg: svg, label: label, table: table, controls: controls };
  }

  // -- calibrationStep: conformal coverage vs nominal ------------------------
  function renderCalibration(data, spec, T) {
    var levels = (data.levels || []).slice().sort(function (a, b) { return a.nominal - b.nominal; });
    if (!levels.length) throw new Error('empty dataset');   // keep static fallback (PE contract)

    var svg = makeSvg('');
    var maxHw = Math.ceil(Math.max.apply(null, levels.map(function (l) { return l.halfwidth; })) / 5) * 5 + 5;

    var f = frame(svg, T, {
      xDomain: [0.75, 1.0], yDomain: [0, maxHw], xGrid: true,
      xLabel: 'Coverage level', yLabel: 'Interval half-width (°)',
      xTicks: [0.80, 0.85, 0.90, 0.95, 1.0],
      xFmt: function (v) { return (v * 100).toFixed(0) + '%'; },
      yFmt: function (v) { return fmt(v, 0); }
    });

    var hov = attachHover(svg, T);

    // nominal coverage diagonal reference (empirical vs nominal is the honest check)
    levels.forEach(function (l, i) {
      var cx = f.sx(l.nominal);
      var cy = f.sy(l.halfwidth);
      var isMid = Math.abs(l.nominal - 0.90) < 1e-6;  // 90% is the headline band
      var color = isMid ? T.lead : T.accent;

      // half-width bar from baseline
      f.layer.appendChild(el('line', {
        x1: cx, y1: f.y0, x2: cx, y2: cy,
        stroke: color, 'stroke-width': isMid ? 2 : 1.25, 'stroke-opacity': 0.5
      }));
      var dot = el('circle', {
        cx: cx, cy: cy, r: isMid ? 6 : 4.5,
        fill: color, stroke: T.bgRaised, 'stroke-width': 1.25
      });
      dot.style.cursor = 'pointer';
      dot.addEventListener('mouseenter', function () {
        hov.show(cx, cy,
          (l.nominal * 100).toFixed(0) + '% nominal  ·  ±' + fmt(l.halfwidth, 2) + '°',
          'empirical ' + (l.empirical * 100).toFixed(1) + '%');
      });
      dot.addEventListener('mouseleave', function () { hov.hide(); });
      f.layer.appendChild(dot);

      // empirical-coverage annotation
      f.layer.appendChild(el('text', {
        x: cx, y: cy - 12, 'text-anchor': 'middle',
        'font-size': 11, fill: color, 'font-weight': isMid ? 600 : 400
      }, '±' + fmt(l.halfwidth, 1) + '°'));
      f.layer.appendChild(el('text', {
        x: cx, y: cy + 18, 'text-anchor': 'middle',
        'font-size': 10, fill: T.inkMute
      }, (l.empirical * 100).toFixed(1) + '% emp.'));
    });

    var label = 'Split-conformal interval half-width against nominal coverage; the 90% band (brick-red) ' +
      'is ±' + fmt(levels.filter(function (l) { return Math.abs(l.nominal - 0.9) < 1e-6; })[0].halfwidth, 2) +
      '°, with empirical coverage at or above nominal. From a single seed-42 patient-disjoint split.';
    svg.setAttribute('aria-label', label);

    var rows = levels.map(function (l) {
      return [(l.nominal * 100).toFixed(0) + '%', '±' + fmt(l.halfwidth, 2), (l.empirical * 100).toFixed(1) + '%'];
    });
    var table = dataTable(rows,
      ['Nominal coverage', 'Half-width (°)', 'Empirical coverage'],
      'Conformal interval half-widths and empirical coverage.');

    return { svg: svg, label: label, table: table };
  }

  // -- slopegraph: image -> patient (or single -> ensemble) ------------------
  // Optional helper kept for the spec's slopegraph type; degrades gracefully.
  function renderSlopegraph(data, spec, T) {
    var series = spec.series || data.series || [];
    var leftLabel = spec.leftLabel || 'Image-level';
    var rightLabel = spec.rightLabel || 'Patient-level';
    var allV = [];
    series.forEach(function (s) { allV.push(s.left, s.right); });
    var vlo = Math.floor(Math.min.apply(null, allV) / 2) * 2;
    var vhi = Math.ceil(Math.max.apply(null, allV) / 2) * 2;

    var svg = makeSvg('');
    var f = frame(svg, T, {
      xDomain: [0, 1], yDomain: [vlo, vhi],
      xCats: [{ label: leftLabel, x: PAD.l + 40 }, { label: rightLabel, x: VIEW_W - PAD.r - 40 }],
      yLabel: spec.metricLabel || 'MAPE (%)',
      yFmt: function (v) { return fmt(v, 0); }
    });
    var xL = f.x0 + 40, xR = f.x1 - 40;
    var hov = attachHover(svg, T);

    series.forEach(function (s) {
      var color = s.lead ? T.lead : T.accent;
      var yL = f.sy(s.left), yR = f.sy(s.right);
      f.layer.appendChild(el('line', {
        x1: xL, y1: yL, x2: xR, y2: yR,
        stroke: color, 'stroke-width': s.lead ? 2.5 : 1.5, 'stroke-opacity': 0.85
      }));
      [[xL, yL, s.left, 'end', -8], [xR, yR, s.right, 'start', 8]].forEach(function (e) {
        f.layer.appendChild(el('circle', { cx: e[0], cy: e[1], r: 3.5, fill: color }));
        f.layer.appendChild(el('text', {
          x: e[0] + e[4], y: e[1] + 4, 'text-anchor': e[3],
          'font-size': 11, fill: color, 'font-weight': s.lead ? 600 : 400
        }, fmt(e[2], 2)));
      });
      f.layer.appendChild(el('text', {
        x: xR + 8, y: yR - 8, 'text-anchor': 'start',
        'font-size': 11, fill: T.inkSoft
      }, s.label || ''));
    });

    var label = 'Slopegraph from ' + leftLabel + ' to ' + rightLabel + ' showing the protocol spread.';
    svg.setAttribute('aria-label', label);
    var table = dataTable(series.map(function (s) {
      return [s.label || '', fmt(s.left, 2), fmt(s.right, 2)];
    }), [' ', leftLabel, rightLabel], label);

    return { svg: svg, label: label, table: table };
  }

  // ---- control widgets (toggle button group) ------------------------------
  function buildToggle(options, active, onChange, groupLabel) {
    var wrap = document.createElement('div');
    wrap.className = 'controls';

    var grp = document.createElement('div');
    grp.className = 'btn-group';
    grp.setAttribute('role', 'group');
    if (groupLabel) grp.setAttribute('aria-label', groupLabel);

    var btns = [];
    options.forEach(function (o) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'btn';
      b.textContent = o.text;
      b.setAttribute('aria-pressed', o.key === active ? 'true' : 'false');
      if (o.key === active) b.classList.add('is-active');
      b.addEventListener('click', function () {
        if (o.key === active) return;
        active = o.key;
        btns.forEach(function (bb) {
          var on = bb._key === active;
          bb.setAttribute('aria-pressed', on ? 'true' : 'false');
          bb.classList.toggle('is-active', on);
        });
        onChange(active);
      });
      b._key = o.key;
      btns.push(b);
      grp.appendChild(b);
    });
    wrap.appendChild(grp);
    return wrap;
  }

  // ==========================================================================
  // DISPATCH + PROGRESSIVE-ENHANCEMENT WRAPPER
  // ==========================================================================

  var RENDERERS = {
    scatter: renderScatter,
    groupedBar: renderGroupedBar,
    slopegraph: renderSlopegraph,
    errorScatter: renderErrorScatter,
    blandAltman: renderBlandAltman,
    calibrationStep: renderCalibration
  };

  // Mount registry: which JSON + spec each canonical chart id uses.
  // Pages expose the <figure>/canvas; we degrade silently if a mount is absent.
  var REGISTRY = {
    'chart-protocol-comparison': { type: 'groupedBar', data: 'protocol_comparison' },
    'chart-architecture-bakeoff': { type: 'groupedBar', data: 'architecture_bakeoff' },
    'chart-pred-vs-actual': { type: 'scatter', data: 'pred_vs_actual' },
    'chart-error-vs-angle': { type: 'errorScatter', data: 'pred_vs_actual' },
    'chart-bland-altman': { type: 'blandAltman', data: 'bland_altman' },
    'chart-conformal-calibration': { type: 'calibrationStep', data: 'conformal' }
    // chart-tuning-history is intentionally absent: static SVG only in v-00.
  };

  // Locate the enclosing <figure class="widget"> and its fallback <img>.
  function widgetParts(mount) {
    var fig = mount.closest ? mount.closest('.widget, figure') : null;
    var fallback = fig ? fig.querySelector('img.fallback, .fallback') : null;
    return { fig: fig, fallback: fallback };
  }

  function markFailed(mount) {
    if (mount) mount.dataset.failed = '1';
    // fallback stays visible by default — do nothing else.
  }

  // Render `spec` into the canvas element with id `mountId`.
  function render(mountId, spec) {
    var mount = document.getElementById(mountId);
    if (!mount) return Promise.resolve(false);   // page didn't expose it
    if (mount.dataset.rendered === '1') return Promise.resolve(true);

    var parts = widgetParts(mount);
    spec = spec || {};
    var rec = REGISTRY[mountId] || {};
    var type = spec.type || rec.type;
    var dataName = spec.data || rec.data;
    var renderer = RENDERERS[type];

    if (!renderer || !dataName) {
      markFailed(mount);
      return Promise.resolve(false);
    }

    var dataApi = global.UDA && global.UDA.data;
    if (!dataApi || typeof dataApi.load !== 'function') {
      markFailed(mount);
      return Promise.resolve(false);
    }

    return dataApi.load(dataName).then(function (data) {
      if (!data) throw new Error('no data for ' + dataName);
      var T = tokens(mount);
      var out = renderer(data, spec, T, mount);

      // first successful paint
      clear(mount);
      if (out.controls) mount.appendChild(out.controls);
      mount.appendChild(out.svg);
      if (out.table) mount.appendChild(out.table);

      // accessibility: the accessible name lives on the inner <svg> (role="img");
      // the mount must NOT be role="img" or its descendant table/controls/bars
      // would be pruned from the a11y tree.

      // reveal live, hide fallback — only AFTER paint
      if (parts.fig) parts.fig.classList.add('js-ready');
      if (parts.fallback) parts.fallback.hidden = true;
      mount.dataset.rendered = '1';
      mount.dataset.failed = '';
      return true;
    }).catch(function (err) {
      if (global.console && console.warn) {
        console.warn('[UDA.charts] ' + mountId + ' render failed; keeping fallback.', err);
      }
      markFailed(mount);
      return false;
    });
  }

  // Auto-bootstrap: render every known chart mount that the current page exposes.
  function init() {
    Object.keys(REGISTRY).forEach(function (id) {
      var mount = document.getElementById(id);
      if (mount) render(id);   // spec pulled from REGISTRY
    });
  }

  global.UDA = global.UDA || {};
  global.UDA.charts = { render: render, init: init, RENDERERS: RENDERERS };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})(window);
