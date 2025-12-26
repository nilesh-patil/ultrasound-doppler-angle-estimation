/*!
 * data.js — shared, baseurl-aware JSON loader for the project site.
 *
 * Why this exists: charts.js (UDA.charts.render) and demo.js both call
 * `UDA.data.load(name)` to fetch their backing JSON. Without this file that
 * call is undefined, so every interactive chart and the prediction explorer
 * silently fail to their static SVG fallbacks. This module defines
 * `window.UDA.data` and `window.UDA.assetBase`.
 *
 * Baseurl awareness: the site is served under a Jekyll baseurl
 * (/ultrasound-doppler-angle-estimation). The layout emits THIS script tag as
 *   <script defer src=".../assets/js/data.js"
 *           data-base="{{ '/assets/data/' | relative_url }}"></script>
 * so the correct, baseurl-prefixed data directory arrives as our own
 * [data-base] attribute. We read it from document.currentScript (resolved at
 * parse time, before any deferred callbacks fire).
 *
 * Contract honoured by callers:
 *   - charts.js calls load('pred_vs_actual') etc. — NO extension; we append
 *     '.json' when the name lacks one.
 *   - demo.js calls load('demo_predictions.json') then load('predictions.json')
 *     — names already carry '.json'; we must NOT double-append.
 *   - demo.js reads window.UDA.assetBase (the /assets/ root, one level above
 *     /assets/data/) to resolve thumbnail image URLs.
 *
 * Caching: each distinct file is fetched at most once; the in-flight (and then
 * resolved) Promise is memoised. A failed fetch is NOT cached, so a transient
 * error can be retried, and rejections propagate to callers so the static
 * fallback SVG stays visible.
 *
 * Loads first (it is listed first in every page's `scripts:` front-matter and
 * all page scripts are `defer`, so execution order is guaranteed before
 * charts.js / demo.js run).
 */
(function (global) {
  'use strict';

  // The <script> element for this file. With classic deferred scripts,
  // document.currentScript is the executing element during initial evaluation.
  var SELF = document.currentScript;

  // Resolve the baseurl-aware data directory, e.g.
  //   /ultrasound-doppler-angle-estimation/assets/data/
  // Preference order: our own [data-base] attr (emitted by the layout) →
  // any [data-base] on another emitted page script → [data-asset-base] on
  // <html> → a relative "../data/" hop from /assets/js/ as a last resort.
  function readDataBase() {
    var b = SELF && SELF.getAttribute && SELF.getAttribute('data-base');
    if (b) return b;

    // Fall back to scanning for any sibling script carrying data-base
    // (every page script is emitted with the same attribute by the layout).
    try {
      var scripts = document.querySelectorAll('script[data-base]');
      for (var i = 0; i < scripts.length; i++) {
        var v = scripts[i].getAttribute('data-base');
        if (v) return v;
      }
    } catch (e) { /* querySelectorAll unavailable — ignore */ }

    var html = document.documentElement;
    var ab = html && html.getAttribute && html.getAttribute('data-asset-base');
    if (ab) return ab.replace(/\/+$/, '') + '/data/';

    // data.js lives at /assets/js/; /assets/data/ is one level up + data/.
    return '../data/';
  }

  // Normalise to a directory path ending in exactly one slash.
  function withTrailingSlash(p) {
    return p.replace(/\/+$/, '') + '/';
  }

  var DATA_BASE = withTrailingSlash(readDataBase());

  // The /assets/ root is the data dir minus a trailing "data/" segment. demo.js
  // reads this to build thumbnail URLs under /assets/images/bmode/.
  function assetsRootFrom(dataBase) {
    return dataBase.replace(/data\/$/, '');
  }
  var ASSET_BASE = assetsRootFrom(DATA_BASE);

  // Single-fetch promise cache, keyed by the resolved URL.
  var cache = Object.create(null);

  // Build the full URL for a requested data file. `name` may or may not carry
  // a '.json' extension; we append it only when absent so both calling
  // conventions (charts.js bare name, demo.js explicit '.json') work.
  function urlFor(name) {
    var file = String(name);
    if (!/\.json$/i.test(file)) file += '.json';
    return DATA_BASE + file;
  }

  /**
   * Load a JSON data file by name, with caching.
   * @param {string} name e.g. 'pred_vs_actual' or 'demo_predictions.json'
   * @returns {Promise<Object>} resolves with parsed JSON; rejects on failure
   *   so callers can keep their static fallback.
   */
  function load(name) {
    if (name == null) return Promise.reject(new Error('UDA.data.load: name is required'));
    var url = urlFor(name);

    if (cache[url]) return cache[url];

    var p = fetch(url, { credentials: 'same-origin' })
      .then(function (resp) {
        if (!resp.ok) {
          throw new Error('UDA.data.load: ' + url + ' -> HTTP ' + resp.status);
        }
        return resp.json();
      })
      .catch(function (err) {
        // Do not cache failures, so a later call may retry; rethrow so the
        // caller's .catch keeps the fallback SVG visible.
        delete cache[url];
        throw err;
      });

    cache[url] = p;
    return p;
  }

  global.UDA = global.UDA || {};
  // assetBase: the /assets/ root (baseurl-aware), consumed by demo.js.
  global.UDA.assetBase = ASSET_BASE;
  global.UDA.data = {
    load: load,
    // Exposed for diagnostics / advanced callers; not required by the contract.
    base: DATA_BASE,
    url: urlFor
  };

})(window);
