/*
 * SAM3 extension — txt2img Workspaces (v0.10.0)
 *
 * Three browser-local txt2img workspaces in one Forge tab.  The manager is
 * deliberately front-end only: it does not patch Forge, does not persist
 * images/files/results, and does not touch global checkpoint/VAE quicksettings.
 *
 * Gradio 4.40 notes:
 *   - The DOM is the source of truth for current values.  /config is fetched
 *     only for component ids, raw types and choice metadata.
 *   - Values are restored through Gradio's typed `prop_change` event.  This is
 *     important for dropdowns (including XYZ's anonymous multiselects), where
 *     assigning input.value only changes the search box.
 *   - elem_id is not globally unique in Forge.  Stable ancestors are used to
 *     disambiguate duplicate ids such as txt2img_scheduler.
 */

(function () {
    "use strict";

    var VERSION = "0.10.0";
    var SCHEMA = 1;
    var STORAGE_KEY = "sam-extra.workspace-manager.v1";
    var ACTIVE_KEY = "sam-extra.workspace-manager.active.v1";
    var SAVE_DELAY_MS = 750;
    var MAX_STORAGE_BYTES = 4 * 1024 * 1024;
    var ALLOWED_TYPES = {
        textbox: true,
        number: true,
        slider: true,
        checkbox: true,
        radio: true,
        checkboxgroup: true,
        dropdown: true
    };

    var configComponents = [];
    var configById = Object.create(null);
    var configPromise = null;
    var initialized = false;
    var bootstrapping = false;
    var restoring = 0;
    var switching = false;
    var resetting = false;
    var externalConflict = false;
    var dirty = false;
    var saveTimer = null;
    var toolbar = null;
    var generationPane = null;
    var restoreBusyRoots = [];
    var capturedComponentIds = new Set();
    var lastCatalogStats = { captured: 0, skipped: 0, collisions: 0 };
    var activeSlot = readActiveSlot();
    var tabId = readTabId();
    var knownSlotRevision = 0;

    function app() {
        return (typeof gradioApp === "function") ? gradioApp() : document;
    }

    function delay(ms) {
        return new Promise(function (resolve) { setTimeout(resolve, ms); });
    }

    async function waitFor(predicate, timeoutMs) {
        var deadline = Date.now() + (timeoutMs || 2000);
        while (Date.now() < deadline) {
            try { if (predicate()) return true; } catch (e) {}
            await delay(50);
        }
        return false;
    }

    function cloneJson(value) {
        return JSON.parse(JSON.stringify(value));
    }

    function safeSessionGet(key) {
        try { return window.sessionStorage.getItem(key); } catch (e) { return null; }
    }

    function safeSessionSet(key, value) {
        try { window.sessionStorage.setItem(key, value); } catch (e) {}
    }

    function readActiveSlot() {
        var raw = safeSessionGet(ACTIVE_KEY);
        return (raw === "1" || raw === "2" || raw === "3") ? raw : "1";
    }

    function readTabId() {
        // Do not persist this in sessionStorage: duplicating a browser tab can
        // clone sessionStorage and would make two writers look identical.
        var value;
        try {
            value = window.crypto && typeof window.crypto.randomUUID === "function"
                ? window.crypto.randomUUID()
                : (Date.now().toString(36) + "-" + Math.random().toString(36).slice(2));
        } catch (e) {
            value = Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
        }
        return value;
    }

    function emptyStore() {
        return {
            schema: SCHEMA,
            revision: 0,
            updatedAt: null,
            updatedBy: null,
            slotRevisions: { "1": 0, "2": 0, "3": 0 },
            slots: { "1": null, "2": null, "3": null }
        };
    }

    function isPlainObject(value) {
        return !!value && typeof value === "object" && !Array.isArray(value);
    }

    function sanitizeSnapshot(snapshot) {
        if (!isPlainObject(snapshot) || !isPlainObject(snapshot.controls)) return null;
        var controls = Object.create(null);
        Object.keys(snapshot.controls).forEach(function (key) {
            if (key === "__proto__" || key === "constructor" || key === "prototype") return;
            var record = snapshot.controls[key];
            if (!isPlainObject(record) || typeof record.kind !== "string") return;
            controls[key] = { kind: record.kind, value: record.value };
            if (typeof record.active === "boolean") controls[key].active = record.active;
        });
        return {
            revision: Number(snapshot.revision) || 0,
            createdAt: typeof snapshot.createdAt === "string" ? snapshot.createdAt : null,
            savedAt: typeof snapshot.savedAt === "string" ? snapshot.savedAt : null,
            writer: typeof snapshot.writer === "string" ? snapshot.writer : null,
            controls: controls
        };
    }

    function normalizeStore(raw) {
        var result = emptyStore();
        if (!isPlainObject(raw) || Number(raw.schema) !== SCHEMA) return result;
        result.revision = Number(raw.revision) || 0;
        result.updatedAt = typeof raw.updatedAt === "string" ? raw.updatedAt : null;
        result.updatedBy = typeof raw.updatedBy === "string" ? raw.updatedBy : null;
        if (isPlainObject(raw.slots)) {
            ["1", "2", "3"].forEach(function (slot) {
                result.slots[slot] = sanitizeSnapshot(raw.slots[slot]);
                var storedClock = isPlainObject(raw.slotRevisions) ? Number(raw.slotRevisions[slot]) || 0 : 0;
                var snapshotClock = result.slots[slot] ? Number(result.slots[slot].revision) || 0 : 0;
                result.slotRevisions[slot] = Math.max(storedClock, snapshotClock);
            });
        }
        return result;
    }

    function slotRevision(store, slot) {
        return store && store.slotRevisions ? Number(store.slotRevisions[slot]) || 0 : 0;
    }

    function readStore() {
        try {
            var raw = window.localStorage.getItem(STORAGE_KEY);
            return raw ? normalizeStore(JSON.parse(raw)) : emptyStore();
        } catch (e) {
            setStatus("저장 데이터를 읽지 못했습니다", "error");
            console.warn("[SAM3 Workspaces] localStorage read failed:", e);
            return emptyStore();
        }
    }

    function writeStore(store) {
        try {
            var serialized = JSON.stringify(store);
            if (serialized.length > MAX_STORAGE_BYTES) {
                throw new Error("workspace data exceeds 4 MiB safety limit");
            }
            window.localStorage.setItem(STORAGE_KEY, serialized);
            return true;
        } catch (e) {
            setStatus("브라우저 저장 공간이 부족합니다", "error");
            console.warn("[SAM3 Workspaces] localStorage write failed:", e);
            return false;
        }
    }

    function setStatus(message, tone) {
        if (!toolbar) return;
        var el = toolbar.querySelector("[data-workspace-status]");
        if (!el) return;
        el.textContent = message || "";
        el.setAttribute("data-tone", tone || "normal");
    }

    function stableId(id) {
        if (!id) return false;
        if (/^(?:component|input|range_id)[-_]\d+$/i.test(id)) return false;
        if (/^uuid[_-][0-9a-f]+$/i.test(id)) return false;
        if (/^input-accordion-(?:m-)?\d+(?:-checkbox)?$/i.test(id)) return false;
        return true;
    }

    function configUrl() {
        try {
            var base = document.querySelector("base");
            return new URL("config", base && base.href ? base.href : window.location.href).toString();
        } catch (e) {
            return "./config";
        }
    }

    function loadConfig() {
        if (configPromise) return configPromise;
        configPromise = fetch(configUrl(), {
            method: "GET",
            credentials: "same-origin",
            cache: "no-store"
        }).then(function (response) {
            if (!response.ok) throw new Error("HTTP " + response.status);
            return response.json();
        }).then(function (cfg) {
            configComponents = Array.isArray(cfg && cfg.components) ? cfg.components : [];
            configById = Object.create(null);
            configComponents.forEach(function (component) {
                if (component && component.id !== undefined) configById[String(component.id)] = component;
            });
            return configComponents;
        }).catch(function (e) {
            console.warn("[SAM3 Workspaces] /config unavailable; using DOM fallback:", e);
            configComponents = [];
            configById = Object.create(null);
            return configComponents;
        });
        return configPromise;
    }

    function findGenerationPane() {
        var container = app().querySelector("#txt2img_extra_tabs");
        if (!container) return null;
        var panes = Array.prototype.slice.call(container.querySelectorAll(":scope > .tabitem"));
        for (var i = 0; i < panes.length; i++) {
            if (panes[i].querySelector("#txt2img_settings")) return panes[i];
        }
        return panes.length ? panes[0] : null;
    }

    function captureRoots() {
        var roots = [];
        var top = app().querySelector("#txt2img_toprow");
        var pane = generationPane || findGenerationPane();
        if (top) roots.push(top);
        if (pane) roots.push(pane);
        return roots;
    }

    function domType(el) {
        if (el.classList.contains("gradio-textbox")) return "textbox";
        if (el.classList.contains("gradio-number")) return "number";
        if (el.classList.contains("gradio-slider")) return "slider";
        if (el.classList.contains("gradio-checkboxgroup")) return "checkboxgroup";
        if (el.classList.contains("gradio-checkbox")) return "checkbox";
        if (el.classList.contains("gradio-radio")) return "radio";
        if (el.classList.contains("gradio-dropdown")) return "dropdown";
        return null;
    }

    function componentSelector() {
        return [
            ".gradio-textbox", ".gradio-number", ".gradio-slider",
            ".gradio-checkbox", ".gradio-radio", ".gradio-checkboxgroup",
            ".gradio-dropdown"
        ].join(",");
    }

    function resolveMeta(el, type) {
        var generated = /^component-(\d+)$/.exec(el.id || "");
        if (generated && configById[generated[1]]) return configById[generated[1]];
        if (!el.id || !configComponents.length) return null;

        var matches = configComponents.filter(function (component) {
            var props = component && component.props;
            return props && props.elem_id === el.id && (!type || component.type === type);
        }).sort(function (a, b) { return Number(a.id) - Number(b.id); });
        if (matches.length === 0) return null;
        if (matches.length === 1) return matches[0];

        var domMatches;
        try {
            domMatches = Array.prototype.slice.call(document.querySelectorAll("#" + CSS.escape(el.id)))
                .filter(function (candidate) { return domType(candidate) === type; });
        } catch (e) {
            domMatches = [];
        }
        var index = domMatches.indexOf(el);
        return index >= 0 && matches[index] ? matches[index] : null;
    }

    function componentLabel(el, meta) {
        if (meta && meta.props && typeof meta.props.label === "string" && meta.props.label) {
            return meta.props.label.trim();
        }
        var info = el.querySelector('[data-testid="block-info"]');
        if (info && info.textContent) return info.textContent.trim();
        var input = el.querySelector("input[aria-label], textarea[aria-label]");
        return input ? (input.getAttribute("aria-label") || "").trim() : "";
    }

    function nearestStableAncestor(el) {
        var roots = captureRoots();
        for (var node = el.parentElement; node; node = node.parentElement) {
            if (roots.indexOf(node) !== -1) break;
            if (stableId(node.id)) return node.id;
            if (node.classList && node.classList.contains("input-accordion")) {
                var span = node.querySelector(":scope > button > span");
                var label = span ? span.textContent.trim() : "";
                if (label) return "accordion:" + label;
            }
        }
        return "txt2img";
    }

    function xyzChoiceAxis(el) {
        if (!el.classList.contains("multiselect")) return null;
        var axes = ["x", "y", "z"];
        for (var i = 0; i < axes.length; i++) {
            var values = app().querySelector("#script_txt2img_xyz_plot_" + axes[i] + "_values");
            if (values && values.parentElement && values.parentElement.contains(el)) return axes[i];
        }
        return null;
    }

    function generatedAccordionKey(el, meta) {
        var id = el.id || "";
        var match = /^(input-accordion-(?:m-)?\d+)-checkbox$/i.exec(id);
        if (!match) return null;
        var label = componentLabel(el, meta);
        var accordion = app().querySelector("#" + CSS.escape(match[1]));
        if (!label && accordion) {
            var span = accordion.querySelector(":scope > button > span");
            label = span ? span.textContent.trim() : "";
        }
        return label ? "checkbox|input-accordion:" + encodeURIComponent(label) : null;
    }

    function shouldSkip(el, type, meta) {
        if (!type || !ALLOWED_TYPES[type]) return true;
        if (el.closest("[data-sam3-workspaces]")) return true;
        if (el.closest(".gradio-image,.gradio-file,.gradio-gallery,.gradio-audio,.gradio-video")) return true;
        if (el.classList.contains("logical_image_foreground") || el.classList.contains("logical_image_background")) return true;
        if (/(?:^|_)(?:generation_info|selected_index)(?:_|$)/i.test(el.id || "")) return true;
        if (/(?:_config_out|_spawn_out|_bridge_out|_bridge_in)$/i.test(el.id || "")) return true;
        if (meta && meta.props && meta.props.interactive === false) return true;
        return false;
    }

    function buildCatalog() {
        var roots = captureRoots();
        var seen = new Set();
        var candidates = [];
        var skipped = 0;

        roots.forEach(function (root) {
            Array.prototype.forEach.call(root.querySelectorAll(componentSelector()), function (el) {
                if (seen.has(el)) return;
                seen.add(el);
                var type = domType(el);
                var meta = resolveMeta(el, type);
                if (shouldSkip(el, type, meta)) { skipped++; return; }

                var axis = xyzChoiceAxis(el);
                var accordionKey = generatedAccordionKey(el, meta);
                var baseKey = null;
                if (axis) {
                    baseKey = "dropdown|xyz:" + axis + "-values-choice";
                } else if (accordionKey) {
                    baseKey = accordionKey;
                } else if (stableId(el.id)) {
                    baseKey = type + "|id:" + encodeURIComponent(el.id);
                }
                if (!baseKey) { skipped++; return; }

                candidates.push({
                    el: el,
                    kind: type,
                    meta: meta,
                    componentId: meta && meta.id !== undefined ? Number(meta.id) : null,
                    baseKey: baseKey,
                    key: baseKey,
                    label: componentLabel(el, meta),
                    ancestor: nearestStableAncestor(el),
                    xyzAxis: axis
                });
            });
        });

        function applyDisambiguator(getSuffix) {
            var counts = Object.create(null);
            candidates.forEach(function (item) { counts[item.key] = (counts[item.key] || 0) + 1; });
            candidates.forEach(function (item) {
                if (counts[item.key] > 1) item.key += getSuffix(item);
            });
        }

        applyDisambiguator(function (item) { return "|within:" + encodeURIComponent(item.ancestor || "txt2img"); });
        applyDisambiguator(function (item) { return "|label:" + encodeURIComponent(item.label || ""); });

        var finalCounts = Object.create(null);
        candidates.forEach(function (item) { finalCounts[item.key] = (finalCounts[item.key] || 0) + 1; });
        var map = Object.create(null);
        var adapters = [];
        var collisions = 0;
        candidates.forEach(function (item) {
            if (finalCounts[item.key] > 1) { collisions++; return; }
            map[item.key] = item;
            adapters.push(item);
        });
        capturedComponentIds = new Set(adapters
            .filter(function (item) { return item.componentId !== null && item.componentId !== undefined; })
            .map(function (item) { return Number(item.componentId); }));
        lastCatalogStats = { captured: adapters.length, skipped: skipped, collisions: collisions };
        return { map: map, adapters: adapters };
    }

    function translated(value) {
        try {
            return typeof getTranslation === "function" ? getTranslation(String(value)) : String(value);
        } catch (e) {
            return String(value);
        }
    }

    function choiceToRaw(meta, displayValue) {
        var choices = meta && meta.props && Array.isArray(meta.props.choices) ? meta.props.choices : [];
        for (var i = 0; i < choices.length; i++) {
            var pair = choices[i];
            var label = Array.isArray(pair) ? pair[0] : pair;
            var raw = Array.isArray(pair) ? pair[1] : pair;
            if (String(displayValue) === String(label) || String(displayValue) === translated(label) || String(displayValue) === String(raw)) {
                return raw;
            }
        }
        return displayValue;
    }

    function readValue(adapter) {
        var el = adapter.el;
        var kind = adapter.kind;
        if (kind === "textbox") {
            var text = el.querySelector("textarea, input:not([type]) , input[type='text']");
            return text ? text.value : undefined;
        }
        if (kind === "number" || kind === "slider") {
            var number = el.querySelector("input[type='number']") || el.querySelector("input[type='range']");
            if (!number) return undefined;
            return number.value === "" ? null : Number(number.value);
        }
        if (kind === "checkbox") {
            var checkbox = el.querySelector("input[type='checkbox']:not(.input-accordion-checkbox)");
            return checkbox ? !!checkbox.checked : undefined;
        }
        if (kind === "radio") {
            var radio = el.querySelector("input[type='radio']:checked");
            return radio ? choiceToRaw(adapter.meta, radio.value) : null;
        }
        if (kind === "checkboxgroup") {
            return Array.prototype.slice.call(el.querySelectorAll("input[type='checkbox']:checked"))
                .map(function (input) { return choiceToRaw(adapter.meta, input.value); });
        }
        if (kind === "dropdown") {
            if (el.classList.contains("multiselect")) {
                return Array.prototype.slice.call(el.querySelectorAll(".token > span:first-child"))
                    .map(function (span) { return choiceToRaw(adapter.meta, span.textContent.trim()); });
            }
            var input = el.querySelector("input[role='listbox'], input");
            if (!input) return undefined;
            if (input.getAttribute("aria-expanded") === "true") return undefined;
            return choiceToRaw(adapter.meta, input.value);
        }
        return undefined;
    }

    function valuesEqual(a, b) {
        if (Array.isArray(a) || Array.isArray(b)) {
            if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
            for (var i = 0; i < a.length; i++) if (String(a[i]) !== String(b[i])) return false;
            return true;
        }
        if (a === null || b === null) return a === b;
        if (typeof a === "number" || typeof b === "number") return Number(a) === Number(b);
        return String(a) === String(b);
    }

    function snapshotValuesEqual(a, b) {
        if (!a || !b || !isPlainObject(a.controls) || !isPlainObject(b.controls)) return false;
        var aKeys = Object.keys(a.controls);
        var bKeys = Object.keys(b.controls);
        if (aKeys.length !== bKeys.length) return false;
        for (var i = 0; i < aKeys.length; i++) {
            var key = aKeys[i];
            var left = a.controls[key];
            var right = b.controls[key];
            if (!right || left.kind !== right.kind || !valuesEqual(left.value, right.value)) return false;
            if (left.active !== right.active) return false;
        }
        return true;
    }

    function captureSnapshot(previous) {
        var catalog = buildCatalog();
        var controls = Object.create(null);
        if (previous && isPlainObject(previous.controls)) {
            Object.keys(previous.controls).forEach(function (key) {
                controls[key] = previous.controls[key];
            });
        }
        catalog.adapters.forEach(function (adapter) {
            var value = readValue(adapter);
            if (value !== undefined) {
                var record = { kind: adapter.kind, value: value };
                if (adapter.xyzAxis) {
                    record.active = adapter.el.offsetParent !== null
                        && window.getComputedStyle(adapter.el).display !== "none";
                }
                controls[adapter.key] = record;
            }
        });
        var now = new Date().toISOString();
        return {
            revision: previous ? (Number(previous.revision) || 0) + 1 : 1,
            createdAt: previous && previous.createdAt ? previous.createdAt : now,
            savedAt: now,
            writer: tabId,
            controls: controls
        };
    }

    function updateButtons() {
        if (!toolbar) return;
        var store = readStore();
        ["1", "2", "3"].forEach(function (slot) {
            var button = toolbar.querySelector('[data-workspace-slot="' + slot + '"]');
            if (!button) return;
            var snapshot = store.slots[slot];
            button.classList.toggle("active", slot === activeSlot);
            button.setAttribute("aria-pressed", slot === activeSlot ? "true" : "false");
            button.setAttribute("data-has-snapshot", snapshot ? "true" : "false");
            button.title = snapshot && snapshot.savedAt
                ? "Workspace " + slot + " · " + new Date(snapshot.savedAt).toLocaleString()
                : "Workspace " + slot + " · 비어 있음";
        });
    }

    function saveNow(reason, force) {
        // A workspace switch must flush the current slot before `activeSlot`
        // changes.  `switching` suppresses delegated autosave events, but must
        // not suppress this explicit flush.
        if (!initialized || restoring || resetting) return null;
        var before = readStore();
        var beforeClock = slotRevision(before, activeSlot);
        if (beforeClock > knownSlotRevision) {
            externalConflict = true;
            if (dirty || force) {
                setStatus("다른 탭에서 W" + activeSlot + "이 변경됨 · 새로고침 필요", "warning");
                return null;
            }
            // Clean tabs can leave the slot without overwriting the newer
            // external snapshot.  It will be restored when this slot is opened.
            return before.slots[activeSlot];
        }
        if (!force && !dirty) return before.slots[activeSlot];
        if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
        var snapshot = captureSnapshot(before.slots[activeSlot]);
        if (before.slots[activeSlot] && snapshotValuesEqual(snapshot, before.slots[activeSlot])) {
            dirty = false;
            externalConflict = false;
            knownSlotRevision = beforeClock;
            setStatus("W" + activeSlot + " 준비됨", "saved");
            return before.slots[activeSlot];
        }

        // Capturing hundreds of controls is synchronous but another browser
        // tab can still update localStorage in its own renderer.  Re-read and
        // merge into the latest store so a W1 save cannot discard a newer W2.
        var latest = readStore();
        var latestClock = slotRevision(latest, activeSlot);
        if (latestClock > beforeClock) {
            externalConflict = true;
            setStatus("다른 탭에서 W" + activeSlot + "이 변경됨 · 새로고침 필요", "warning");
            return null;
        }
        var nextClock = latestClock + 1;
        snapshot.revision = nextClock;
        latest.slotRevisions[activeSlot] = nextClock;
        latest.slots[activeSlot] = snapshot;
        latest.revision = (Number(latest.revision) || 0) + 1;
        latest.updatedAt = snapshot.savedAt;
        latest.updatedBy = tabId;
        if (!writeStore(latest)) return null;
        dirty = false;
        externalConflict = false;
        knownSlotRevision = nextClock;
        updateButtons();
        setStatus("W" + activeSlot + " 자동 저장됨", "saved");
        return snapshot;
    }

    function scheduleSave() {
        if (!initialized || bootstrapping || restoring || switching || resetting) return;
        dirty = true;
        setStatus("W" + activeSlot + " 저장 대기…", "pending");
        if (saveTimer) clearTimeout(saveTimer);
        saveTimer = setTimeout(function () { saveNow("autosave", false); }, SAVE_DELAY_MS);
    }

    function gradioContainer() {
        var root = app();
        if (root && root.classList && root.classList.contains("gradio-container")) return root;
        return (root && root.querySelector && root.querySelector(".gradio-container"))
            || document.querySelector(".gradio-container")
            || root;
    }

    function dispatchPropChange(adapter, value) {
        if (adapter.componentId === null || adapter.componentId === undefined) return false;
        var container = gradioContainer();
        if (!container || typeof CustomEvent !== "function") return false;
        try {
            container.dispatchEvent(new CustomEvent("prop_change", {
                bubbles: true,
                detail: { id: adapter.componentId, prop: "value", value: value }
            }));
            return true;
        } catch (e) {
            console.warn("[SAM3 Workspaces] prop_change failed for", adapter.key, e);
            return false;
        }
    }

    function dispatchGradioChange(adapter) {
        if (adapter.componentId === null || adapter.componentId === undefined) return false;
        var container = gradioContainer();
        if (!container || typeof CustomEvent !== "function") return false;
        try {
            // Gradio's dropdown component updates its value first and then
            // dispatches this event to run `.change()` dependencies.  Driver
            // controls must do both: `prop_change` alone only updates the UI.
            container.dispatchEvent(new CustomEvent("gradio", {
                bubbles: true,
                detail: { id: adapter.componentId, event: "change", data: undefined }
            }));
            return true;
        } catch (e) {
            console.warn("[SAM3 Workspaces] Gradio change failed for", adapter.key, e);
            return false;
        }
    }

    function nativeSetValue(input, value) {
        var proto = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        var descriptor = Object.getOwnPropertyDescriptor(proto, "value");
        if (descriptor && descriptor.set) descriptor.set.call(input, value === null ? "" : String(value));
        else input.value = value === null ? "" : String(value);
        if (typeof updateInput === "function") {
            try { updateInput(input); } catch (e) { input.dispatchEvent(new Event("input", { bubbles: true })); }
        } else {
            input.dispatchEvent(new Event("input", { bubbles: true }));
        }
        input.dispatchEvent(new Event("change", { bubbles: true }));
    }

    function optionMatches(option, value) {
        var wanted = String(value);
        var aria = option.getAttribute("aria-label") || "";
        var text = (option.textContent || "").replace(/^\s*✓\s*/, "").trim();
        return aria === wanted || text === wanted || translated(aria) === wanted;
    }

    function optionMouseDown(option) {
        if (!option) return;
        option.dispatchEvent(new MouseEvent("mousedown", {
            bubbles: true,
            cancelable: true,
            view: window,
            button: 0
        }));
        option.dispatchEvent(new MouseEvent("mouseup", {
            bubbles: true,
            cancelable: true,
            view: window,
            button: 0
        }));
    }

    async function setDropdownDom(adapter, value) {
        var el = adapter.el;
        var input = el.querySelector("input[role='listbox'], input");
        if (!input) return;
        if (el.classList.contains("multiselect")) {
            var current = readValue(adapter) || [];
            if (valuesEqual(current, value)) return;
            var removeAll = el.querySelector(".token-remove.remove-all");
            if (removeAll) removeAll.click();
            else Array.prototype.slice.call(el.querySelectorAll(".token .token-remove")).reverse()
                .forEach(function (remove) { remove.click(); });
            await delay(30);
            var wanted = Array.isArray(value) ? value : [];
            for (var i = 0; i < wanted.length; i++) {
                input.focus();
                input.click();
                await delay(25);
                var option = Array.prototype.slice.call(document.querySelectorAll('[role="option"]'))
                    .find(function (candidate) { return optionMatches(candidate, wanted[i]); });
                if (option && option.getAttribute("aria-selected") !== "true") {
                    optionMouseDown(option);
                    await delay(20);
                    var selectedNow = readValue(adapter) || [];
                    if (selectedNow.map(String).indexOf(String(wanted[i])) === -1) option.click();
                }
                await delay(25);
            }
            input.blur();
            return;
        }

        input.focus();
        input.click();
        await delay(30);
        var found = Array.prototype.slice.call(document.querySelectorAll('[role="option"]'))
            .find(function (candidate) { return optionMatches(candidate, value); });
        if (found) {
            optionMouseDown(found);
            await delay(20);
            if (!valuesEqual(readValue(adapter), value)) found.click();
            await delay(20);
        } else {
            nativeSetValue(input, value);
            input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", code: "Enter", bubbles: true }));
            input.dispatchEvent(new KeyboardEvent("keyup", { key: "Enter", code: "Enter", bubbles: true }));
        }
        input.blur();
    }

    function syncInputAccordion(adapter, value) {
        if (adapter.kind !== "checkbox") return;
        var elemId = adapter.el.id || (adapter.meta && adapter.meta.props && adapter.meta.props.elem_id) || "";
        if (!/-checkbox$/.test(elemId)) return;
        var accordionId = elemId.slice(0, -"-checkbox".length);
        var accordion = app().querySelector("#" + CSS.escape(accordionId));
        if (!accordion || !accordion.classList.contains("input-accordion")) return;
        try {
            if (typeof inputAccordionChecked === "function") inputAccordionChecked(accordionId, !!value);
            var visible = accordion.querySelector(":scope > button .input-accordion-checkbox");
            if (visible) visible.checked = !!value;
        } catch (e) {}
    }

    async function setDomFallback(adapter, value) {
        var el = adapter.el;
        if (adapter.kind === "textbox") {
            var text = el.querySelector("textarea, input:not([type]), input[type='text']");
            if (text) nativeSetValue(text, value);
        } else if (adapter.kind === "number" || adapter.kind === "slider") {
            var number = el.querySelector("input[type='number']") || el.querySelector("input[type='range']");
            if (number) nativeSetValue(number, value);
        } else if (adapter.kind === "checkbox") {
            var checkbox = el.querySelector("input[type='checkbox']:not(.input-accordion-checkbox)");
            if (checkbox && checkbox.checked !== !!value) checkbox.click();
            syncInputAccordion(adapter, value);
        } else if (adapter.kind === "radio") {
            var radios = Array.prototype.slice.call(el.querySelectorAll("input[type='radio']"));
            var radio = radios.find(function (item) { return String(item.value) === String(value); });
            if (radio && !radio.checked) radio.click();
        } else if (adapter.kind === "checkboxgroup") {
            var wanted = Array.isArray(value) ? value.map(String) : [];
            Array.prototype.forEach.call(el.querySelectorAll("input[type='checkbox']"), function (item) {
                var shouldCheck = wanted.indexOf(String(item.value)) !== -1;
                if (item.checked !== shouldCheck) item.click();
            });
        } else if (adapter.kind === "dropdown") {
            await setDropdownDom(adapter, value);
        }
    }

    async function applyOne(adapter, record, verifyNow) {
        if (!adapter || !record || adapter.kind !== record.kind) return false;
        var current = readValue(adapter);
        if (current !== undefined && valuesEqual(current, record.value)) return false;
        var dispatched = dispatchPropChange(adapter, record.value);
        syncInputAccordion(adapter, record.value);
        if (verifyNow) {
            await delay(dispatched ? 80 : 0);
            current = readValue(adapter);
            if (current === undefined || !valuesEqual(current, record.value)) {
                await setDomFallback(adapter, record.value);
            }
        } else if (!dispatched) {
            await setDomFallback(adapter, record.value);
        }
        return true;
    }

    function isDriver(adapter) {
        var id = adapter.el.id || "";
        return id === "script_list"
            || /^script_txt2img_xyz_plot_[xyz]_type$/.test(id)
            || id === "script_txt2img_xyz_plot_csv_mode";
    }

    function driverPriority(adapter) {
        var id = adapter.el.id || "";
        if (id === "script_list") return 0;
        if (/^script_txt2img_xyz_plot_[xyz]_type$/.test(id)) return 1;
        return 2;
    }

    async function stageDriverValue(adapter, record) {
        if (!adapter || !record || adapter.kind !== record.kind) return false;
        var current = readValue(adapter);
        if (current !== undefined && valuesEqual(current, record.value)) return false;

        var dispatched = dispatchPropChange(adapter, record.value);
        if (dispatched) {
            await waitFor(function () {
                var value = readValue(adapter);
                return value !== undefined && valuesEqual(value, record.value);
            }, 1200);
        }
        current = readValue(adapter);
        if (current === undefined || !valuesEqual(current, record.value)) {
            await setDomFallback(adapter, record.value);
            await waitFor(function () {
                var value = readValue(adapter);
                return value !== undefined && valuesEqual(value, record.value);
            }, 1200);
        }
        return true;
    }

    async function triggerDriverDependency(adapter, record) {
        if (!adapter || !record || adapter.kind !== record.kind) return false;
        var id = adapter.el.id || "";

        // Run the dependency even when the displayed value already matched.
        // A previous workspace may have left dependent panels/options stale.
        if (!dispatchGradioChange(adapter)) {
            // Components without a stable Gradio id can only take the normal
            // DOM path. Most drivers have ids, so this is a compatibility
            // fallback for unusual downstream builds.
            await setDomFallback(adapter, record.value);
        }
        await delay(60);

        if (id === "script_list") {
            await waitFor(function () {
                var xyz = app().querySelector("#script_txt2img_xyz_plot_x_type");
                if (!xyz) return String(record.value) !== "X/Y/Z plot";
                var visible = xyz.offsetParent !== null && window.getComputedStyle(xyz).display !== "none";
                return String(record.value) === "X/Y/Z plot" ? visible : !visible;
            }, 2500);
        }
        return true;
    }

    async function waitForXyzChoiceOptions(snapshot, catalog) {
        var axes = ["x", "y", "z"];
        var notReady = 0;
        for (var i = 0; i < axes.length; i++) {
            var key = "dropdown|xyz:" + axes[i] + "-values-choice";
            var adapter = catalog.map[key];
            var record = snapshot.controls[key];
            if (!adapter || !record || !Array.isArray(record.value)) continue;

            if (record.active === false) {
                await waitFor(function () {
                    return adapter.el.offsetParent === null
                        || window.getComputedStyle(adapter.el).display === "none";
                }, 3000);
                continue;
            }

            var visible = await waitFor(function () {
                return adapter.el.offsetParent !== null
                    && window.getComputedStyle(adapter.el).display !== "none";
            }, record.active === true ? 4000 : 400);
            // Old exports have no `active` flag. Preserve their old behavior:
            // only restore a choice dropdown if it is currently the live one.
            if (!visible) {
                if (record.active === true) notReady++;
                continue;
            }
            var input = adapter.el.querySelector("input");
            if (!input) { notReady++; continue; }
            if (record.value.length === 0) {
                // An empty selection is already restorable without loading an
                // option menu. Some valid axes (for example an empty Styles
                // list) legitimately expose zero choices.
                continue;
            }
            input.focus();
            input.click();
            var choicesReady = await waitFor(function () {
                var options = Array.prototype.slice.call(adapter.el.querySelectorAll('[role="option"]'));
                if (!options.length) {
                    options = Array.prototype.slice.call(document.querySelectorAll('[role="option"]'))
                        .filter(function (option) { return option.offsetParent !== null; });
                }
                return record.value.every(function (value) {
                    return options.some(function (option) { return optionMatches(option, value); });
                });
            }, 4000);
            input.blur();
            if (!choicesReady) notReady++;
        }
        return notReady;
    }

    function setRestoreBusy(enabled) {
        if (enabled) {
            var focused = document.activeElement;
            restoreBusyRoots = captureRoots().map(function (root) {
                var entry = {
                    root: root,
                    busy: root.getAttribute("aria-busy"),
                    hadClass: root.classList.contains("sam3-workspace-restoring")
                };
                if (focused && root.contains(focused) && typeof focused.blur === "function") focused.blur();
                root.classList.add("sam3-workspace-restoring");
                root.setAttribute("aria-busy", "true");
                return entry;
            });
            return;
        }
        restoreBusyRoots.forEach(function (entry) {
            if (!entry.root) return;
            if (!entry.hadClass) entry.root.classList.remove("sam3-workspace-restoring");
            if (entry.busy === null) entry.root.removeAttribute("aria-busy");
            else entry.root.setAttribute("aria-busy", entry.busy);
        });
        restoreBusyRoots = [];
    }

    async function restoreSnapshot(snapshot, reason) {
        if (!snapshot || !isPlainObject(snapshot.controls)) return;
        restoring++;
        if (restoring === 1) setRestoreBusy(true);
        dirty = false;
        if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
        setStatus("W" + activeSlot + " 복원 중…", "pending");
        var applied = 0;
        var missing = 0;
        try {
            await loadConfig();
            var catalog = buildCatalog();
            var driverAdapters = catalog.adapters.filter(isDriver).sort(function (a, b) {
                return driverPriority(a) - driverPriority(b);
            });
            var driverKeys = driverAdapters.map(function (adapter) { return adapter.key; });

            // Phase 1: commit every driver value before starting any Gradio
            // dependency. XYZ axis callbacks also read CSV mode and sibling
            // axes, so dispatching while values are still changing can let an
            // older async response win the race and rebuild stale controls.
            for (var i = 0; i < driverAdapters.length; i++) {
                var driver = driverAdapters[i];
                var driverRecord = snapshot.controls[driver.key];
                if (driverRecord && await stageDriverValue(driver, driverRecord)) applied++;
            }
            await delay(60);

            // Phase 2: all callbacks now observe the same final driver state.
            // Trigger even for values that already matched so stale dependent
            // panels from the previous workspace are rebuilt deterministically.
            for (var d = 0; d < driverAdapters.length; d++) {
                var dependencyDriver = driverAdapters[d];
                var dependencyRecord = snapshot.controls[dependencyDriver.key];
                if (dependencyRecord && await triggerDriverDependency(dependencyDriver, dependencyRecord)) applied++;
            }

            if (driverKeys.length) await delay(120);
            catalog = buildCatalog();
            var xyzNotReady = await waitForXyzChoiceOptions(snapshot, catalog);
            catalog = buildCatalog();
            var keys = Object.keys(snapshot.controls);
            for (var j = 0; j < keys.length; j++) {
                var key = keys[j];
                if (driverKeys.indexOf(key) !== -1) continue;
                var adapter = catalog.map[key];
                if (!adapter) { missing++; continue; }
                var record = snapshot.controls[key];
                if (adapter.xyzAxis && record.active === false) continue;
                if (await applyOne(adapter, record, false)) applied++;
            }

            await delay(160);
            catalog = buildCatalog();
            var fallbackCount = 0;
            for (var k = 0; k < keys.length; k++) {
                var verifyAdapter = catalog.map[keys[k]];
                var verifyRecord = snapshot.controls[keys[k]];
                if (!verifyAdapter || !verifyRecord || verifyAdapter.kind !== verifyRecord.kind) continue;
                if (verifyAdapter.xyzAxis && verifyRecord.active === false) continue;
                var actual = readValue(verifyAdapter);
                if (actual !== undefined && !valuesEqual(actual, verifyRecord.value)) {
                    await setDomFallback(verifyAdapter, verifyRecord.value);
                    fallbackCount++;
                }
            }
            knownSlotRevision = Number(snapshot.revision) || 0;
            setStatus(
                "W" + activeSlot + " 복원됨 · " + applied + "개 적용",
                missing || xyzNotReady ? "warning" : "saved"
            );
            if (xyzNotReady) console.warn("[SAM3 Workspaces] XYZ choice controls not ready:", xyzNotReady);
            if (fallbackCount) console.debug("[SAM3 Workspaces] DOM fallback count:", fallbackCount);
        } catch (e) {
            console.error("[SAM3 Workspaces] restore failed:", e);
            setStatus("W" + activeSlot + " 복원 오류", "error");
        } finally {
            restoring--;
            if (restoring === 0) setRestoreBusy(false);
            dirty = false;
            updateButtons();
        }
    }

    function generationRunning() {
        var interrupt = app().querySelector("#txt2img_interrupt");
        if (!interrupt) return false;
        var style = window.getComputedStyle(interrupt);
        return style.display !== "none" && style.visibility !== "hidden" && interrupt.offsetParent !== null;
    }

    async function switchWorkspace(target) {
        target = String(target);
        if (target === activeSlot || switching || restoring) return;
        if (target !== "1" && target !== "2" && target !== "3") return;
        if (generationRunning()) {
            setStatus("생성 중에는 전환할 수 없습니다", "warning");
            return;
        }
        switching = true;
        try {
            var hadDirtyChanges = dirty;
            var sourceSnapshot = saveNow("switch", false);
            if (hadDirtyChanges && !sourceSnapshot) {
                setStatus("W" + activeSlot + " 저장 실패 · 전환 취소", "error");
                return;
            }
            var store = readStore();
            if (!store.slots[target]) {
                var clone = sourceSnapshot ? cloneJson(sourceSnapshot) : captureSnapshot(null);
                clone.createdAt = new Date().toISOString();
                clone.savedAt = clone.createdAt;
                clone.writer = tabId;
                // Merge into the latest store and honor a target created by a
                // different tab while the source was being flushed.
                store = readStore();
                if (!store.slots[target]) {
                    var targetClock = slotRevision(store, target) + 1;
                    clone.revision = targetClock;
                    store.slotRevisions[target] = targetClock;
                    store.slots[target] = clone;
                    store.revision = (Number(store.revision) || 0) + 1;
                    store.updatedAt = clone.savedAt;
                    store.updatedBy = tabId;
                    if (!writeStore(store)) {
                        setStatus("Workspace " + target + " 생성 실패 · 전환 취소", "error");
                        return;
                    }
                }
            }
            activeSlot = target;
            safeSessionSet(ACTIVE_KEY, activeSlot);
            store = readStore();
            knownSlotRevision = slotRevision(store, activeSlot);
            externalConflict = false;
            updateButtons();
            switching = false;
            await restoreSnapshot(store.slots[activeSlot], "switch");
        } finally {
            switching = false;
        }
    }

    function downloadExport() {
        var hadDirtyChanges = dirty;
        var saved = saveNow("export", false);
        if (hadDirtyChanges && !saved) {
            setStatus("현재 작업공간 저장 실패 · 내보내기 취소", "error");
            return;
        }
        var store = readStore();
        var payload = {
            schema: SCHEMA,
            format: "sam-extra-workspaces",
            extensionVersion: VERSION,
            exportedAt: new Date().toISOString(),
            revision: store.revision,
            updatedAt: store.updatedAt,
            updatedBy: store.updatedBy,
            slotRevisions: store.slotRevisions,
            slots: store.slots
        };
        try {
            var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
            var url = URL.createObjectURL(blob);
            var anchor = document.createElement("a");
            anchor.href = url;
            anchor.download = "sam-extra-workspaces-" + new Date().toISOString().replace(/[:.]/g, "-") + ".json";
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
            setStatus("작업공간을 내보냈습니다", "saved");
        } catch (e) {
            setStatus("내보내기에 실패했습니다", "error");
        }
    }

    function importFile(file) {
        if (!file) return;
        if (file.size > MAX_STORAGE_BYTES) {
            setStatus("가져올 파일이 너무 큽니다", "error");
            return;
        }
        var reader = new FileReader();
        reader.onload = async function () {
            try {
                var parsed = JSON.parse(String(reader.result || ""));
                var imported = normalizeStore(parsed);
                var hasAny = imported.slots["1"] || imported.slots["2"] || imported.slots["3"];
                if (!hasAny) throw new Error("no workspace snapshots");
                if (!window.confirm("현재 3개 작업공간을 가져온 파일로 교체할까요?")) return;
                switching = true;
                dirty = false;
                if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
                var current = readStore();
                var importedAt = new Date().toISOString();
                ["1", "2", "3"].forEach(function (slot) {
                    var clock = Math.max(
                        slotRevision(current, slot),
                        slotRevision(imported, slot),
                        imported.slots[slot] ? Number(imported.slots[slot].revision) || 0 : 0
                    ) + 1;
                    imported.slotRevisions[slot] = clock;
                    if (imported.slots[slot]) {
                        imported.slots[slot].revision = clock;
                        imported.slots[slot].writer = tabId;
                        imported.slots[slot].savedAt = importedAt;
                    }
                });
                imported.revision = Math.max(Number(current.revision) || 0, Number(imported.revision) || 0) + 1;
                imported.updatedAt = importedAt;
                imported.updatedBy = tabId;
                if (!writeStore(imported)) {
                    switching = false;
                    return;
                }
                var restoreSlot = imported.slots[activeSlot]
                    ? activeSlot
                    : (["1", "2", "3"].find(function (slot) { return !!imported.slots[slot]; }) || "1");
                activeSlot = restoreSlot;
                safeSessionSet(ACTIVE_KEY, activeSlot);
                knownSlotRevision = slotRevision(imported, activeSlot);
                externalConflict = false;
                updateButtons();
                switching = false;
                await restoreSnapshot(imported.slots[activeSlot], "import");
                setStatus("작업공간을 가져왔습니다", "saved");
            } catch (e) {
                switching = false;
                console.warn("[SAM3 Workspaces] import failed:", e);
                setStatus("올바른 작업공간 파일이 아닙니다", "error");
            }
        };
        reader.readAsText(file);
    }

    function resetCurrentWorkspace() {
        if (!window.confirm("Workspace " + activeSlot + " 저장본을 지우고 페이지를 새로 고칠까요?")) return;
        resetting = true;
        dirty = false;
        if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
        var store = readStore();
        var nextClock = slotRevision(store, activeSlot) + 1;
        store.slots[activeSlot] = null;
        store.slotRevisions[activeSlot] = nextClock;
        store.revision = (Number(store.revision) || 0) + 1;
        store.updatedAt = new Date().toISOString();
        store.updatedBy = tabId;
        if (writeStore(store)) {
            knownSlotRevision = nextClock;
            window.location.reload();
        } else {
            resetting = false;
        }
    }

    function createToolbar() {
        var bar = document.createElement("div");
        bar.id = "sam3_workspace_bar";
        bar.className = "sam3-workspace-bar";
        bar.setAttribute("data-sam3-workspaces", "1");
        bar.innerHTML = [
            '<span class="sam3-workspace-title">Workspaces</span>',
            '<div class="sam3-workspace-slots" role="group" aria-label="txt2img workspaces">',
            '  <button type="button" data-workspace-slot="1" aria-pressed="false"><span>1</span></button>',
            '  <button type="button" data-workspace-slot="2" aria-pressed="false"><span>2</span></button>',
            '  <button type="button" data-workspace-slot="3" aria-pressed="false"><span>3</span></button>',
            '</div>',
            '<span class="sam3-workspace-status" data-workspace-status aria-live="polite"></span>',
            '<details class="sam3-workspace-menu">',
            '  <summary aria-label="Workspace 메뉴" title="Workspace 메뉴">⋯</summary>',
            '  <div class="sam3-workspace-menu-panel">',
            '    <button type="button" data-workspace-export>내보내기</button>',
            '    <label>가져오기<input type="file" accept="application/json,.json" data-workspace-import></label>',
            '    <button type="button" class="danger" data-workspace-reset>이 슬롯 새로 시작</button>',
            '  </div>',
            '</details>'
        ].join("");

        Array.prototype.forEach.call(bar.querySelectorAll("[data-workspace-slot]"), function (button) {
            button.addEventListener("click", function () { switchWorkspace(button.getAttribute("data-workspace-slot")); });
        });
        bar.querySelector("[data-workspace-export]").addEventListener("click", function () {
            downloadExport();
            bar.querySelector("details").open = false;
        });
        bar.querySelector("[data-workspace-import]").addEventListener("change", function (event) {
            importFile(event.target.files && event.target.files[0]);
            event.target.value = "";
            bar.querySelector("details").open = false;
        });
        bar.querySelector("[data-workspace-reset]").addEventListener("click", resetCurrentWorkspace);
        return bar;
    }

    function mountToolbar() {
        var pane = findGenerationPane();
        if (!pane) return false;
        generationPane = pane;
        var existing = app().querySelector("#sam3_workspace_bar");
        if (existing && existing.isConnected) {
            toolbar = existing;
            updateButtons();
            return true;
        }
        toolbar = createToolbar();
        // #txt2img_settings is one column of Forge's resize row.  Adding the
        // toolbar *inside that row* creates a third flex column and crushes the
        // controls.  Insert it immediately above the whole row: full width,
        // and still visible when Forge wraps settings in a closed accordion.
        var settings = pane.querySelector("#txt2img_settings");
        var layoutRow = settings && settings.closest(".resize-handle-row");
        if (layoutRow && layoutRow.parentNode) layoutRow.parentNode.insertBefore(toolbar, layoutRow);
        else if (settings) settings.insertBefore(toolbar, settings.firstChild);
        else pane.insertBefore(toolbar, pane.firstChild);
        updateButtons();
        return true;
    }

    function eventInsideCaptureRoots(target) {
        if (!target || !target.closest || target.closest("[data-sam3-workspaces]")) return false;
        var roots = captureRoots();
        for (var i = 0; i < roots.length; i++) if (roots[i].contains(target)) return true;
        if (target.getAttribute && target.getAttribute("role") === "option") {
            var focused = document.activeElement;
            for (var j = 0; j < roots.length; j++) if (focused && roots[j].contains(focused)) return true;
        }
        return false;
    }

    function attachAutosave() {
        if (window.__sam3WorkspaceAutosaveAttached) return;
        window.__sam3WorkspaceAutosaveAttached = true;
        document.addEventListener("input", function (event) {
            if (eventInsideCaptureRoots(event.target)) scheduleSave();
        }, true);
        document.addEventListener("change", function (event) {
            if (eventInsideCaptureRoots(event.target)) scheduleSave();
        }, true);
        document.addEventListener("prop_change", function (event) {
            var detail = event && event.detail;
            if (!detail || detail.prop !== "value") return;
            if (capturedComponentIds.has(Number(detail.id))) scheduleSave();
        }, true);
        document.addEventListener("click", function (event) {
            if (!initialized || restoring || switching) return;
            var target = event.target && event.target.closest
                ? event.target.closest('[role="option"],.token-remove,.input-accordion-checkbox')
                : null;
            if (target && eventInsideCaptureRoots(target)) scheduleSave();
        }, true);
        window.addEventListener("pagehide", function () {
            if (dirty && !bootstrapping && !restoring && !switching && !resetting) saveNow("pagehide", true);
        });
        window.addEventListener("storage", function (event) {
            if (event.key !== STORAGE_KEY) return;
            updateButtons();
            var store = readStore();
            if (slotRevision(store, activeSlot) > knownSlotRevision) {
                externalConflict = true;
                setStatus("다른 탭에서 W" + activeSlot + "이 변경됨", "warning");
            }
        });
    }

    async function initialize() {
        if (initialized) return;
        if (!mountToolbar()) return;
        if (!app().querySelector("#txt2img_prompt textarea") || !app().querySelector("#txt2img_steps input")) return;
        initialized = true;
        bootstrapping = true;
        setStatus("작업공간 준비 중…", "pending");
        try {
            await loadConfig();
            await delay(250);
            var store = readStore();
            var snapshot = store.slots[activeSlot];
            knownSlotRevision = slotRevision(store, activeSlot);
            externalConflict = false;
            if (snapshot) {
                await restoreSnapshot(snapshot, "startup");
            } else {
                dirty = true;
                var created = saveNow("initial", true);
                if (created) setStatus("W" + activeSlot + "을 현재 설정으로 만들었습니다", "saved");
            }
        } finally {
            bootstrapping = false;
            attachAutosave();
            updateButtons();
        }
    }

    function ensureMounted() {
        var pane = findGenerationPane();
        if (!pane) return;
        var replaced = generationPane && pane !== generationPane;
        if (!toolbar || !toolbar.isConnected || replaced) mountToolbar();
        if (!initialized) {
            initialize();
        } else if (replaced && !dirty && !restoring && !switching) {
            var remountSlot = activeSlot;
            setTimeout(function () {
                if (activeSlot !== remountSlot || dirty || bootstrapping || restoring || switching || resetting) return;
                var snapshot = readStore().slots[remountSlot];
                if (snapshot) restoreSnapshot(snapshot, "remount");
            }, 300);
        }
    }

    function diagnostics() {
        var store = readStore();
        return {
            version: VERSION,
            activeSlot: activeSlot,
            initialized: initialized,
            bootstrapping: bootstrapping,
            restoring: !!restoring,
            switching: switching,
            resetting: resetting,
            externalConflict: externalConflict,
            dirty: dirty,
            knownSlotRevision: knownSlotRevision,
            catalog: lastCatalogStats,
            slots: ["1", "2", "3"].map(function (slot) {
                var snapshot = store.slots[slot];
                return {
                    slot: Number(slot),
                    exists: !!snapshot,
                    revision: snapshot ? snapshot.revision : 0,
                    controlCount: snapshot ? Object.keys(snapshot.controls).length : 0,
                    savedAt: snapshot ? snapshot.savedAt : null
                };
            })
        };
    }

    window.__sam3WorkspaceManager = {
        version: VERSION,
        diagnostics: diagnostics
    };

    function start() {
        ensureMounted();
        var observer = new MutationObserver(function () { ensureMounted(); });
        try { observer.observe(document.documentElement, { childList: true, subtree: true }); } catch (e) {}
        var interval = setInterval(ensureMounted, 800);
        setTimeout(function () {
            try { observer.disconnect(); } catch (e) {}
            clearInterval(interval);
        }, 300000);
        if (typeof onAfterUiUpdate === "function") onAfterUiUpdate(ensureMounted);
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () { setTimeout(start, 400); });
    } else {
        document.addEventListener("DOMContentLoaded", function () { setTimeout(start, 1200); });
    }
})();
