/* SAM3 Notebook — single-page prompt / XYZ / generation-parameter presets. */
(function () {
    "use strict";

    if (window.__sam3NotebookLoaded) return;
    window.__sam3NotebookLoaded = true;

    var API_PATH = "/sam3-notebook";
    var DRAFT_KEY = "sam-extra.notebook.draft.v1";
    var UI_KEY = "sam-extra.notebook.ui.v1";
    var LEGACY_WORKSPACE_KEY = "sam-extra.workspace-manager.v1";
    var SAVE_DELAY_MS = 650;
    var MAX_PRESETS = 200;
    var MAX_ENTRIES_PER_PRESET = 64;
    var MAX_NAME_LENGTH = 80;
    var MAX_VALUE_LENGTH = 100000;
    var FAST_DROPDOWN_LIMIT_OPTION = "sam3_fast_dropdown_visible_choices";
    var DEFAULT_FAST_DROPDOWN_VISIBLE_CHOICES = 60;
    var MAX_RENDERED_FAST_CHOICES = 200;

    var TARGETS = [
        { key: "prompt", label: "프롬프트", kind: "text" },
        { key: "negative_prompt", label: "네거티브 프롬프트", kind: "text" },
        { key: "lora_tokens", label: "LoRA 토큰", kind: "text" },
        { key: "xyz_x", label: "XYZ · X Plot", kind: "xyz", axis: "x" },
        { key: "xyz_y", label: "XYZ · Y Plot", kind: "xyz", axis: "y" },
        { key: "xyz_z", label: "XYZ · Z Plot", kind: "xyz", axis: "z" },
        { key: "sampling_method", label: "Sampling Method", kind: "choice", elemId: "txt2img_sampling" },
        { key: "schedule_type", label: "Schedule Type", kind: "choice", elemId: "txt2img_scheduler" },
        { key: "sampling_steps", label: "Sampling Steps", kind: "number", elemId: "txt2img_steps" },
        { key: "width", label: "너비", kind: "number", elemId: "txt2img_width" },
        { key: "height", label: "높이", kind: "number", elemId: "txt2img_height" },
        { key: "batch_count", label: "Batch Count", kind: "number", elemId: "txt2img_batch_count" },
        { key: "batch_size", label: "Batch Size", kind: "number", elemId: "txt2img_batch_size" },
        { key: "shift", label: "Shift", kind: "number", elemId: "txt2img_distilled_cfg_scale" },
        { key: "cfg_scale", label: "CFG Scale", kind: "number", elemId: "txt2img_cfg_scale" },
        { key: "rescale_cfg", label: "Rescale CFG", kind: "number", elemId: "txt2img_rescale_cfg_scale" },
        { key: "seed", label: "Seed", kind: "number", elemId: "txt2img_seed" }
    ];
    var TARGET_BY_KEY = Object.create(null);
    TARGETS.forEach(function (target) { TARGET_BY_KEY[target.key] = target; });

    var model = { version: 1, revision: 0, updated_at: "", presets: [] };
    var panel = null;
    var presetList = null;
    var statusElement = null;
    var searchInput = null;
    var selectedPresetId = null;
    var saveTimer = null;
    var saveChain = Promise.resolve();
    var changeVersion = 0;
    var undoState = null;
    var gradioConfigSnapshot = null;
    var gradioConfigPromise = null;
    var delegatedEventRoots = typeof WeakSet === "function" ? new WeakSet() : [];
    var boundPanels = typeof WeakSet === "function" ? new WeakSet() : [];
    var fastDropdownVisibilityObserver = null;
    var fastDropdownObserved = typeof WeakSet === "function" ? new WeakSet() : [];
    var fastDropdownWarmQueue = Promise.resolve();
    var fastDropdownWarming = Object.create(null);
    var fastDropdownGlobalEventsInstalled = false;
    var fastDropdownModelRefreshPending = false;
    var fastDropdownScanTimer = null;

    function app() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function delay(ms) {
        return new Promise(function (resolve) { setTimeout(resolve, ms); });
    }

    function waitFor(predicate, timeout) {
        return new Promise(function (resolve) {
            var deadline = Date.now() + (timeout || 15000);
            var timer = setInterval(function () {
                var value = null;
                try { value = predicate(); } catch (error) {}
                if (value || Date.now() >= deadline) {
                    clearInterval(timer);
                    resolve(value);
                }
            }, 80);
        });
    }

    function makeId(prefix) {
        if (window.crypto && typeof window.crypto.randomUUID === "function") {
            return prefix + "-" + window.crypto.randomUUID();
        }
        return prefix + "-" + Date.now().toString(36) + "-"
            + Math.random().toString(36).slice(2);
    }

    function clone(value) {
        return JSON.parse(JSON.stringify(value));
    }

    function normalize(raw) {
        var result = { version: 1, revision: 0, updated_at: "", presets: [] };
        if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
            throw new Error("Notebook JSON의 최상위 값은 객체여야 합니다");
        }
        if (raw.version !== undefined && Number(raw.version) !== 1) {
            throw new Error("지원하지 않는 Notebook 형식입니다: version " + raw.version);
        }
        result.revision = Math.max(0, Number(raw.revision) || 0);
        result.updated_at = typeof raw.updated_at === "string" ? raw.updated_at : "";
        if (!Array.isArray(raw.presets)) {
            throw new Error("Notebook presets는 배열이어야 합니다");
        }
        if (raw.presets.length > MAX_PRESETS) {
            throw new Error("Notebook 프리셋은 최대 " + MAX_PRESETS + "개입니다");
        }
        result.presets = raw.presets.map(function (preset, presetIndex) {
            if (!preset || typeof preset !== "object" || Array.isArray(preset)) {
                throw new Error("Notebook 프리셋은 객체여야 합니다");
            }
            if (!Array.isArray(preset.entries)) {
                throw new Error("Notebook 프리셋의 entries는 배열이어야 합니다");
            }
            var entries = preset.entries;
            if (entries.length > MAX_ENTRIES_PER_PRESET) {
                throw new Error(
                    "프리셋 하나에는 항목을 최대 "
                    + MAX_ENTRIES_PER_PRESET + "개 저장할 수 있습니다"
                );
            }
            var name = String(
                preset && preset.name || ("preset" + (presetIndex + 1))
            );
            if (name.length > MAX_NAME_LENGTH) {
                throw new Error(
                    "프리셋 이름은 최대 " + MAX_NAME_LENGTH + "자입니다"
                );
            }
            return {
                id: String(preset && preset.id || makeId("preset")),
                name: name,
                entries: entries.map(function (entry) {
                    if (!entry || !TARGET_BY_KEY[entry.target]) {
                        throw new Error(
                            "지원하지 않는 Notebook 항목이 있습니다: "
                            + String(entry && entry.target || "<empty>")
                        );
                    }
                    var value = String(
                        entry.value === undefined || entry.value === null
                            ? "" : entry.value
                    );
                    if (value.length > MAX_VALUE_LENGTH) {
                        throw new Error(
                            "Notebook 항목 값은 최대 "
                            + MAX_VALUE_LENGTH + "자입니다"
                        );
                    }
                    var clean = {
                        id: String(entry.id || makeId("entry")),
                        target: String(entry.target),
                        value: value
                    };
                    if (clean.target.indexOf("xyz_") === 0) {
                        clean.axis_type = String(entry.axis_type || "");
                    }
                    return clean;
                })
            };
        });
        return result;
    }

    function readUiState() {
        try {
            var parsed = JSON.parse(window.localStorage.getItem(UI_KEY) || "{}");
            return parsed && typeof parsed === "object" ? parsed : {};
        } catch (error) {
            return {};
        }
    }

    function writeUiState() {
        try {
            window.localStorage.setItem(UI_KEY, JSON.stringify({
                open: !!(panel && panel.open),
                selectedPresetId: selectedPresetId,
                search: searchInput ? searchInput.value : ""
            }));
        } catch (error) {}
    }

    function writeDraft() {
        try {
            window.localStorage.setItem(DRAFT_KEY, JSON.stringify({
                baseRevision: model.revision,
                version: 1,
                presets: model.presets,
                changedAt: new Date().toISOString()
            }));
        } catch (error) {}
    }

    function clearDraft(versionAtSave) {
        if (versionAtSave !== changeVersion) return;
        try { window.localStorage.removeItem(DRAFT_KEY); } catch (error) {}
    }

    function recoverDraftIfSafe(serverModel) {
        try {
            var draft = JSON.parse(window.localStorage.getItem(DRAFT_KEY) || "null");
            if (!draft || Number(draft.baseRevision) !== Number(serverModel.revision)
                    || !Array.isArray(draft.presets)) return false;
            serverModel.presets = normalize({ presets: draft.presets }).presets;
            return true;
        } catch (error) {
            return false;
        }
    }

    function setStatus(message, tone) {
        if (!statusElement) return;
        statusElement.textContent = message || "";
        statusElement.setAttribute("data-tone", tone || "normal");
    }

    async function responseError(response) {
        var detail = "";
        try {
            var payload = await response.json();
            detail = payload && payload.detail ? String(payload.detail) : "";
        } catch (error) {}
        return new Error(detail || ("HTTP " + response.status));
    }

    function scheduleSave() {
        changeVersion++;
        writeDraft();
        setStatus("저장 대기…", "pending");
        if (saveTimer) clearTimeout(saveTimer);
        saveTimer = setTimeout(queueSave, SAVE_DELAY_MS);
    }

    function queueSave() {
        if (saveTimer) {
            clearTimeout(saveTimer);
            saveTimer = null;
        }
        var versionAtQueue = changeVersion;
        saveChain = saveChain.then(async function () {
            var payload = {
                version: 1,
                revision: model.revision,
                updated_at: model.updated_at,
                presets: clone(model.presets)
            };
            setStatus("저장 중…", "pending");
            var response = await window.fetch(API_PATH, {
                method: "PUT",
                credentials: "same-origin",
                cache: "no-store",
                headers: {
                    "Content-Type": "application/json",
                    "X-SAM3-Notebook": "1"
                },
                body: JSON.stringify(payload)
            });
            if (response.status === 409) {
                throw new Error("다른 페이지에서 Notebook이 변경되었습니다. 새로고침해 주세요.");
            }
            if (!response.ok) throw await responseError(response);
            var saved = await response.json();
            model.revision = Number(saved.revision) || model.revision;
            model.updated_at = String(saved.updated_at || "");
            clearDraft(versionAtQueue);
            if (versionAtQueue === changeVersion) {
                var time = new Date();
                setStatus(
                    "저장됨 " + time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                    "saved"
                );
            } else {
                // The server revision advanced while a later local edit was
                // pending. Rebase the crash-recovery draft onto that revision
                // so a reload can still recover the unsaved edit.
                writeDraft();
                setStatus("추가 변경 저장 대기…", "pending");
                if (!saveTimer) saveTimer = setTimeout(queueSave, 80);
            }
        }).catch(function (error) {
            console.error("[SAM3 Notebook] save failed:", error);
            setStatus(String(error && error.message || error), "error");
        });
        return saveChain;
    }

    function configComponents() {
        var config = window.gradio_config
            || (typeof gradio_config !== "undefined" ? gradio_config : null)
            || gradioConfigSnapshot;
        return config && Array.isArray(config.components) ? config.components : [];
    }

    async function ensureGradioConfig() {
        if (configComponents().length || gradioConfigPromise) {
            if (gradioConfigPromise) await gradioConfigPromise;
            return;
        }
        gradioConfigPromise = window.fetch("/config", {
            method: "GET",
            credentials: "same-origin",
            cache: "no-store"
        }).then(function (response) {
            if (!response.ok) throw new Error("HTTP " + response.status);
            return response.json();
        }).then(function (config) {
            if (config && Array.isArray(config.components)) {
                gradioConfigSnapshot = config;
            }
        }).catch(function (error) {
            console.warn("[SAM3 Notebook] Gradio config unavailable:", error);
        });
        await gradioConfigPromise;
    }

    function metaForElemId(elemId) {
        var components = configComponents();
        for (var i = 0; i < components.length; i++) {
            var props = components[i] && components[i].props;
            if (props && props.elem_id === elemId) return components[i];
        }
        return null;
    }

    function choiceLabels(rawChoices) {
        var raw = Array.isArray(rawChoices) ? rawChoices : [];
        return raw.map(function (choice) {
            return translatedChoice(Array.isArray(choice) ? choice[0] : choice);
        });
    }

    function displayChoices(elemId) {
        var meta = metaForElemId(elemId);
        return choiceLabels(meta && meta.props ? meta.props.choices : null);
    }

    function choiceRawValue(meta, displayValue) {
        var choices = meta && meta.props && Array.isArray(meta.props.choices)
            ? meta.props.choices : [];
        for (var i = 0; i < choices.length; i++) {
            var choice = choices[i];
            var label = Array.isArray(choice) ? choice[0] : choice;
            var raw = Array.isArray(choice) ? choice[1] : choice;
            if (String(label) === String(displayValue)
                    || String(raw) === String(displayValue)
                    || translatedChoice(label) === String(displayValue)) {
                return raw;
            }
        }
        return displayValue;
    }

    function choiceDisplayValue(meta, rawValue) {
        var choices = meta && meta.props && Array.isArray(meta.props.choices)
            ? meta.props.choices : [];
        for (var i = 0; i < choices.length; i++) {
            var choice = choices[i];
            var label = Array.isArray(choice) ? choice[0] : choice;
            var raw = Array.isArray(choice) ? choice[1] : choice;
            if (String(raw) === String(rawValue) || String(label) === String(rawValue)) {
                return translatedChoice(label);
            }
        }
        return String(rawValue);
    }

    function gradioContainer() {
        var root = app();
        if (root && root.classList && root.classList.contains("gradio-container")) return root;
        return root && root.querySelector
            ? (root.querySelector(".gradio-container") || root) : document;
    }

    function dispatchPropChange(componentId, value) {
        if (componentId === null || componentId === undefined) return false;
        var container = gradioContainer();
        if (!container || typeof CustomEvent !== "function") return false;
        container.dispatchEvent(new CustomEvent("prop_change", {
            bubbles: true,
            detail: { id: componentId, prop: "value", value: value }
        }));
        return true;
    }

    function dispatchGradioEvent(componentId, eventName) {
        if (componentId === null || componentId === undefined) return false;
        var container = gradioContainer();
        if (!container || typeof CustomEvent !== "function") return false;
        container.dispatchEvent(new CustomEvent("gradio", {
            bubbles: true,
            detail: { id: componentId, event: eventName, data: undefined }
        }));
        return true;
    }

    function dispatchGradioChange(componentId) {
        var input = dispatchGradioEvent(componentId, "input");
        var change = dispatchGradioEvent(componentId, "change");
        return input || change;
    }

    function nativeSetValue(input, value) {
        if (!input) return;
        var proto = input instanceof HTMLTextAreaElement
            ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        var descriptor = Object.getOwnPropertyDescriptor(proto, "value");
        var text = value === null || value === undefined ? "" : String(value);
        if (descriptor && descriptor.set) descriptor.set.call(input, text);
        else input.value = text;
        if (typeof updateInput === "function") {
            try { updateInput(input); } catch (error) {
                input.dispatchEvent(new Event("input", { bubbles: true }));
            }
        } else {
            input.dispatchEvent(new Event("input", { bubbles: true }));
        }
        input.dispatchEvent(new Event("change", { bubbles: true }));
    }

    function wrapperFor(elemId) {
        try { return app().querySelector("#" + CSS.escape(elemId)); }
        catch (error) { return null; }
    }

    function readWrapperDisplay(elemId) {
        var wrapper = wrapperFor(elemId);
        if (!wrapper) return "";
        if (wrapper.classList.contains("gradio-checkbox")) {
            var checkbox = wrapper.querySelector("input[type='checkbox']");
            return checkbox ? !!checkbox.checked : false;
        }
        // A multiselect keeps its selection in tokens, not in the search input
        // (which stays empty). Reading the input here would make every
        // multiselect look unchanged - see the triggerChange guard in
        // setWrapperValue, which would then double-fire the backend.
        if (wrapper.classList.contains("multiselect")) {
            return Array.prototype.slice.call(
                wrapper.querySelectorAll(".token > span:first-child")
            ).map(function (span) {
                return span.textContent.trim();
            }).filter(Boolean).join(String.fromCharCode(31));
        }
        var input = wrapper.querySelector(
            "textarea, input[role='listbox'], input[type='number'], "
            + "input[type='range'], input[type='text'], input:not([type])"
        );
        return input ? input.value : "";
    }

    async function setWrapperValue(elemId, displayValue, triggerChange) {
        var wrapper = wrapperFor(elemId);
        if (!wrapper) throw new Error(elemId + " 입력을 찾지 못했습니다");
        var meta = metaForElemId(elemId);
        var isMulti = wrapper.classList.contains("multiselect");
        var displayValues = Array.isArray(displayValue)
            ? displayValue.map(String) : [String(displayValue)];
        var raw = isMulti
            ? displayValues.map(function (item) {
                return choiceRawValue(meta, item);
            })
            : choiceRawValue(meta, displayValue);
        if (wrapper.classList.contains("gradio-slider")
                || wrapper.classList.contains("gradio-number")) {
            var numeric = Number(displayValue);
            if (Number.isFinite(numeric)) raw = numeric;
        } else if (wrapper.classList.contains("gradio-checkbox")) {
            raw = !!displayValue;
        }

        var componentId = meta && meta.id !== undefined ? Number(meta.id) : null;
        // Captured before the update so we can tell whether Gradio's own
        // change/input dispatch will fire. See the triggerChange branch below.
        var previousDisplay = readWrapperDisplay(elemId);
        var dispatched = dispatchPropChange(componentId, raw);
        await delay(dispatched ? 60 : 0);

        if (wrapper.classList.contains("gradio-checkbox")) {
            var checkbox = wrapper.querySelector("input[type='checkbox']");
            if (checkbox && checkbox.checked !== !!raw) checkbox.click();
        } else if (isMulti) {
            if (!dispatched) {
                throw new Error(elemId + " 다중 선택 값을 전달하지 못했습니다");
            }
        } else {
            var input = wrapper.querySelector(
                "textarea, input[role='listbox'], input[type='number'], "
                + "input[type='range'], input[type='text'], input:not([type])"
            );
            var current = input ? String(input.value) : "";
            if (!dispatched || (String(displayValue) !== current
                    && String(raw) !== current)) {
                nativeSetValue(input, displayValue);
            }
        }
        if (triggerChange) {
            // Do NOT dispatch change/input unconditionally here. When the value
            // actually changed, Gradio already did it for us: setting the value
            // (through prop_change or a native input event) runs the component's
            // handle_change — compiled as `gt(l,e,t){l("change",e),t||l("input")}`
            // in the frontend bundle — and value_is_output is only true for
            // server outputs, so both events fire and Blocks' "gradio" listener
            // triggers every bound backend handler. Firing again would run each
            // handler twice; picking a checkpoint would load the model twice.
            // Only force the events when nothing changed, so an explicit
            // triggerChange still re-runs the handler for an already-set value.
            var settledDisplay = readWrapperDisplay(elemId);
            if (String(settledDisplay) === String(previousDisplay)) {
                dispatchGradioChange(componentId);
            }
            await delay(120);
        }
        if (typeof wrapper.__sam3FastDropdownSync === "function") {
            wrapper.__sam3FastDropdownSync(
                isMulti ? displayValues : String(displayValue)
            );
        }
    }

    function normalizeFastChoices(values) {
        var seen = Object.create(null);
        var result = [];
        (values || []).forEach(function (value) {
            var text = String(value === undefined || value === null ? "" : value);
            if (!text || seen[text]) return;
            seen[text] = true;
            result.push(text);
        });
        return result;
    }

    function fastDropdownVisibleChoiceLimit() {
        var configured = DEFAULT_FAST_DROPDOWN_VISIBLE_CHOICES;
        try {
            if (typeof opts !== "undefined" && opts
                    && opts[FAST_DROPDOWN_LIMIT_OPTION] !== undefined) {
                configured = Number(opts[FAST_DROPDOWN_LIMIT_OPTION]);
            }
        } catch (error) {}
        if (!Number.isFinite(configured)) {
            configured = DEFAULT_FAST_DROPDOWN_VISIBLE_CHOICES;
        }
        return Math.max(
            10,
            Math.min(MAX_RENDERED_FAST_CHOICES, Math.round(configured))
        );
    }

    function fastDropdownInput(wrapper) {
        if (!wrapper) return null;
        return wrapper.querySelector("input[role='listbox']")
            || wrapper.querySelector(":scope > .container input:not([type='hidden'])");
    }

    function installFastDropdown(elemId, labelText, choiceOverride) {
        var wrapper = wrapperFor(elemId);
        if (!wrapper) return false;
        if (wrapper.querySelector(".sam3-fast-dropdown-trigger")) {
            if (Array.isArray(choiceOverride)
                    && typeof wrapper.__sam3FastDropdownSetChoices === "function") {
                wrapper.__sam3FastDropdownSetChoices(choiceOverride);
            }
            return true;
        }
        var input = fastDropdownInput(wrapper);
        var original = wrapper.querySelector(":scope > .container");
        var meta = metaForElemId(elemId);
        var choices = Array.isArray(choiceOverride)
            ? normalizeFastChoices(choiceOverride) : displayChoices(elemId);
        var isMulti = wrapper.classList.contains("multiselect")
            || !!(meta && meta.props && meta.props.multiselect);
        if (!input || !original || !choices.length) return false;

        var safeId = elemId.replace(/[^a-zA-Z0-9_-]/g, "-");
        var popoverId = "sam3-fast-dropdown-" + safeId;
        var listId = popoverId + "-list";
        var proxy = document.createElement("div");
        proxy.className = "sam3-fast-dropdown-proxy";
        proxy.setAttribute("data-multiselect", String(isMulti));
        var originalWidth = original.getBoundingClientRect().width;
        if (originalWidth > 0) {
            proxy.style.minWidth = Math.ceil(originalWidth) + "px";
        }
        var caption = document.createElement("span");
        caption.className = "sam3-fast-dropdown-label";
        caption.textContent = labelText;
        var trigger = document.createElement("button");
        trigger.type = "button";
        trigger.className = "sam3-fast-dropdown-trigger";
        trigger.setAttribute("aria-label", labelText);
        trigger.setAttribute("aria-haspopup", "listbox");
        trigger.setAttribute("aria-controls", listId);
        trigger.setAttribute("aria-expanded", "false");
        var value = document.createElement("span");
        value.className = "sam3-fast-dropdown-value";
        if (isMulti) value.classList.add("sam3-fast-dropdown-multi-value");
        var indicator = document.createElement("span");
        indicator.className = "sam3-fast-dropdown-indicator";
        indicator.setAttribute("aria-hidden", "true");
        trigger.appendChild(value);
        trigger.appendChild(indicator);
        var status = document.createElement("span");
        status.className = "sam3-fast-dropdown-status";
        status.setAttribute("aria-live", "polite");
        proxy.appendChild(caption);
        proxy.appendChild(trigger);
        proxy.appendChild(status);
        original.classList.add("sam3-fast-dropdown-original");
        wrapper.appendChild(proxy);

        var popover = document.createElement("div");
        popover.id = popoverId;
        popover.className = "sam3-fast-dropdown-popover";
        popover.setAttribute("popover", "auto");
        popover.setAttribute("data-open", "false");
        var search = document.createElement("input");
        search.type = "search";
        search.className = "sam3-fast-dropdown-search";
        search.placeholder = "목록 검색";
        search.setAttribute("aria-label", labelText + " 목록 검색");
        search.setAttribute("aria-controls", listId);
        var list = document.createElement("div");
        list.id = listId;
        list.className = "sam3-fast-dropdown-options";
        list.setAttribute("role", "listbox");
        list.setAttribute("aria-label", labelText);
        if (isMulti) list.setAttribute("aria-multiselectable", "true");
        var empty = document.createElement("p");
        empty.className = "sam3-fast-dropdown-empty";
        empty.textContent = "일치하는 항목이 없습니다";
        empty.hidden = true;
        popover.appendChild(search);
        popover.appendChild(list);
        popover.appendChild(empty);
        document.body.appendChild(popover);

        var allChoices = normalizeFastChoices(choices);
        var selectedValues = [];
        var optionButtons = [];
        var renderedChoiceLimit = fastDropdownVisibleChoiceLimit();
        var lastRenderedQuery = "";
        var lastOrderedChoices = [];

        function selectedTokenValues() {
            if (!isMulti) return [];
            return Array.prototype.slice.call(
                original.querySelectorAll(".token > span:first-child")
            ).map(function (span) {
                return span.textContent.trim();
            }).filter(Boolean);
        }

        function selectionFromGradio() {
            var currentInput = fastDropdownInput(wrapper) || input;
            return isMulti
                ? selectedTokenValues()
                : [choiceDisplayValue(meta, currentInput.value)];
        }

        function selectionKey(values) {
            return (values || []).map(String).join("\u001f");
        }

        function syncValue(nextValue) {
            selectedValues = normalizeFastChoices(
                Array.isArray(nextValue) ? nextValue : [nextValue]
            );
            if (isMulti) {
                value.textContent = "";
                selectedValues.forEach(function (choice) {
                    var chip = document.createElement("span");
                    chip.className = "sam3-fast-dropdown-chip";
                    chip.textContent = choice;
                    value.appendChild(chip);
                });
                if (!selectedValues.length) value.textContent = "선택 없음";
            } else {
                value.textContent = selectedValues[0] || "";
            }
            trigger.title = selectedValues.join(", ");
            optionButtons.forEach(function (option) {
                option.setAttribute(
                    "aria-selected",
                    String(selectedValues.indexOf(option.textContent) !== -1)
                );
            });
        }

        function syncFromGradio() {
            var currentInput = fastDropdownInput(wrapper) || input;
            syncValue(selectionFromGradio());
            trigger.disabled = !!currentInput.disabled;
            proxy.setAttribute("aria-disabled", String(!!currentInput.disabled));
        }

        function visibleOptions() {
            return optionButtons.slice();
        }

        function renderOptions(query) {
            var preserveLimit = arguments.length > 1 && arguments[1] === true;
            var normalizedQuery = String(query || "").trim().toLocaleLowerCase();
            if (!preserveLimit || normalizedQuery !== lastRenderedQuery) {
                renderedChoiceLimit = fastDropdownVisibleChoiceLimit();
            }
            lastRenderedQuery = normalizedQuery;
            var matches = allChoices.filter(function (choice) {
                return !normalizedQuery
                    || choice.toLocaleLowerCase().indexOf(normalizedQuery) !== -1;
            });
            var ordered = [];
            selectedValues.forEach(function (choice) {
                if (matches.indexOf(choice) !== -1 && ordered.indexOf(choice) === -1) {
                    ordered.push(choice);
                }
            });
            matches.forEach(function (choice) {
                if (ordered.indexOf(choice) === -1) ordered.push(choice);
            });
            lastOrderedChoices = ordered;
            var rendered = ordered.slice(0, renderedChoiceLimit);
            list.textContent = "";
            optionButtons = [];
            rendered.forEach(function (choice, index) {
                var option = document.createElement("button");
                option.type = "button";
                option.id = listId + "-option-" + index;
                option.className = "sam3-fast-dropdown-option";
                option.setAttribute("role", "option");
                option.setAttribute(
                    "aria-selected",
                    String(selectedValues.indexOf(choice) !== -1)
                );
                option.setAttribute("tabindex", "-1");
                option.textContent = choice;
                option.addEventListener("click", function () {
                    choose(choice);
                });
                option.addEventListener("keydown", function (event) {
                    var visible = visibleOptions();
                    var current = visible.indexOf(option);
                    var next = null;
                    if (event.key === "ArrowDown") next = visible[current + 1] || visible[0];
                    if (event.key === "ArrowUp") next = visible[current - 1] || visible[visible.length - 1];
                    if (event.key === "Home") next = visible[0];
                    if (event.key === "End") next = visible[visible.length - 1];
                    if (event.key === "Escape") {
                        closePopover();
                        trigger.focus();
                    }
                    if (next) {
                        event.preventDefault();
                        next.focus();
                    }
                });
                optionButtons.push(option);
                list.appendChild(option);
            });
            if (!matches.length) {
                empty.textContent = "일치하는 항목이 없습니다";
                empty.hidden = false;
            } else if (matches.length > rendered.length) {
                empty.textContent = matches.length + "개 중 "
                    + rendered.length + "개 표시 · 아래로 스크롤하면 더 표시";
                empty.hidden = false;
            } else {
                empty.hidden = true;
            }
            list.setAttribute("data-rendered-count", String(rendered.length));
            list.setAttribute(
                "data-has-more",
                String(rendered.length < lastOrderedChoices.length)
            );
            if (popoverIsOpen()) positionPopover();
        }

        function filterOptions() {
            renderOptions(search.value);
        }

        function popoverIsOpen() {
            try { return popover.matches(":popover-open"); }
            catch (error) { return popover.getAttribute("data-open") === "true"; }
        }

        function positionPopover() {
            var rect = trigger.getBoundingClientRect();
            var gutter = 8;
            var baseWidth = Math.max(rect.width, 280);
            var availableWidth = Math.max(240, window.innerWidth - gutter * 2);
            var availableHeight = Math.max(240, window.innerHeight - gutter * 2);
            var optionRowHeight = 44;
            var popoverChromeHeight = 64;
            var maxRows = Math.max(
                4,
                Math.floor(
                    (availableHeight - popoverChromeHeight) / optionRowHeight
                )
            );
            var itemCount = Math.max(1, optionButtons.length);
            var idealColumns = Math.max(1, Math.ceil(itemCount / maxRows));
            var maxColumns = Math.max(
                1,
                Math.floor(availableWidth / baseWidth)
            );
            var columns = Math.min(idealColumns, maxColumns);
            var hasMore = list.getAttribute("data-has-more") === "true";
            if (hasMore) {
                // Keep at least one row below the viewport so a real scrollbar
                // exists even on a large monitor. Reaching its end appends the
                // next configured page instead of hiding unknown choices.
                var maxColumnsWithOverflow = Math.max(
                    1,
                    Math.floor((itemCount - 1) / maxRows)
                );
                columns = Math.min(columns, maxColumnsWithOverflow);
            }
            var rows = Math.max(1, Math.ceil(itemCount / columns));
            var visibleRows = Math.min(rows, maxRows);
            if (hasMore && rows <= maxRows) {
                visibleRows = Math.max(4, rows - 1);
            }
            var needsScroll = hasMore || rows > maxRows;
            var width = Math.min(baseWidth * columns, availableWidth);
            var panelHeight = Math.min(
                availableHeight,
                popoverChromeHeight + visibleRows * optionRowHeight
            );
            var below = window.innerHeight - rect.bottom - gutter;
            var above = rect.top - gutter;
            var top;
            if (below >= panelHeight) {
                top = rect.bottom + 4;
            } else if (above >= panelHeight) {
                top = rect.top - panelHeight - 4;
            } else {
                top = gutter;
            }
            var left = Math.min(
                Math.max(gutter, rect.left),
                Math.max(gutter, window.innerWidth - width - gutter)
            );
            list.style.gridTemplateColumns = "repeat("
                + columns + ", minmax(0, 1fr))";
            list.style.maxHeight = Math.round(
                visibleRows * optionRowHeight
            ) + "px";
            list.style.overflowY = needsScroll ? "auto" : "visible";
            list.setAttribute("data-scroll", String(needsScroll));
            popover.style.left = Math.round(left) + "px";
            popover.style.top = Math.round(top) + "px";
            popover.style.width = Math.round(width) + "px";
            popover.style.maxHeight = Math.round(panelHeight) + "px";
        }

        function openPopover(focusSearch) {
            if (trigger.disabled || popoverIsOpen()) return;
            if (search.value) {
                search.value = "";
                renderOptions("");
            }
            positionPopover();
            popover.setAttribute("data-open", "true");
            if (typeof popover.showPopover === "function") popover.showPopover();
            trigger.setAttribute("aria-expanded", "true");
            window.setTimeout(function () {
                positionPopover();
                var selected = optionButtons.find(function (option) {
                    return option.getAttribute("aria-selected") === "true";
                });
                if (selected) selected.scrollIntoView({ block: "nearest" });
                if (focusSearch) search.focus();
            }, 0);
        }

        function closePopover() {
            if (!popoverIsOpen()) return;
            if (typeof popover.hidePopover === "function") popover.hidePopover();
            popover.setAttribute("data-open", "false");
            trigger.setAttribute("aria-expanded", "false");
        }

        async function choose(choice) {
            var previous = selectedValues.slice();
            var next;
            if (isMulti) {
                next = selectedValues.indexOf(choice) === -1
                    ? selectedValues.concat([choice])
                    : selectedValues.filter(function (item) {
                        return item !== choice;
                    });
            } else {
                next = [choice];
            }
            syncValue(next);
            if (!isMulti) closePopover();
            trigger.removeAttribute("aria-invalid");
            proxy.setAttribute("data-state", "loading");
            trigger.setAttribute("aria-busy", "true");
            status.textContent = "적용 중";
            try {
                await setWrapperValue(
                    elemId,
                    isMulti ? next : choice,
                    true
                );
                proxy.setAttribute("data-state", "success");
                status.textContent = "적용됨";
                window.setTimeout(function () {
                    if (proxy.getAttribute("data-state") === "success") {
                        proxy.removeAttribute("data-state");
                        status.textContent = "";
                    }
                }, 900);
            } catch (error) {
                console.error("[SAM3 Notebook] dropdown update failed:", error);
                syncValue(previous);
                proxy.setAttribute("data-state", "error");
                trigger.setAttribute("aria-invalid", "true");
                status.textContent = "적용하지 못했습니다";
            } finally {
                trigger.removeAttribute("aria-busy");
            }
        }

        function setChoices(nextChoices) {
            allChoices = normalizeFastChoices(nextChoices);
            renderOptions(search.value);
        }

        list.addEventListener("scroll", function () {
            if (list.getAttribute("data-has-more") !== "true") return;
            if (list.scrollTop + list.clientHeight < list.scrollHeight - 48) return;
            var previousScrollTop = list.scrollTop;
            renderedChoiceLimit += fastDropdownVisibleChoiceLimit();
            renderedChoiceLimit = Math.min(
                renderedChoiceLimit,
                lastOrderedChoices.length
            );
            renderOptions(search.value, true);
            list.scrollTop = previousScrollTop;
        });

        wrapper.__sam3FastDropdownSync = syncValue;
        wrapper.__sam3FastDropdownSetChoices = setChoices;
        wrapper.__sam3FastDropdownRefreshLimit = function () {
            renderOptions(search.value);
        };
        syncFromGradio();
        setChoices(allChoices);
        input.addEventListener("input", syncFromGradio);
        input.addEventListener("change", syncFromGradio);

        // The proxy hides the real component, so a server-side choices refresh
        // (Styles refresh, Forge UI preset switch, any extension's refresh
        // button) would otherwise be invisible: the option list was captured
        // once at install time. Gradio keeps window.gradio_config live — its
        // update queue assigns straight onto the same props objects the config
        // exposes (`s = k.reduce((P,D)=>(P[D.id]=D,P),{})` then
        // `$.props[y.prop] = E` in the frontend bundle) — so re-reading the
        // config picks up every refresh generically, with no per-control
        // endpoint. Only externally injected lists (choiceOverride, i.e. the
        // model dropdowns fed from /sdapi/v1/*) keep their own list.
        //
        // The check must stay O(1) on the hot path: this runs per visible
        // dropdown on an interval, and metaForElemId is a linear scan over
        // thousands of config components. Gradio replaces the array wholesale
        // on update, so comparing the retained reference is enough and the
        // label mapping only runs when it actually changed.
        var externalChoices = Array.isArray(choiceOverride);
        var lastConfigChoices = meta && meta.props ? meta.props.choices : null;

        function refreshChoicesFromConfig() {
            if (externalChoices || !meta || !meta.props) return;
            var liveChoices = meta.props.choices;
            if (liveChoices === lastConfigChoices) return;
            lastConfigChoices = liveChoices;
            // An empty refresh is a real state, not a no-op: bailing out here
            // would keep the previous options on screen forever, because the
            // new reference is already cached and never re-examined. The
            // comparison below handles [] correctly and clears the list.
            var next = normalizeFastChoices(choiceLabels(liveChoices));
            var same = next.length === allChoices.length
                && next.every(function (choice, index) {
                    return choice === allChoices[index];
                });
            if (same) return;
            setChoices(next);
        }

        var syncTimer = window.setInterval(function () {
            if (!wrapper.isConnected) {
                window.clearInterval(syncTimer);
                return;
            }
            if (!fastDropdownVisible(wrapper)) return;
            refreshChoicesFromConfig();
            if (
                !proxy.hasAttribute("data-state")
                && selectionKey(selectedValues)
                    !== selectionKey(selectionFromGradio())
            ) {
                syncFromGradio();
            }
        }, 800);
        var triggerWasOpenOnPointerDown = false;
        trigger.addEventListener("pointerdown", function () {
            triggerWasOpenOnPointerDown = popoverIsOpen();
        });
        trigger.addEventListener("click", function () {
            if (triggerWasOpenOnPointerDown) {
                triggerWasOpenOnPointerDown = false;
                closePopover();
                return;
            }
            if (popoverIsOpen()) closePopover();
            else openPopover(true);
        });
        trigger.addEventListener("keydown", function (event) {
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                event.preventDefault();
                openPopover(false);
                window.setTimeout(function () {
                    var visible = visibleOptions();
                    var selected = visible.find(function (option) {
                        return option.getAttribute("aria-selected") === "true";
                    });
                    var target = selected
                        || (event.key === "ArrowDown" ? visible[0] : visible[visible.length - 1]);
                    if (target) target.focus();
                }, 0);
            }
        });
        search.addEventListener("input", filterOptions);
        search.addEventListener("keydown", function (event) {
            if (event.key === "ArrowDown") {
                var first = visibleOptions()[0];
                if (first) {
                    event.preventDefault();
                    first.focus();
                }
            }
            if (event.key === "Escape") {
                closePopover();
                trigger.focus();
            }
        });
        popover.addEventListener("toggle", function (event) {
            var open = event.newState === "open";
            popover.setAttribute("data-open", String(open));
            trigger.setAttribute("aria-expanded", String(open));
        });
        document.addEventListener("pointerdown", function (event) {
            if (!popoverIsOpen()) return;
            if (!popover.contains(event.target) && !trigger.contains(event.target)) {
                closePopover();
            }
        }, true);
        window.addEventListener("resize", function () {
            if (popoverIsOpen()) positionPopover();
        }, { passive: true });
        window.addEventListener("scroll", function () {
            if (popoverIsOpen()) positionPopover();
        }, { passive: true, capture: true });
        return true;
    }

    function fastDropdownVisible(wrapper) {
        return !!(wrapper && wrapper.isConnected
            && (wrapper.offsetWidth || wrapper.offsetHeight
                || wrapper.getClientRects().length));
    }

    function fastDropdownLabel(wrapper) {
        var info = wrapper && wrapper.querySelector("[data-testid='block-info']");
        var input = fastDropdownInput(wrapper);
        return String(
            info && info.childNodes.length
                ? info.childNodes[0].textContent
                : (input && input.getAttribute("aria-label"))
                    || (wrapper && wrapper.id) || "선택"
        ).trim();
    }

    async function fetchDynamicDropdownChoices(wrapper) {
        if (!wrapper || !wrapper.id) return [];
        var endpoint = null;
        var toLabel = null;
        if (wrapper.id === "setting_sd_model_checkpoint") {
            endpoint = "/sdapi/v1/sd-models";
            toLabel = function (item) {
                return String(item && (item.title || item.model_name) || "")
                    .replace(/\s+\[[0-9a-fA-F]{8,}\]$/, "");
            };
        } else if (wrapper.id === "setting_sd_modules") {
            endpoint = "/sdapi/v1/sd-modules";
            toLabel = function (item) {
                return String(item && (item.model_name || item.title) || "");
            };
        }
        if (!endpoint || !toLabel) return [];
        var response = await window.fetch(endpoint, {
            method: "GET",
            credentials: "same-origin",
            cache: "no-store"
        });
        if (!response.ok) throw new Error("HTTP " + response.status);
        var payload = await response.json();
        if (!Array.isArray(payload)) return [];
        return normalizeFastChoices(payload.map(toLabel)).sort();
    }

    function queueDynamicDropdownWarm(wrapper, force) {
        if (!wrapper || !wrapper.id || fastDropdownWarming[wrapper.id]) return;
        if (!force && wrapper.getAttribute("data-sam3-fast-warm-attempted") === "true") {
            return;
        }
        wrapper.setAttribute("data-sam3-fast-warm-attempted", "true");
        fastDropdownWarming[wrapper.id] = true;
        fastDropdownWarmQueue = fastDropdownWarmQueue.then(async function () {
            if (!wrapper.isConnected) return;
            var choices = await fetchDynamicDropdownChoices(wrapper);
            if (!choices.length) return;
            installFastDropdown(
                wrapper.id,
                fastDropdownLabel(wrapper),
                choices
            );
        }).catch(function (error) {
            console.warn(
                "[SAM3 Notebook] dropdown warm-up skipped:",
                wrapper.id,
                error
            );
        }).then(function () {
            delete fastDropdownWarming[wrapper.id];
        });
    }

    function prepareGlobalFastDropdown(wrapper) {
        if (!wrapper || !wrapper.id || wrapperFor(wrapper.id) !== wrapper) return;
        if (wrapper.closest("#tab_settings, #settings")) return;
        var meta = metaForElemId(wrapper.id);
        if (!meta || !meta.props || meta.props.allow_custom_value) return;
        if (wrapper.querySelector(".sam3-fast-dropdown-trigger")) return;
        var choices = displayChoices(wrapper.id);
        if (choices.length) {
            // Deliberately NOT passing the list: the third argument means
            // "externally injected choices" (only the /sdapi/v1 model
            // dropdowns use it) and opts that control out of the live-config
            // refresh. installFastDropdown falls back to displayChoices, so
            // this keeps the same initial list while staying refreshable.
            installFastDropdown(wrapper.id, fastDropdownLabel(wrapper));
        } else if (fastDropdownVisible(wrapper)) {
            queueDynamicDropdownWarm(wrapper, false);
        }
    }

    function refreshFastModelDropdowns() {
        [
            "setting_sd_model_checkpoint",
            "setting_sd_modules"
        ].forEach(function (elemId) {
            var wrapper = wrapperFor(elemId);
            if (!wrapper) return;
            wrapper.removeAttribute("data-sam3-fast-warm-attempted");
            queueDynamicDropdownWarm(wrapper, true);
        });
    }

    function refreshFastDropdownVisibleLimits() {
        var root = app();
        if (!root || !root.querySelectorAll) return;
        root.querySelectorAll(".gradio-dropdown").forEach(function (wrapper) {
            if (typeof wrapper.__sam3FastDropdownRefreshLimit === "function") {
                wrapper.__sam3FastDropdownRefreshLimit();
            }
        });
    }

    function scheduleGlobalFastDropdownScan(delayMs) {
        if (fastDropdownScanTimer !== null) {
            window.clearTimeout(fastDropdownScanTimer);
        }
        fastDropdownScanTimer = window.setTimeout(function () {
            fastDropdownScanTimer = null;
            installGlobalFastDropdowns();
        }, Math.max(200, Number(delayMs) || 0));
    }

    function queueGlobalFastDropdownPrepare(wrapper) {
        if (!wrapper || !wrapper.isConnected
                || wrapper.getAttribute("data-sam3-fast-prepare-queued") === "true"
                || wrapper.querySelector(".sam3-fast-dropdown-trigger")) {
            return;
        }
        wrapper.setAttribute("data-sam3-fast-prepare-queued", "true");
        var prepare = function () {
            wrapper.removeAttribute("data-sam3-fast-prepare-queued");
            if (wrapper.isConnected && fastDropdownVisible(wrapper)) {
                prepareGlobalFastDropdown(wrapper);
            }
        };
        if (typeof window.requestIdleCallback === "function") {
            window.requestIdleCallback(prepare, { timeout: 1200 });
        } else {
            window.setTimeout(prepare, 120);
        }
    }

    function installGlobalFastDropdowns() {
        var root = app();
        if (!root || !root.querySelectorAll) return 0;
        if (!fastDropdownVisibilityObserver
                && typeof IntersectionObserver === "function") {
            fastDropdownVisibilityObserver = new IntersectionObserver(
                function (entries) {
                    var becameVisible = entries.some(function (entry) {
                        return entry.isIntersecting;
                    });
                    if (becameVisible) scheduleGlobalFastDropdownScan(300);
                },
                { root: null, threshold: 0 }
            );
        }
        var wrappers = Array.prototype.slice.call(
            root.querySelectorAll(".gradio-dropdown")
        );
        wrappers.forEach(function (wrapper) {
            if (!wrapper.id || wrapper.closest("#tab_settings, #settings")) return;
            var meta = metaForElemId(wrapper.id);
            if (!meta || !meta.props || meta.props.allow_custom_value) return;
            if (fastDropdownVisibilityObserver
                    && !seenIn(fastDropdownObserved, wrapper)) {
                rememberIn(fastDropdownObserved, wrapper);
                fastDropdownVisibilityObserver.observe(wrapper);
            }
            if (fastDropdownVisible(wrapper)) {
                queueGlobalFastDropdownPrepare(wrapper);
            }
        });
        if (!fastDropdownGlobalEventsInstalled) {
            fastDropdownGlobalEventsInstalled = true;
            root.addEventListener("click", function (event) {
                var refresh = event.target && event.target.closest
                    ? event.target.closest("#forge_refresh_checkpoint") : null;
                if (!refresh) return;
                fastDropdownModelRefreshPending = true;
                window.setTimeout(function () {
                    if (!fastDropdownModelRefreshPending) return;
                    fastDropdownModelRefreshPending = false;
                    refreshFastModelDropdowns();
                }, 4000);
            }, true);
            if (typeof onAfterUiUpdate === "function") {
                onAfterUiUpdate(function () {
                    scheduleGlobalFastDropdownScan(300);
                    if (!fastDropdownModelRefreshPending) return;
                    fastDropdownModelRefreshPending = false;
                    window.setTimeout(refreshFastModelDropdowns, 80);
                });
            }
            if (typeof onOptionsChanged === "function") {
                onOptionsChanged(function () {
                    window.setTimeout(refreshFastDropdownVisibleLimits, 0);
                });
            }
        }
        return wrappers.length;
    }

    function installFastDropdowns() {
        var installed = [
            installFastDropdown(
                "txt2img_sampling", "Sampling Method"
            ),
            installFastDropdown(
                "txt2img_scheduler", "Schedule Type"
            ),
            installFastDropdown(
                "script_list", "Script"
            ),
            installFastDropdown(
                "script_txt2img_xyz_plot_x_type", "X 유형"
            ),
            installFastDropdown(
                "script_txt2img_xyz_plot_y_type", "Y 유형"
            ),
            installFastDropdown(
                "script_txt2img_xyz_plot_z_type", "Z 유형"
            )
        ].filter(Boolean).length;
        installGlobalFastDropdowns();
        return installed;
    }

    function promptTextarea(negative) {
        var id = negative ? "txt2img_neg_prompt" : "txt2img_prompt";
        var wrapper = wrapperFor(id);
        return wrapper && wrapper.querySelector("textarea");
    }

    function readPrompt(negative) {
        var textarea = promptTextarea(negative);
        return textarea ? textarea.value : "";
    }

    function writePrompt(negative, value) {
        var textarea = promptTextarea(negative);
        if (!textarea) throw new Error(negative ? "네거티브 프롬프트를 찾지 못했습니다" : "프롬프트를 찾지 못했습니다");
        nativeSetValue(textarea, value);
    }

    function loraTokensFromPrompt() {
        return (readPrompt(false).match(/<lora:[^>]+>/gi) || []).join(", ");
    }

    function replaceLoraTokens(tokens) {
        var current = readPrompt(false)
            .replace(/<lora:[^>]+>/gi, "")
            .replace(/(?:,\s*){2,}/g, ", ")
            .replace(/^[\s,]+|[\s,]+$/g, "");
        var next = String(tokens || "").trim();
        writePrompt(false, current && next ? current + ", " + next : (current || next));
    }

    function xyzValue(axis) {
        var textId = "script_txt2img_xyz_plot_" + axis + "_values";
        var textWrapper = wrapperFor(textId);
        if (!textWrapper) return "";
        var textInput = textWrapper.querySelector("textarea, input[type='text']");
        // The whole XYZ panel can be hidden because another Script is active.
        // Check the value control's own mode, not offsetParent, so Undo still
        // captures its dormant text value before temporarily opening XYZ.
        var textMode = window.getComputedStyle(textWrapper).display !== "none";
        if (textMode && textInput) return textInput.value;
        var parent = textWrapper.parentElement;
        var choiceWrapper = parent && parent.querySelector(".gradio-dropdown.multiselect");
        if (choiceWrapper) {
            return Array.prototype.slice.call(
                choiceWrapper.querySelectorAll(".token > span:first-child")
            ).map(function (span) { return span.textContent.trim(); }).join(", ");
        }
        return textInput ? textInput.value : "";
    }

    function captureEntry(entry) {
        var target = TARGET_BY_KEY[entry.target];
        if (!target) throw new Error("지원하지 않는 항목: " + entry.target);
        if (entry.target === "prompt") entry.value = readPrompt(false);
        else if (entry.target === "negative_prompt") entry.value = readPrompt(true);
        else if (entry.target === "lora_tokens") entry.value = loraTokensFromPrompt();
        else if (target.kind === "xyz") {
            entry.axis_type = String(readWrapperDisplay(
                "script_txt2img_xyz_plot_" + target.axis + "_type"
            ) || "");
            entry.value = xyzValue(target.axis);
        } else if (target.elemId) {
            entry.value = String(readWrapperDisplay(target.elemId));
        }
        if (String(entry.value || "").length > MAX_VALUE_LENGTH) {
            throw new Error(
                "현재 값이 " + MAX_VALUE_LENGTH + "자 제한을 초과합니다"
            );
        }
    }

    function hasXyzEntries(entries) {
        return entries.some(function (entry) {
            var target = TARGET_BY_KEY[entry.target];
            return target && target.kind === "xyz";
        });
    }

    function translatedChoice(value) {
        var source = String(value);
        try {
            if (typeof getTranslation !== "function") return source;
            var translated = getTranslation(source);
            return translated === undefined || translated === null || translated === ""
                ? source : String(translated);
        } catch (error) {
            return source;
        }
    }

    function choiceIsAvailable(elemId, displayValue) {
        var meta = metaForElemId(elemId);
        var choices = meta && meta.props && Array.isArray(meta.props.choices)
            ? meta.props.choices : [];
        if (!choices.length) return true;
        return choices.some(function (choice) {
            var label = Array.isArray(choice) ? choice[0] : choice;
            var raw = Array.isArray(choice) ? choice[1] : choice;
            return String(displayValue) === String(label)
                || String(displayValue) === String(raw)
                || String(displayValue) === translatedChoice(label);
        });
    }

    function requireWrapper(elemId, label) {
        var wrapper = wrapperFor(elemId);
        if (!wrapper) throw new Error((label || elemId) + " 입력을 찾지 못했습니다");
        return wrapper;
    }

    function preflightEntries(entries) {
        entries.forEach(function (entry) {
            var target = TARGET_BY_KEY[entry.target];
            if (!target) throw new Error("지원하지 않는 항목: " + entry.target);
            if (entry.target === "prompt" || entry.target === "lora_tokens") {
                if (!promptTextarea(false)) {
                    throw new Error("프롬프트 입력을 찾지 못했습니다");
                }
                return;
            }
            if (entry.target === "negative_prompt") {
                if (!promptTextarea(true)) {
                    throw new Error("네거티브 프롬프트 입력을 찾지 못했습니다");
                }
                return;
            }
            if (target.kind === "xyz") {
                requireWrapper("script_list", "Scripts 선택기");
                requireWrapper(
                    "script_txt2img_xyz_plot_csv_mode",
                    "XYZ CSV 입력 모드"
                );
                var prefix = "script_txt2img_xyz_plot_" + target.axis;
                requireWrapper(prefix + "_type", target.axis.toUpperCase() + " 유형");
                requireWrapper(prefix + "_values", target.axis.toUpperCase() + " 값");
                if (!choiceIsAvailable(prefix + "_type", entry.axis_type || "Nothing")) {
                    throw new Error(
                        target.axis.toUpperCase() + " 축 유형을 찾지 못했습니다: "
                        + String(entry.axis_type || "Nothing")
                    );
                }
                return;
            }
            requireWrapper(target.elemId, target.label);
            if (target.kind === "number" && !Number.isFinite(Number(entry.value))) {
                throw new Error(target.label + " 값이 숫자가 아닙니다: " + entry.value);
            }
            if (target.kind === "choice"
                    && !choiceIsAvailable(target.elemId, entry.value)) {
                throw new Error(
                    target.label + " 선택지를 찾지 못했습니다: " + entry.value
                );
            }
        });
    }

    function captureApplyState(entries) {
        var state = {
            entries: entries.map(function (entry) {
                var captured = {
                    id: makeId("undo"),
                    target: entry.target,
                    value: "",
                    axis_type: ""
                };
                captureEntry(captured);
                return captured;
            }),
            xyz: null
        };
        if (hasXyzEntries(entries)) {
            state.xyz = {
                script: readWrapperDisplay("script_list"),
                csvMode: readWrapperDisplay(
                    "script_txt2img_xyz_plot_csv_mode"
                )
            };
        }
        return state;
    }

    async function ensureXyzReady() {
        if (!wrapperFor("script_list")) throw new Error("Scripts 선택기를 찾지 못했습니다");
        await setWrapperValue("script_list", "X/Y/Z plot", true);
        var xyzReady = await waitFor(function () {
            var xType = wrapperFor("script_txt2img_xyz_plot_x_type");
            return xType && xType.offsetParent !== null;
        }, 5000);
        if (!xyzReady) throw new Error("X/Y/Z Plot 화면을 열지 못했습니다");
        await setWrapperValue("script_txt2img_xyz_plot_csv_mode", true, true);
        await delay(180);
    }

    async function applyEntry(entry, xyzPrepared) {
        var target = TARGET_BY_KEY[entry.target];
        if (!target) return xyzPrepared;
        if (entry.target === "prompt") writePrompt(false, entry.value);
        else if (entry.target === "negative_prompt") writePrompt(true, entry.value);
        else if (entry.target === "lora_tokens") replaceLoraTokens(entry.value);
        else if (target.kind === "xyz") {
            if (!xyzPrepared) {
                await ensureXyzReady();
                xyzPrepared = true;
            }
            var prefix = "script_txt2img_xyz_plot_" + target.axis;
            await setWrapperValue(prefix + "_type", entry.axis_type || "Nothing", true);
            await delay(180);
            await setWrapperValue(prefix + "_values", entry.value || "", false);
        } else if (target.elemId) {
            await setWrapperValue(target.elemId, entry.value, false);
        }
        return xyzPrepared;
    }

    async function applyEntries(entries, label) {
        var ordered = entries.filter(function (entry) {
            return entry.target !== "lora_tokens";
        }).concat(entries.filter(function (entry) {
            return entry.target === "lora_tokens";
        }));
        var xyzPrepared = false;
        for (var i = 0; i < ordered.length; i++) {
            xyzPrepared = await applyEntry(ordered[i], xyzPrepared);
        }
        if (label) setStatus(label, "saved");
    }

    async function restoreApplyState(state, label) {
        if (!state || !Array.isArray(state.entries)) {
            throw new Error("되돌릴 상태가 없습니다");
        }
        preflightEntries(state.entries);
        if (state.xyz) {
            requireWrapper("script_list", "Scripts 선택기");
            requireWrapper(
                "script_txt2img_xyz_plot_csv_mode",
                "XYZ CSV 입력 모드"
            );
            if (!choiceIsAvailable("script_list", state.xyz.script)) {
                throw new Error(
                    "이전 Script 선택지를 찾지 못했습니다: " + state.xyz.script
                );
            }
        }
        await applyEntries(state.entries, null);
        if (state.xyz) {
            await setWrapperValue(
                "script_txt2img_xyz_plot_csv_mode",
                state.xyz.csvMode,
                true
            );
            await setWrapperValue("script_list", state.xyz.script, true);
        }
        if (label) setStatus(label, "saved");
    }

    async function applyPreset(preset) {
        if (!preset.entries.length) {
            setStatus("적용할 항목이 없습니다", "warning");
            return;
        }
        var previousState = null;
        setStatus(preset.name + " 적용 중…", "pending");
        try {
            preflightEntries(preset.entries);
            previousState = captureApplyState(preset.entries);
            await applyEntries(preset.entries, preset.name + " 적용됨");
            undoState = previousState;
            syncUndoButton();
        } catch (error) {
            console.error("[SAM3 Notebook] apply failed:", error);
            var message = String(error && error.message || error);
            if (previousState) {
                try {
                    await restoreApplyState(previousState, null);
                    message += " · 변경을 원래대로 복구했습니다";
                } catch (rollbackError) {
                    console.error(
                        "[SAM3 Notebook] apply rollback failed:",
                        rollbackError
                    );
                    message += " · 자동 복구도 실패: "
                        + String(rollbackError && rollbackError.message || rollbackError);
                }
            }
            setStatus(message, "error");
        }
    }

    async function applyUndo() {
        if (!undoState) return;
        var state = undoState;
        setStatus("되돌리는 중…", "pending");
        try {
            await restoreApplyState(state, "이전 값으로 되돌림");
            undoState = null;
            syncUndoButton();
        } catch (error) {
            setStatus(String(error && error.message || error), "error");
        }
    }

    function smallestSection(element, boundary, reject) {
        var candidate = element;
        for (var node = element; node && node !== boundary; node = node.parentElement) {
            if (reject && node.contains(reject)) break;
            if (node.classList && (
                node.classList.contains("gradio-group")
                || node.classList.contains("gradio-column")
                || node.classList.contains("form")
            )) candidate = node;
        }
        return candidate;
    }

    function directChildContaining(container, element) {
        var node = element;
        while (node && node.parentElement && node.parentElement !== container) {
            node = node.parentElement;
        }
        return node && node.parentElement === container ? node : null;
    }

    function scriptLayoutNodes(container, scriptList) {
        var content = directChildContaining(container, scriptList);
        if (!content || content === scriptList) content = container.firstElementChild || container;
        var selectorGroup = directChildContaining(content, scriptList);
        if (!selectorGroup) return [];
        var result = [];
        var node = selectorGroup;
        while (node) {
            // Forge appends every mutually exclusive Script panel after the
            // Script selector. Always-on extension accordions are before it.
            result.push(node);
            node = node.nextElementSibling;
        }
        return result;
    }

    function setPixelStyle(element, property, value) {
        var next = Math.round(value * 100) / 100 + "px";
        if (element.style[property] !== next) element.style[property] = next;
    }

    // Keep Forge/Gradio's selectable-script nodes under their original
    // #txt2img_settings parent. Reparenting those live Svelte components makes
    // every dropdown interaction trigger expensive reconciliation. Absolute
    // positioning gives the same center-column presentation without changing
    // component ownership.
    function positionScriptPanels(layout, settings, scriptColumn, nodes, scriptList) {
        if (!layout || !settings || !scriptColumn || !nodes.length) return;
        var body = scriptColumn.querySelector(":scope > div");
        if (!body) return;
        settings.classList.add("sam3-notebook-floating-script-owner");
        nodes.forEach(function (node) {
            node.classList.add("sam3-notebook-script-float");
        });

        var framePending = false;
        function sync() {
            framePending = false;
            if (!layout.isConnected || !settings.isConnected) return;
            var settingsRect = settings.getBoundingClientRect();
            var bodyRect = body.getBoundingClientRect();
            if (settingsRect.width < 1 || bodyRect.width < 1) return;
            var gap = parseFloat(
                getComputedStyle(layout).getPropertyValue("--layout-gap")
            );
            if (!Number.isFinite(gap)) gap = 8;
            var top = bodyRect.top - settingsRect.top;
            var left = bodyRect.left - settingsRect.left;
            var cursor = top;
            var visibleHeight = 0;

            nodes.forEach(function (node) {
                setPixelStyle(node, "left", left);
                setPixelStyle(node, "top", cursor);
                setPixelStyle(node, "width", bodyRect.width);
                if (getComputedStyle(node).display === "none") return;
                // Gradio temporarily counts its open dropdown menu in the
                // selector group's height. The actual #script_list control
                // keeps its stable height, so use that for the first row and
                // avoid throwing the chosen Script panel far down the page.
                var height = node === nodes[0] && scriptList
                    ? scriptList.getBoundingClientRect().height
                    : node.getBoundingClientRect().height;
                if (height <= 0) return;
                cursor += height + gap;
                visibleHeight += height + gap;
            });
            body.style.minHeight = Math.max(
                0,
                visibleHeight ? visibleHeight - gap : 0
            ) + "px";
        }

        function schedule() {
            if (framePending) return;
            framePending = true;
            window.requestAnimationFrame(sync);
        }

        if (typeof ResizeObserver === "function") {
            var resizeObserver = new ResizeObserver(schedule);
            [layout, settings, scriptColumn, body].concat(nodes).forEach(
                function (node) { resizeObserver.observe(node); }
            );
        }
        var scriptInput = scriptList && scriptList.querySelector("input");
        if (scriptInput) {
            scriptInput.addEventListener("change", function () {
                schedule();
                window.setTimeout(schedule, 120);
                window.setTimeout(schedule, 500);
            });
        }
        window.addEventListener("resize", schedule, { passive: true });
        schedule();
        window.setTimeout(schedule, 250);
        window.setTimeout(schedule, 1000);
    }

    // The three-column layout extracts the Generation tab's settings/gallery
    // from #txt2img_extra_tabs. Move the remaining Forge-owned tab strip only
    // after that extraction so Textual Inversion / Checkpoints / Lora stay
    // reachable directly below Prompt / Negative.
    function restoreForgeExtraNetworkTabs(layout) {
        var extraTabs = app().querySelector("#txt2img_extra_tabs");
        if (!extraTabs || layout.contains(extraTabs)) return;
        var host = layout.querySelector(".sam3-notebook-prompt");
        if (!host) return;
        extraTabs.classList.remove("sam3-notebook-original-root");
        extraTabs.classList.add("sam3-notebook-extra-tabs");
        host.appendChild(extraTabs);
    }

    function promptLayoutNodes() {
        var classic = app().querySelector("#txt2img_toprow");
        if (classic) return [classic];

        // Forge's Compact prompt layout does not create #txt2img_toprow.
        // Its prompt container and tools/style row live in settings while its
        // Generate box lives in results, so move all three after the gallery
        // boundary is resolved.
        var prompt = app().querySelector("#txt2img_prompt_container");
        var generate = app().querySelector("#txt2img_generate_box");
        var styleRow = app().querySelector(
            "#txt2img_settings .toprow-compact-stylerow"
        );
        return prompt && generate
            ? [prompt, generate].concat(styleRow ? [styleRow] : [])
            : [];
    }

    function mountLayout(notebookPanel) {
        var existingLayout = app().querySelector("#sam3_notebook_layout")
            || document.querySelector("#sam3_notebook_layout");
        if (existingLayout) {
            var existingGallery = existingLayout.querySelector(
                '[data-column="gallery"] > div'
            );
            if (notebookPanel && !existingLayout.contains(notebookPanel)) {
                if (!existingGallery) return false;
                existingGallery.appendChild(notebookPanel);
            }
            restoreForgeExtraNetworkTabs(existingLayout);
            document.documentElement.classList.add(
                "sam3-notebook-layout-active"
            );
            return true;
        }
        var promptNodes = promptLayoutNodes();
        var settings = app().querySelector("#txt2img_settings");
        var scriptContainer = app().querySelector("#txt2img_script_container");
        var scripts = scriptContainer && scriptContainer.querySelector("#script_list");
        var gallery = app().querySelector("#txt2img_gallery");
        var layoutHost = app().querySelector("#tab_txt2img");
        if (!promptNodes.length || !settings || !scriptContainer
                || !scripts || !gallery || !layoutHost) {
            return false;
        }

        // Resolve both sections before moving Compact's prompt/generate nodes
        // out of settings/results.
        var boundary = app().querySelector("#txt2img_extra_tabs") || settings.parentElement;
        var gallerySection = smallestSection(gallery, boundary, settings);
        var scriptNodes = scriptLayoutNodes(scriptContainer, scripts);
        var layout = document.createElement("main");
        layout.id = "sam3_notebook_layout";
        layout.innerHTML = [
            '<section class="sam3-notebook-prompt"></section>',
            '<section class="sam3-notebook-columns">',
            ' <div class="sam3-notebook-column" data-column="parameters"><h2>Parameters</h2><div></div></div>',
            ' <div class="sam3-notebook-column" data-column="scripts"><h2>Scripts</h2><div></div></div>',
            ' <div class="sam3-notebook-column" data-column="gallery"><h2>Gallery</h2><div></div></div>',
            '</section>'
        ].join("");

        var promptTarget = layout.querySelector(".sam3-notebook-prompt");
        if (promptNodes.length > 1) {
            promptTarget.classList.add("sam3-notebook-prompt-compact");
        }
        promptNodes.forEach(function (node) { promptTarget.appendChild(node); });
        var parameterTarget = layout.querySelector('[data-column="parameters"] > div');
        parameterTarget.appendChild(settings);
        var scriptsColumn = layout.querySelector('[data-column="scripts"]');
        var galleryTarget = layout.querySelector('[data-column="gallery"] > div');
        galleryTarget.appendChild(gallerySection);
        galleryTarget.appendChild(notebookPanel);

        restoreForgeExtraNetworkTabs(layout);
        Array.prototype.forEach.call(layoutHost.children, function (child) {
            if (child !== layout && child.classList) {
                child.classList.add("sam3-notebook-original-root");
            }
        });
        layoutHost.appendChild(layout);
        document.documentElement.classList.add("sam3-notebook-layout-active");
        positionScriptPanels(
            layout,
            settings,
            scriptsColumn,
            scriptNodes,
            scripts
        );
        return true;
    }

    function targetOptionSelect(entry) {
        var select = document.createElement("select");
        select.className = "sam3-notebook-target";
        select.setAttribute("aria-label", "교체할 항목");
        TARGETS.forEach(function (target) {
            var option = document.createElement("option");
            option.value = target.key;
            option.textContent = target.label;
            option.selected = target.key === entry.target;
            select.appendChild(option);
        });
        select.addEventListener("change", function () {
            entry.target = select.value;
            entry.value = "";
            delete entry.axis_type;
            scheduleSave();
            renderPresets();
        });
        return select;
    }

    function appendChoices(select, choices, selectedValue) {
        var seen = Object.create(null);
        if (selectedValue && choices.indexOf(selectedValue) === -1) {
            choices = [selectedValue].concat(choices);
        }
        choices.forEach(function (choice) {
            choice = String(choice);
            if (seen[choice]) return;
            seen[choice] = true;
            var option = document.createElement("option");
            option.value = choice;
            option.textContent = choice;
            option.selected = choice === String(selectedValue || "");
            select.appendChild(option);
        });
    }

    function valueEditor(entry) {
        var target = TARGET_BY_KEY[entry.target] || TARGETS[0];
        var editor = document.createElement("div");
        editor.className = "sam3-notebook-value-editor";

        if (target.kind === "xyz") {
            var typeLabel = document.createElement("label");
            typeLabel.textContent = target.axis.toUpperCase() + " 유형";
            var typeSelect = document.createElement("select");
            appendChoices(
                typeSelect,
                displayChoices("script_txt2img_xyz_plot_x_type"),
                entry.axis_type || ""
            );
            if (!typeSelect.options.length) {
                var blank = document.createElement("option");
                blank.value = "";
                blank.textContent = "유형을 선택하세요";
                typeSelect.appendChild(blank);
            }
            typeSelect.addEventListener("change", function () {
                entry.axis_type = typeSelect.value;
                scheduleSave();
            });
            typeLabel.appendChild(typeSelect);
            editor.appendChild(typeLabel);

            var valuesLabel = document.createElement("label");
            valuesLabel.textContent = target.axis.toUpperCase() + " 값";
            var values = document.createElement("textarea");
            values.rows = 2;
            values.maxLength = MAX_VALUE_LENGTH;
            values.placeholder = "예: 4, 5, 6";
            values.value = entry.value || "";
            values.addEventListener("input", function () {
                entry.value = values.value;
                scheduleSave();
            });
            valuesLabel.appendChild(values);
            editor.appendChild(valuesLabel);
            return editor;
        }

        if (target.kind === "choice") {
            var choiceLabel = document.createElement("label");
            choiceLabel.textContent = "저장할 값";
            var choiceSelect = document.createElement("select");
            appendChoices(choiceSelect, displayChoices(target.elemId), entry.value);
            choiceSelect.addEventListener("change", function () {
                entry.value = choiceSelect.value;
                scheduleSave();
            });
            choiceLabel.appendChild(choiceSelect);
            editor.appendChild(choiceLabel);
            return editor;
        }

        var valueLabel = document.createElement("label");
        valueLabel.textContent = "저장할 값";
        var input;
        if (target.kind === "number") {
            input = document.createElement("input");
            input.type = "number";
            input.step = "any";
        } else {
            input = document.createElement("textarea");
            input.rows = entry.target === "lora_tokens" ? 2 : 3;
            input.maxLength = MAX_VALUE_LENGTH;
            input.placeholder = entry.target === "lora_tokens"
                ? "<lora:name:1.0>, <lora:other:0.7>" : "";
        }
        input.value = entry.value || "";
        input.addEventListener("input", function () {
            entry.value = input.value;
            scheduleSave();
        });
        valueLabel.appendChild(input);
        editor.appendChild(valueLabel);
        return editor;
    }

    function entryRow(preset, entry) {
        var row = document.createElement("div");
        row.className = "sam3-notebook-entry";
        row.appendChild(targetOptionSelect(entry));
        row.appendChild(valueEditor(entry));

        var actions = document.createElement("div");
        actions.className = "sam3-notebook-entry-actions";
        var capture = document.createElement("button");
        capture.type = "button";
        capture.textContent = "현재값";
        capture.title = "현재 Forge UI 값을 이 항목에 가져오기";
        capture.addEventListener("click", function () {
            try {
                var captured = {
                    id: entry.id,
                    target: entry.target,
                    value: "",
                    axis_type: ""
                };
                captureEntry(captured);
                entry.value = captured.value;
                if (entry.target.indexOf("xyz_") === 0) {
                    entry.axis_type = captured.axis_type;
                }
                scheduleSave();
                renderPresets();
                setStatus("현재값을 가져왔습니다", "saved");
            } catch (error) {
                setStatus(String(error && error.message || error), "error");
            }
        });
        var remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "×";
        remove.title = "항목 삭제";
        remove.className = "danger";
        remove.addEventListener("click", function () {
            preset.entries = preset.entries.filter(function (candidate) {
                return candidate.id !== entry.id;
            });
            scheduleSave();
            renderPresets();
        });
        actions.appendChild(capture);
        actions.appendChild(remove);
        row.appendChild(actions);
        return row;
    }

    function nextPresetName() {
        var used = Object.create(null);
        model.presets.forEach(function (preset) { used[preset.name.toLowerCase()] = true; });
        var index = 1;
        while (used[("preset" + index).toLowerCase()]) index++;
        return "preset" + index;
    }

    function addPreset() {
        if (model.presets.length >= MAX_PRESETS) {
            setStatus("Notebook 프리셋 최대 개수에 도달했습니다", "error");
            return;
        }
        var preset = {
            id: makeId("preset"),
            name: nextPresetName(),
            entries: [
                { id: makeId("entry"), target: "prompt", value: "" }
            ]
        };
        model.presets.push(preset);
        selectedPresetId = preset.id;
        if (panel) panel.open = true;
        writeUiState();
        scheduleSave();
        renderPresets();
    }

    function startRename(preset, nameButton) {
        var input = document.createElement("input");
        input.type = "text";
        input.maxLength = 80;
        input.value = preset.name;
        input.className = "sam3-notebook-rename";
        var committed = false;
        function commit(cancel) {
            if (committed) return;
            committed = true;
            var next = cancel ? preset.name : input.value.trim();
            if (next && next !== preset.name) {
                preset.name = next;
                scheduleSave();
            }
            renderPresets();
        }
        input.addEventListener("keydown", function (event) {
            if (event.key === "Enter") commit(false);
            if (event.key === "Escape") commit(true);
        });
        input.addEventListener("blur", function () { commit(false); });
        nameButton.replaceWith(input);
        input.focus();
        input.select();
    }

    function presetCard(preset) {
        var card = document.createElement("article");
        card.className = "sam3-notebook-preset";
        card.setAttribute("data-preset-id", preset.id);
        if (selectedPresetId === preset.id) card.classList.add("selected");

        var header = document.createElement("header");
        var name = document.createElement("button");
        name.type = "button";
        name.className = "sam3-notebook-preset-name";
        name.textContent = preset.name;
        name.addEventListener("click", function () {
            selectedPresetId = selectedPresetId === preset.id ? null : preset.id;
            writeUiState();
            renderPresets();
        });
        header.appendChild(name);

        var count = document.createElement("span");
        count.className = "sam3-notebook-count";
        count.textContent = preset.entries.length + "개";
        header.appendChild(count);

        var apply = document.createElement("button");
        apply.type = "button";
        apply.textContent = "적용";
        apply.className = "primary";
        apply.addEventListener("click", function () { applyPreset(preset); });
        header.appendChild(apply);

        var rename = document.createElement("button");
        rename.type = "button";
        rename.textContent = "✎";
        rename.title = "프리셋 이름 변경";
        rename.addEventListener("click", function () { startRename(preset, name); });
        header.appendChild(rename);

        var menu = document.createElement("details");
        menu.className = "sam3-notebook-preset-menu";
        var summary = document.createElement("summary");
        summary.textContent = "…";
        summary.title = "프리셋 메뉴";
        menu.appendChild(summary);
        var deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.textContent = "프리셋 삭제";
        deleteButton.className = "danger";
        deleteButton.addEventListener("click", function () {
            if (!window.confirm("'" + preset.name + "' 프리셋을 삭제할까요?")) return;
            model.presets = model.presets.filter(function (candidate) {
                return candidate.id !== preset.id;
            });
            if (selectedPresetId === preset.id) selectedPresetId = null;
            writeUiState();
            scheduleSave();
            renderPresets();
        });
        menu.appendChild(deleteButton);
        header.appendChild(menu);
        card.appendChild(header);

        if (selectedPresetId === preset.id) {
            var body = document.createElement("div");
            body.className = "sam3-notebook-preset-body";
            var entries = document.createElement("div");
            entries.className = "sam3-notebook-entries";
            preset.entries.forEach(function (entry) {
                entries.appendChild(entryRow(preset, entry));
            });
            body.appendChild(entries);
            var addEntry = document.createElement("button");
            addEntry.type = "button";
            addEntry.className = "sam3-notebook-add-entry";
            addEntry.textContent = "＋ 항목 추가";
            addEntry.addEventListener("click", function () {
                if (preset.entries.length >= MAX_ENTRIES_PER_PRESET) {
                    setStatus(
                        "프리셋 하나에는 항목을 최대 "
                        + MAX_ENTRIES_PER_PRESET + "개 저장할 수 있습니다",
                        "error"
                    );
                    return;
                }
                preset.entries.push({
                    id: makeId("entry"),
                    target: "prompt",
                    value: ""
                });
                scheduleSave();
                renderPresets();
            });
            body.appendChild(addEntry);
            card.appendChild(body);
        }
        return card;
    }

    function renderPresets() {
        if (!presetList) return;
        presetList.textContent = "";
        var query = searchInput ? searchInput.value.trim().toLocaleLowerCase() : "";
        var visible = model.presets.filter(function (preset) {
            if (!query) return true;
            if (preset.name.toLocaleLowerCase().indexOf(query) !== -1) return true;
            return preset.entries.some(function (entry) {
                var target = TARGET_BY_KEY[entry.target];
                return String(entry.value || "").toLocaleLowerCase().indexOf(query) !== -1
                    || (target && target.label.toLocaleLowerCase().indexOf(query) !== -1);
            });
        });
        if (!visible.length) {
            var empty = document.createElement("p");
            empty.className = "sam3-notebook-empty";
            empty.textContent = model.presets.length
                ? "검색 결과가 없습니다."
                : "＋ 버튼으로 첫 프리셋을 추가하세요.";
            presetList.appendChild(empty);
            return;
        }
        visible.forEach(function (preset) {
            presetList.appendChild(presetCard(preset));
        });
    }

    function bindPanelReferences(details) {
        panel = details;
        presetList = details.querySelector("[data-notebook-presets]");
        statusElement = details.querySelector("[data-notebook-status]");
        searchInput = details.querySelector("[data-notebook-search]");
    }

    function syncUndoButton() {
        if (!panel) return;
        var button = panel.querySelector("[data-notebook-undo]");
        if (button) button.hidden = !undoState;
    }

    function isPlainObject(value) {
        return !!value && typeof value === "object" && !Array.isArray(value);
    }

    function readLegacyWorkspaceData() {
        try {
            var parsed = JSON.parse(
                window.localStorage.getItem(LEGACY_WORKSPACE_KEY) || "null"
            );
            return isPlainObject(parsed)
                && Number(parsed.schema) === 1
                && isPlainObject(parsed.slots)
                ? parsed : null;
        } catch (error) {
            return null;
        }
    }

    function legacyElemId(key) {
        var marker = "|id:";
        var start = String(key).indexOf(marker);
        if (start === -1) return null;
        var encoded = String(key).slice(start + marker.length).split("|")[0];
        try { return decodeURIComponent(encoded); }
        catch (error) { return encoded; }
    }

    function legacyRecordById(controls, elemId, usedKeys) {
        var keys = Object.keys(controls || {});
        for (var i = 0; i < keys.length; i++) {
            if (legacyElemId(keys[i]) === elemId
                    && isPlainObject(controls[keys[i]])) {
                if (usedKeys) usedKeys[keys[i]] = true;
                return controls[keys[i]];
            }
        }
        return null;
    }

    function legacyXyzChoiceRecord(controls, axis, usedKeys) {
        var prefix = "dropdown|xyz:" + axis + "-values-choice";
        var key = Object.keys(controls || {}).find(function (candidate) {
            return candidate.indexOf(prefix) === 0;
        });
        if (key && usedKeys) usedKeys[key] = true;
        return key && isPlainObject(controls[key]) ? controls[key] : null;
    }

    function legacyValue(record) {
        if (!record || !Object.prototype.hasOwnProperty.call(record, "value")) {
            return undefined;
        }
        if (Array.isArray(record.value)) return record.value.join(", ");
        return record.value === null || record.value === undefined
            ? "" : String(record.value);
    }

    function legacyEntry(target, value, axisType) {
        var text = String(value === undefined ? "" : value);
        if (text.length > MAX_VALUE_LENGTH) {
            throw new Error(
                "이전 Workspace 항목이 "
                + MAX_VALUE_LENGTH + "자 제한을 초과합니다"
            );
        }
        var entry = {
            id: makeId("entry"),
            target: target,
            value: text
        };
        if (target.indexOf("xyz_") === 0) {
            entry.axis_type = String(axisType || "Nothing");
        }
        return entry;
    }

    function legacySnapshotEntries(snapshot, stats) {
        if (!isPlainObject(snapshot) || !isPlainObject(snapshot.controls)) return [];
        var controls = snapshot.controls;
        var entries = [];
        var usedKeys = Object.create(null);

        // Legacy Workspaces stored LoRA tags in the complete txt2img prompt,
        // not in a separate field. Importing the prompt verbatim preserves
        // both the LoRA tokens and their original position.
        [
            ["prompt", "txt2img_prompt"],
            ["negative_prompt", "txt2img_neg_prompt"]
        ].concat(TARGETS.filter(function (target) {
            return target.elemId;
        }).map(function (target) {
            return [target.key, target.elemId];
        })).forEach(function (mapping) {
            var value = legacyValue(
                legacyRecordById(controls, mapping[1], usedKeys)
            );
            if (value !== undefined) entries.push(legacyEntry(mapping[0], value));
        });

        ["x", "y", "z"].forEach(function (axis) {
            var prefix = "script_txt2img_xyz_plot_" + axis;
            var typeRecord = legacyRecordById(
                controls, prefix + "_type", usedKeys
            );
            if (!typeRecord) return;
            var textRecord = legacyRecordById(
                controls, prefix + "_values", usedKeys
            );
            var choiceRecord = legacyXyzChoiceRecord(
                controls, axis, usedKeys
            );
            var valueRecord = choiceRecord && choiceRecord.active === true
                ? choiceRecord : textRecord;
            if (!valueRecord) valueRecord = choiceRecord;
            entries.push(legacyEntry(
                "xyz_" + axis,
                legacyValue(valueRecord),
                legacyValue(typeRecord)
            ));
        });
        if (stats) {
            stats.skippedControls += Object.keys(controls).filter(function (key) {
                return isPlainObject(controls[key]) && !usedKeys[key];
            }).length;
        }
        return entries;
    }

    function legacyWorkspaceConversion(raw) {
        var stats = { skippedControls: 0, emptySlots: 0 };
        if (!raw || !isPlainObject(raw.slots)) {
            return { presets: [], skippedControls: 0, emptySlots: 0 };
        }
        var order = [];
        function addSlot(value) {
            var slot = String(value || "");
            if (slot && order.indexOf(slot) === -1) order.push(slot);
        }
        if (Array.isArray(raw.slotOrder)) raw.slotOrder.forEach(addSlot);
        Object.keys(raw.slots).forEach(addSlot);

        var presets = order.map(function (slot) {
            var entries = legacySnapshotEntries(raw.slots[slot], stats);
            if (!entries.length) {
                stats.emptySlots += 1;
                return null;
            }
            var rawName = isPlainObject(raw.slotNames) ? raw.slotNames[slot] : "";
            var name = String(rawName || slot).trim() || slot;
            if (/^\d+$/.test(name)) name = "Workspace " + name;
            return {
                id: makeId("preset"),
                name: name.slice(0, MAX_NAME_LENGTH),
                entries: entries
            };
        }).filter(Boolean);
        return {
            presets: presets,
            skippedControls: stats.skippedControls,
            emptySlots: stats.emptySlots
        };
    }

    function legacyWorkspacePresets(raw) {
        return legacyWorkspaceConversion(raw).presets;
    }

    function syncLegacyImportButton(details) {
        var button = details && details.querySelector("[data-notebook-legacy]");
        if (button) button.hidden = !readLegacyWorkspaceData();
    }

    function importLegacyWorkspaces() {
        var conversion;
        var presets;
        try {
            conversion = legacyWorkspaceConversion(readLegacyWorkspaceData());
            presets = conversion.presets;
        } catch (error) {
            console.error("[SAM3 Notebook] legacy import failed:", error);
            setStatus(
                "이전 Workspace 변환 실패: "
                + String(error && error.message || error),
                "error"
            );
            return;
        }
        if (!presets.length) {
            var emptyMessage = "가져올 지원 대상 Workspace 값이 없습니다";
            if (conversion && conversion.skippedControls) {
                emptyMessage += " · 지원 대상 밖 컨트롤 "
                    + conversion.skippedControls + "개";
            }
            setStatus(emptyMessage, "warning");
            return;
        }
        var capacity = MAX_PRESETS - model.presets.length;
        if (capacity <= 0) {
            setStatus("Notebook 프리셋 최대 개수에 도달했습니다", "error");
            return;
        }
        var overflow = Math.max(0, presets.length - capacity);
        if (overflow) presets = presets.slice(0, capacity);
        var exclusions = [];
        if (conversion.skippedControls) {
            exclusions.push(
                "지원 대상 밖 컨트롤 " + conversion.skippedControls + "개"
            );
        }
        if (conversion.emptySlots) {
            exclusions.push("빈 Workspace " + conversion.emptySlots + "개");
        }
        if (overflow) exclusions.push("프리셋 한도 초과 " + overflow + "개");
        if (!window.confirm(
            "이전 Live Workspace " + presets.length
            + "개를 Notebook 프리셋 뒤에 추가할까요?\n"
            + "브라우저의 원본 Workspace 데이터는 삭제하지 않습니다."
            + (exclusions.length
                ? "\n\n가져오지 않는 항목: " + exclusions.join(" · ")
                : "")
        )) return;
        model.presets = model.presets.concat(presets);
        scheduleSave();
        renderPresets();
        setStatus(
            "이전 Workspace " + presets.length + "개"
            + (exclusions.length ? " · " + exclusions.join(" · ") + " 제외" : "")
            + " · 저장 대기…",
            "pending"
        );
    }

    function exportNotebook() {
        var blob = new Blob([JSON.stringify({
            version: 1,
            revision: model.revision,
            updated_at: model.updated_at,
            presets: model.presets
        }, null, 2)], { type: "application/json" });
        var anchor = document.createElement("a");
        var objectUrl = URL.createObjectURL(blob);
        anchor.href = objectUrl;
        anchor.download = "sam-extra-notebook-"
            + new Date().toISOString().replace(/[:.]/g, "-") + ".json";
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        setTimeout(function () { URL.revokeObjectURL(objectUrl); }, 1000);
    }

    function importNotebook(file) {
        if (!file) return;
        if (file.size > 2 * 1024 * 1024) {
            setStatus("2 MiB보다 큰 파일은 가져올 수 없습니다", "error");
            return;
        }
        var reader = new FileReader();
        reader.onload = function () {
            try {
                var imported = normalize(JSON.parse(String(reader.result || "")));
                if (!window.confirm(
                    "현재 Notebook의 " + model.presets.length + "개 프리셋을 "
                    + imported.presets.length + "개로 교체할까요?"
                )) return;
                model.presets = imported.presets;
                selectedPresetId = null;
                writeUiState();
                scheduleSave();
                renderPresets();
                setStatus("가져온 내용 저장 대기…", "pending");
            } catch (error) {
                setStatus(
                    "Notebook JSON을 읽지 못했습니다: "
                    + String(error && error.message || error),
                    "error"
                );
            }
        };
        reader.readAsText(file, "utf-8");
    }

    function seenIn(collection, value) {
        if (collection && typeof collection.has === "function") return collection.has(value);
        return collection.indexOf(value) !== -1;
    }

    function rememberIn(collection, value) {
        if (collection && typeof collection.add === "function") collection.add(value);
        else collection.push(value);
    }

    function handlePanelClick(event) {
        if (event.__sam3NotebookHandled) return;
        var target = event.target && event.target.closest
            ? event.target.closest(
                "[data-notebook-add], [data-notebook-undo], "
                + "[data-notebook-export], [data-notebook-legacy]"
            ) : null;
        if (!target) return;
        var currentPanel = target.closest("#sam3_notebook_panel");
        if (!currentPanel) return;
        event.__sam3NotebookHandled = true;
        bindPanelReferences(currentPanel);

        if (target.hasAttribute("data-notebook-add")) {
            event.preventDefault();
            event.stopPropagation();
            addPreset();
        } else if (target.hasAttribute("data-notebook-undo")) {
            event.preventDefault();
            event.stopPropagation();
            applyUndo();
        } else if (target.hasAttribute("data-notebook-export")) {
            exportNotebook();
        } else if (target.hasAttribute("data-notebook-legacy")) {
            importLegacyWorkspaces();
        }
    }

    function handlePanelInput(event) {
        var target = event.target;
        if (!target || !target.matches
                || !target.matches("#sam3_notebook_panel [data-notebook-search]")) {
            return;
        }
        var currentPanel = target.closest("#sam3_notebook_panel");
        if (!currentPanel) return;
        bindPanelReferences(currentPanel);
        writeUiState();
        renderPresets();
    }

    function handlePanelChange(event) {
        var target = event.target;
        if (!target || !target.matches
                || !target.matches("#sam3_notebook_panel [data-notebook-import]")) {
            return;
        }
        var currentPanel = target.closest("#sam3_notebook_panel");
        if (!currentPanel) return;
        bindPanelReferences(currentPanel);
        importNotebook(target.files && target.files[0]);
        target.value = "";
    }

    // Gradio can replace or reparent extension-owned DOM during reconciliation.
    // Bind both the live Gradio root and the concrete panel. The root catches a
    // replaced panel; the panel binding keeps the controls working after moves.
    function installDelegatedPanelEvents() {
        var eventRoot = app();
        if (!eventRoot || seenIn(delegatedEventRoots, eventRoot)) return;
        rememberIn(delegatedEventRoots, eventRoot);
        eventRoot.addEventListener("click", handlePanelClick, true);
        eventRoot.addEventListener("input", handlePanelInput, true);
        eventRoot.addEventListener("change", handlePanelChange, true);
    }

    function installPanelEvents(details) {
        if (!details || seenIn(boundPanels, details)) return;
        rememberIn(boundPanels, details);
        details.addEventListener("click", handlePanelClick, true);
        details.addEventListener("input", handlePanelInput, true);
        details.addEventListener("change", handlePanelChange, true);
        details.addEventListener("toggle", writeUiState);
    }

    function buildPanel() {
        var details = document.createElement("details");
        details.id = "sam3_notebook_panel";
        details.innerHTML = [
            '<summary>',
            ' <strong>Notebook</strong>',
            ' <span class="sam3-notebook-status" data-notebook-status aria-live="polite"></span>',
            '</summary>',
            '<div class="sam3-notebook-head-actions">',
            ' <button type="button" data-notebook-undo hidden title="직전 적용 되돌리기">↶</button>',
            ' <button type="button" data-notebook-add title="프리셋 추가">＋</button>',
            '</div>',
            '<div class="sam3-notebook-toolbar">',
            ' <input type="search" data-notebook-search placeholder="프리셋 검색">',
            ' <button type="button" data-notebook-legacy hidden>이전 Workspaces 가져오기</button>',
            ' <button type="button" data-notebook-export>내보내기</button>',
            ' <label class="sam3-notebook-import">불러오기',
            '  <input type="file" accept="application/json,.json" data-notebook-import>',
            ' </label>',
            '</div>',
            '<div class="sam3-notebook-presets" data-notebook-presets></div>'
        ].join("");
        var uiState = readUiState();
        details.open = uiState.open !== false;
        selectedPresetId = uiState.selectedPresetId || null;
        bindPanelReferences(details);
        searchInput.value = uiState.search || "";
        syncLegacyImportButton(details);
        installPanelEvents(details);
        return details;
    }

    async function loadNotebook() {
        setStatus("불러오는 중…", "pending");
        try {
            var response = await window.fetch(API_PATH, {
                method: "GET",
                credentials: "same-origin",
                cache: "no-store",
                headers: { "X-SAM3-Notebook": "1" }
            });
            if (!response.ok) throw await responseError(response);
            model = normalize(await response.json());
            var recovered = recoverDraftIfSafe(model);
            renderPresets();
            if (recovered) {
                setStatus("닫기 전 변경 복구 · 저장 대기…", "pending");
                scheduleSave();
            } else {
                setStatus(model.presets.length + "개 프리셋", "saved");
            }
        } catch (error) {
            console.error("[SAM3 Notebook] load failed:", error);
            setStatus(
                "Notebook 불러오기 실패: "
                + String(error && error.message || error),
                "error"
            );
        }
    }

    async function start() {
        var ready = await waitFor(function () {
            var scriptContainer = app().querySelector("#txt2img_script_container");
            return promptLayoutNodes().length
                && app().querySelector("#txt2img_settings")
                && scriptContainer
                && scriptContainer.querySelector("#script_list")
                && app().querySelector("#txt2img_gallery")
                && app().querySelector("#tab_txt2img");
        }, 30000);
        if (!ready) {
            console.warn("[SAM3 Notebook] txt2img UI was not ready");
            return;
        }
        await ensureGradioConfig();
        // Install on the actual live Gradio root, not the early page shell.
        installDelegatedPanelEvents();
        var existingPanel = document.querySelector("#sam3_notebook_panel");
        if (existingPanel) {
            bindPanelReferences(existingPanel);
            syncLegacyImportButton(existingPanel);
            installPanelEvents(existingPanel);
            installFastDropdowns();
            await loadNotebook();
            return;
        }
        var notebookPanel = buildPanel();
        if (!mountLayout(notebookPanel)) {
            var gallery = app().querySelector("#txt2img_gallery");
            var fallback = gallery && gallery.parentElement;
            if (fallback) fallback.appendChild(notebookPanel);
        }
        installFastDropdowns();
        await loadNotebook();
    }

    // Opt-in test seam only. Production pages never create this object, while
    // jsdom can exercise the real dropdown DOM/event implementation without
    // booting the complete Forge txt2img layout.
    if (window.__sam3NotebookTestHooks
            && typeof window.__sam3NotebookTestHooks === "object") {
        window.__sam3NotebookTestHooks.installFastDropdown = installFastDropdown;
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () { setTimeout(start, 250); });
    } else {
        document.addEventListener("DOMContentLoaded", function () {
            setTimeout(start, 1200);
        });
    }
})();
