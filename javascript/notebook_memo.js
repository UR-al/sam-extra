/* SAM3 Notebook 메모장 — Notebook 패널 안의 "메모" 탭.
 *
 * notebook.js 가 만든 #sam3_notebook_panel 에 탭 줄과 메모 칸만 덧붙인다. notebook.js 는 고치지 않는다 —
 * 프리셋 쪽 요소는 패널의 data-sam3-notebook-tab 속성과 이 파일이 넣는 CSS 로 숨기기만 한다.
 *
 * 저장소: /sam3-notebook/memos (sam3ext/notebook_memos.py) — notebook.json 옆의 memos.json.
 * UR_IV 앱도 같은 경로로 오프라인 편집을 병합한다. 그래서 저장은 메모 하나씩 PUT 하고, 마지막으로 본 서버
 * updated_at 을 base_updated_at 으로 붙인다. 그사이 다른 곳에서 바뀌었으면 409 가 오고, 서버 쪽은 원래 메모에
 * 두고 내 변경은 "(충돌 사본)" 새 메모로 남긴다 — 어느 쪽도 버리지 않는다. 삭제도 마지막으로 본 시각을
 * ?base_updated_at= 으로 붙여, 그 뒤 다른 곳에서 고친 메모는 지우지 않는다(409).
 *
 * 저장 전 편집은 localStorage 초안으로 남긴다. 탭마다 제 칸(TAB_ID)만 고치고, 닫혔거나(pagehide) 소식이 끊긴
 * 탭의 칸만 다른 탭이 이어받는다 — 새 탭이 지난 세션이나 열려 있는 다른 탭의 초안을 지우거나 가로채지 않게.
 */
(function () {
    "use strict";

    if (window.__sam3NotebookMemoLoaded) return;
    window.__sam3NotebookMemoLoaded = true;

    var API_PATH = "/sam3-notebook/memos";
    var UI_KEY = "sam-extra.notebook.memo.ui.v1";
    // { "<탭 id>": { at: ms, hidden: bool, closed: bool, drafts: { "<메모 id>": 초안 } } }
    var DRAFT_KEY = "sam-extra.notebook.memo.drafts.v2";
    var LEGACY_DRAFT_KEY = "sam-extra.notebook.memo.drafts.v1";   // 탭 구분 없는 { "<메모 id>": 초안 }
    var LEGACY_OWNER = "legacy-v1";
    var DRAFT_HEARTBEAT_MS = 5000;
    var DRAFT_STALE_MS = 30000;            // 보이던 탭이 이만큼 소식이 없으면 닫힌(멈춘) 것으로 본다
    var DRAFT_STALE_HIDDEN_MS = 300000;    // 숨은 탭은 브라우저가 타이머를 분 단위로 늦춘다
    var STYLE_ID = "sam3_notebook_memo_style";
    var SAVE_DELAY_MS = 600;
    var MOUNT_POLL_MS = 1000;
    var FOCUS_REFRESH_GAP_MS = 1500;
    var MAX_TITLE_LENGTH = 120;
    var MAX_TEXT_LENGTH = 100000;
    var CONFLICT_SUFFIX = " (충돌 사본)";
    var UNTITLED = "제목 없음";
    var ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$/;

    // 메모 항목: { id, title, text, created_at, updated_at, base, version, savedVersion }
    //   updated_at   목록 정렬·표시용 — 마지막 로컬 편집 시각이나 서버 시각
    //   base         마지막으로 본 서버 updated_at. 서버에 아직 없으면 null (PUT 의 base_updated_at)
    //   version      로컬 편집마다 +1. savedVersion 과 다르면 저장할 것이 남은 것이다
    var memos = [];
    var selectedId = null;
    var tab = "presets";
    var available = null;            // null: 아직 모름, true: 경로 있음, false: 404 (새 경로 없는 Forge)
    var queue = Promise.resolve();   // 저장·삭제·새로 고침을 한 줄로 세운다(응답 순서가 뒤섞이지 않게)
    var saveTimers = Object.create(null);
    // 삭제를 큐에 넣고 아직 보내지 않은 메모: id → { cancelled }. 다른 곳의 새 편집을 알게 되면 취소한다.
    var pendingDeletes = Object.create(null);
    var refreshPromise = null;
    var lastRefreshAt = 0;
    var heartbeatTimer = null;
    var uiStateRead = false;
    var mountTimer = null;
    var ui = {
        panel: null, tabs: null, section: null, list: null, title: null, text: null,
        status: null, notice: null, noticeText: null, create: null, remove: null
    };

    function app() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function randomId(prefix) {
        if (window.crypto && typeof window.crypto.randomUUID === "function") {
            return prefix + window.crypto.randomUUID();
        }
        return prefix + Date.now().toString(36) + "-"
            + Math.random().toString(36).slice(2, 12);
    }

    function makeId() {
        return randomId("memo-");
    }

    var TAB_ID = randomId("tab-");   // 이 페이지(탭)의 초안 칸 이름

    // 글자 수 한도는 서버(파이썬 len)처럼 코드 포인트로 센다. UTF-16 단위로 자르면 이모지 한가운데서 끊겨
    // 짝 없는 서로게이트가 남고, 서버는 그런 글을 저장하지 못한다.
    function clip(value, limit) {
        var text = asText(value);
        if (text.length <= limit) return text;
        return Array.from(text).slice(0, limit).join("");
    }

    function nowIso() {
        return new Date().toISOString();
    }

    function asText(value) {
        return value === undefined || value === null ? "" : String(value);
    }

    function errorText(error) {
        return String(error && error.message || error);
    }

    function timeOf(iso) {
        var time = iso ? Date.parse(iso) : NaN;
        return isNaN(time) ? 0 : time;
    }

    function pad(number) {
        return number < 10 ? "0" + number : String(number);
    }

    function formatTime(iso) {
        var time = timeOf(iso);
        if (!time) return "";
        var date = new Date(time);
        var today = new Date();
        if (date.toDateString() === today.toDateString()) {
            return pad(date.getHours()) + ":" + pad(date.getMinutes());
        }
        var day = pad(date.getMonth() + 1) + "-" + pad(date.getDate());
        return date.getFullYear() === today.getFullYear()
            ? day : date.getFullYear() + "-" + day;
    }

    function clock() {
        var date = new Date();
        return pad(date.getHours()) + ":" + pad(date.getMinutes());
    }

    // ---- 메모 모델 --------------------------------------------------------

    function localMemo(fields) {
        return {
            id: fields.id,
            title: asText(fields.title),
            text: asText(fields.text),
            created_at: fields.created_at || null,
            updated_at: fields.updated_at || null,
            base: fields.base || null,
            version: 0,
            savedVersion: 0
        };
    }

    function fromServer(raw) {
        return localMemo({
            id: String(raw.id),
            title: raw.title,
            text: raw.text,
            created_at: raw.created_at,
            updated_at: raw.updated_at,
            base: raw.updated_at
        });
    }

    function isDirty(memo) {
        return memo.version !== memo.savedVersion;
    }

    function findMemo(id) {
        if (!id) return null;
        for (var index = 0; index < memos.length; index++) {
            if (memos[index].id === id) return memos[index];
        }
        return null;
    }

    function sortMemos(list) {
        return list.slice().sort(function (left, right) {
            var diff = timeOf(right.updated_at) - timeOf(left.updated_at);
            if (diff) return diff;
            return left.id < right.id ? 1 : (left.id > right.id ? -1 : 0);
        });
    }

    function firstLine(text) {
        var lines = asText(text).split(/\r?\n/);
        for (var index = 0; index < lines.length; index++) {
            var line = lines[index].trim();
            if (line) return clip(line, 80);
        }
        return "";
    }

    function displayTitle(memo) {
        return memo.title.trim() || firstLine(memo.text) || UNTITLED;
    }

    function conflictTitle(title) {
        var base = asText(title).trim() || UNTITLED;
        if (base.slice(-CONFLICT_SUFFIX.length) === CONFLICT_SUFFIX) {
            return clip(base, MAX_TITLE_LENGTH);
        }
        return clip(base, MAX_TITLE_LENGTH - CONFLICT_SUFFIX.length) + CONFLICT_SUFFIX;
    }

    // ---- 브라우저 저장 (탭·선택, 닫기 전 초안) ------------------------------

    function readJsonKey(key, fallback) {
        try {
            var parsed = JSON.parse(window.localStorage.getItem(key) || "null");
            return parsed && typeof parsed === "object" && !Array.isArray(parsed)
                ? parsed : fallback;
        } catch (error) {
            return fallback;
        }
    }

    function writeJsonKey(key, value) {
        try {
            if (value === null) window.localStorage.removeItem(key);
            else window.localStorage.setItem(key, JSON.stringify(value));
        } catch (error) {}
    }

    function writeUiState() {
        writeJsonKey(UI_KEY, { tab: tab, selectedId: selectedId });
    }

    // 모든 탭의 초안. 옛 모양(v1)은 닫힌 탭 하나로 읽고, 다음 쓰기 때 v2 로 옮긴다.
    function readDraftStore() {
        var store = readJsonKey(DRAFT_KEY, {});
        var legacy = readJsonKey(LEGACY_DRAFT_KEY, null);
        if (legacy && !store[LEGACY_OWNER]) {
            store[LEGACY_OWNER] = { at: 0, hidden: true, closed: true, drafts: legacy };
        }
        return store;
    }

    function writeDraftStore(store) {
        writeJsonKey(DRAFT_KEY, Object.keys(store).length ? store : null);
        writeJsonKey(LEGACY_DRAFT_KEY, null);
    }

    function ownDrafts() {
        var drafts = null;
        memos.forEach(function (memo) {
            if (!isDirty(memo)) return;
            drafts = drafts || {};
            drafts[memo.id] = {
                title: memo.title,
                text: memo.text,
                base: memo.base,
                created_at: memo.created_at,
                updated_at: memo.updated_at
            };
        });
        return drafts;
    }

    // 저장 전 편집을 초안으로 남겨 창을 닫아도 다음에 다시 저장한다. 이 탭 칸만 고친다 — 다른 탭이나 아직
    // 복구하지 않은 지난 세션의 초안은 그대로 둔다. adopted: 이 탭이 이어받아 이제 지울 다른 칸들.
    function writeDrafts(adopted) {
        var store = readDraftStore();
        (adopted || []).forEach(function (owner) { delete store[owner]; });
        var drafts = ownDrafts();
        if (drafts) {
            store[TAB_ID] = { at: Date.now(), hidden: !!document.hidden, closed: false, drafts: drafts };
        } else {
            delete store[TAB_ID];
        }
        writeDraftStore(store);
        setHeartbeat(!!drafts);
    }

    // 초안이 있는 동안 '아직 열려 있다'고 알린다. 다른 탭이 가져갔어도 다시 적는다(이 탭도 저장한다).
    function setHeartbeat(on) {
        if (on && !heartbeatTimer) {
            heartbeatTimer = setInterval(function () { writeDrafts(); }, DRAFT_HEARTBEAT_MS);
        } else if (!on && heartbeatTimer) {
            clearInterval(heartbeatTimer);
            heartbeatTimer = null;
        }
    }

    function markDraftsClosed() {
        var store = readDraftStore();
        if (!store[TAB_ID]) return;
        store[TAB_ID].closed = true;
        writeDraftStore(store);
    }

    function draftOwnerGone(section, now) {
        if (!section || typeof section !== "object" || section.closed === true) return true;
        var at = typeof section.at === "number" ? section.at : 0;
        return now - at > (section.hidden === false ? DRAFT_STALE_MS : DRAFT_STALE_HIDDEN_MS);
    }

    function adoptDraft(id, draft) {
        if (!ID_RE.test(id) || !draft || typeof draft !== "object") return null;
        if (pendingDeletes[id]) return null;                               // 이 탭에서 지우는 중
        var title = clip(draft.title, MAX_TITLE_LENGTH);
        var text = clip(draft.text, MAX_TEXT_LENGTH);
        var memo = findMemo(id);
        if (memo && memo.title === title && memo.text === text) return null;   // 이미 저장됐다
        if (memo && isDirty(memo)) {
            // 이 탭도 같은 메모를 고치는 중이다 — 어느 쪽도 덮지 않게 초안은 사본으로 살린다.
            memo = localMemo({ id: makeId(), title: conflictTitle(title), text: text, created_at: nowIso() });
            memo.updated_at = memo.created_at;
            memo.version++;
            memos.unshift(memo);
            return memo;
        }
        if (!memo) {
            memo = localMemo({ id: id, created_at: draft.created_at });
            memos.unshift(memo);
        }
        memo.title = title;
        memo.text = text;
        // 초안이 딛고 있던 서버 시각 그대로 저장한다. 그사이 서버가 바뀌었으면 409 → 충돌 사본.
        memo.base = typeof draft.base === "string" ? draft.base : null;
        memo.updated_at = typeof draft.updated_at === "string" ? draft.updated_at : nowIso();
        memo.version++;
        return memo;
    }

    // 닫혔거나 소식이 끊긴 탭(지난 세션 포함)의 초안만 이어받는다. 열려 있는 탭의 초안은 그 탭이 저장한다.
    function recoverDrafts() {
        var store = readDraftStore();
        var now = Date.now();
        var adopted = [];
        var recovered = [];
        Object.keys(store).forEach(function (owner) {
            if (owner === TAB_ID || !draftOwnerGone(store[owner], now)) return;
            adopted.push(owner);
            var drafts = store[owner] && store[owner].drafts;
            if (!drafts || typeof drafts !== "object") return;
            Object.keys(drafts).forEach(function (id) {
                var memo = adoptDraft(id, drafts[id]);
                if (memo) recovered.push(memo.id);
            });
        });
        if (!adopted.length) return recovered;
        memos = sortMemos(memos);
        writeDrafts(adopted);
        return recovered;
    }

    // ---- 서버 -------------------------------------------------------------

    function request(method, url, body) {
        var init = {
            method: method,
            credentials: "same-origin",
            cache: "no-store",
            headers: { "X-SAM3-Notebook": "1" }
        };
        if (body !== undefined) {
            init.headers["Content-Type"] = "application/json";
            init.body = JSON.stringify(body);
        }
        return window.fetch(url, init);
    }

    function itemUrl(id) {
        return API_PATH + "/" + encodeURIComponent(id);
    }

    async function readJson(response) {
        try {
            return await response.json();
        } catch (error) {
            return null;
        }
    }

    async function responseError(response) {
        var payload = await readJson(response);
        var detail = payload && payload.detail ? String(payload.detail) : "";
        return new Error(detail || ("HTTP " + response.status));
    }

    function markMissingRoutes() {
        available = false;
        setStatus("이 Forge 에는 아직 메모 저장소가 없습니다 — Forge 를 다시 시작하세요", "warning");
        renderAll();
    }

    function enqueue(task) {
        var run = queue.then(task);
        queue = run.catch(function () {});
        return run;
    }

    function markEdited(memo) {
        memo.version++;
        memo.updated_at = nowIso();
        writeDrafts();
    }

    function scheduleSave(memo) {
        markEdited(memo);
        setStatus("저장 대기…", "pending");
        if (saveTimers[memo.id]) clearTimeout(saveTimers[memo.id]);
        saveTimers[memo.id] = setTimeout(function () {
            queueSave(memo.id);
        }, SAVE_DELAY_MS);
    }

    function queueSave(id) {
        if (saveTimers[id]) {
            clearTimeout(saveTimers[id]);
            delete saveTimers[id];
        }
        return enqueue(function () { return saveNow(id); }).catch(function (error) {
            console.error("[SAM3 Notebook] memo save failed:", error);
            setStatus("메모 저장 실패: " + errorText(error), "error");
        });
    }

    async function saveNow(id) {
        var memo = findMemo(id);
        if (!memo || !isDirty(memo)) return;
        var version = memo.version;
        var body = {
            title: memo.title,
            text: memo.text,
            updated_at: memo.updated_at,
            base_updated_at: memo.base
        };
        if (memo.created_at) body.created_at = memo.created_at;
        setStatus("저장 중…", "pending");
        var response = await request("PUT", itemUrl(id), body);
        if (response.status === 409) {
            var payload = await readJson(response);
            var stored = payload && payload.memo;
            if (stored && typeof stored === "object") {
                if (stored.deleted !== true && stored.title === body.title && stored.text === body.text) {
                    // 서버가 이미 같은 글이다(다른 탭이 이 초안을 먼저 올린 경우 등) — 사본 없이 그 시각을 딛는다.
                    acceptSaved(memo, stored, version);
                    return;
                }
                resolveConflict(memo, stored);
                return;
            }
            throw new Error(payload && payload.detail ? String(payload.detail) : "HTTP 409");
        }
        if (response.status === 404) {
            markMissingRoutes();
            throw new Error("메모 저장소가 없습니다");
        }
        if (!response.ok) throw await responseError(response);
        acceptSaved(memo, ((await readJson(response)) || {}).memo || {}, version);
    }

    // 서버가 version 판을 저장했다(또는 이미 같은 글을 갖고 있다).
    function acceptSaved(memo, saved, version) {
        available = true;
        memo.base = saved.updated_at || memo.base;
        memo.created_at = saved.created_at || memo.created_at;
        memo.savedVersion = version;
        if (memo.version === version) {
            memo.updated_at = saved.updated_at || memo.updated_at;
            setStatus("저장됨 " + clock(), "saved");
        } else if (!saveTimers[memo.id]) {
            // 저장하는 사이 더 고쳤다. 새 base 로 곧 한 번 더 저장한다.
            saveTimers[memo.id] = setTimeout(function () { queueSave(memo.id); }, 80);
        }
        writeDrafts();
        memos = sortMemos(memos);
        renderList();
    }

    // 409: 서버 쪽이 더 새롭다. 서버 것은 원래 id 로 보여 주고, 내 변경은 새 id 사본으로 저장한다.
    function resolveConflict(memo, stored) {
        if (saveTimers[memo.id]) {
            clearTimeout(saveTimers[memo.id]);
            delete saveTimers[memo.id];
        }
        if (pendingDeletes[memo.id]) {
            // 저장이 가는 동안 이 메모를 지웠다. 지운 글로 사본을 만들지 않고, 다른 곳의 새 편집도 지우지 않는다.
            keepChangedElsewhere(memo.id, stored);
            return;
        }
        var copy = localMemo({
            id: makeId(),
            title: conflictTitle(memo.title),
            text: memo.text,
            created_at: nowIso()
        });
        var index = memos.indexOf(memo);
        if (index !== -1) memos.splice(index, 1);
        else index = 0;
        if (stored.deleted !== true && !findMemo(asText(stored.id))) {
            memos.splice(index, 0, fromServer(stored));
        }
        memos.splice(index, 0, copy);
        if (selectedId === memo.id) selectedId = copy.id;
        markEdited(copy);
        queueSave(copy.id);
        showNotice(
            (stored.deleted === true
                ? "다른 곳에서 삭제된 메모였습니다. "
                : "다른 곳에서 먼저 바뀐 메모였습니다. 원래 메모에는 그쪽 내용을 두고, ")
            + "내 변경은 '" + copy.title + "' 로 따로 저장합니다."
        );
        writeUiState();
        renderAll();
    }

    function merge(serverList) {
        var seen = Object.create(null);
        var next = [];
        serverList.forEach(function (raw) {
            if (!raw || typeof raw !== "object" || raw.deleted === true) return;
            var id = asText(raw.id);
            if (!id || seen[id]) return;
            seen[id] = true;
            if (pendingDeletes[id] && !pendingDeletes[id].cancelled) return;   // 곧 보낼 삭제가 있다
            var local = findMemo(id);
            // 저장 전 편집은 덮지 않는다. 서버도 바뀌었다면 저장 때 409 로 갈라진다.
            next.push(local && isDirty(local) ? local : fromServer(raw));
        });
        memos.forEach(function (local) {
            if (!seen[local.id] && isDirty(local)) next.push(local);   // 아직 서버에 없는 새 메모
        });
        memos = sortMemos(next);
        if (selectedId && !findMemo(selectedId)) {
            selectedId = memos.length ? memos[0].id : null;
        }
    }

    async function loadFromServer() {
        var response = await request("GET", API_PATH);
        if (response.status === 404) {
            markMissingRoutes();
            return;
        }
        if (!response.ok) throw await responseError(response);
        var payload = (await readJson(response)) || {};
        available = true;
        merge(Array.isArray(payload.memos) ? payload.memos : []);
        // 불러올 때마다 본다 — 첫 불러오기가 실패했거나, 그때는 아직 열려 있던 탭이 그 뒤 닫혔을 수 있다.
        var recovered = recoverDrafts();
        if (!selectedId && memos.length) selectedId = memos[0].id;
        renderAll();
        var pending = memos.filter(function (memo) {
            return isDirty(memo) && !saveTimers[memo.id];
        });
        if (recovered.length) {
            setStatus("닫기 전 메모 변경 " + recovered.length + "개 복구 · 저장 중…", "pending");
        } else if (!pending.length) {
            setStatus("메모 " + memos.length + "개", "saved");
        }
        // 실패했던 저장과 복구한 초안을 다시 보낸다.
        pending.forEach(function (memo) { queueSave(memo.id); });
    }

    function refresh() {
        if (refreshPromise) return refreshPromise;
        lastRefreshAt = Date.now();
        refreshPromise = enqueue(loadFromServer).catch(function (error) {
            console.error("[SAM3 Notebook] memo load failed:", error);
            setStatus("메모 불러오기 실패: " + errorText(error), "error");
        }).then(function () {
            refreshPromise = null;
        });
        return refreshPromise;
    }

    function maybeRefresh() {
        if (!ui.section || !ui.section.isConnected || !ui.panel) return;
        if (tab !== "memo" || !ui.panel.open) return;
        if (Date.now() - lastRefreshAt < FOCUS_REFRESH_GAP_MS) return;
        refresh();
    }

    // ---- 동작 -------------------------------------------------------------

    function select(id) {
        if (!findMemo(id)) return;
        selectedId = id;
        writeUiState();
        renderList();
        renderEditor();
    }

    function createMemo() {
        if (available === false) {
            markMissingRoutes();
            return;
        }
        var memo = localMemo({ id: makeId(), created_at: nowIso() });
        memos.unshift(memo);
        selectedId = memo.id;
        markEdited(memo);
        writeUiState();
        renderAll();
        queueSave(memo.id);
        if (ui.title) ui.title.focus();
    }

    function deleteSelected() {
        var memo = findMemo(selectedId);
        if (!memo) return;
        if (!window.confirm("'" + displayTitle(memo) + "' 메모를 삭제할까요?")) return;
        if (saveTimers[memo.id]) {
            clearTimeout(saveTimers[memo.id]);
            delete saveTimers[memo.id];
        }
        var index = memos.indexOf(memo);
        memos.splice(index, 1);
        var next = memos[index] || memos[index - 1] || null;
        selectedId = next ? next.id : null;
        var pending = { cancelled: false };
        pendingDeletes[memo.id] = pending;
        writeDrafts();
        writeUiState();
        renderAll();
        setStatus("삭제 중…", "pending");
        // 앞선 저장이 끝난 뒤에 지운다. 서버에 한 번도 저장되지 않았으면 404 — 그래도 지워진 것이다.
        return enqueue(async function () {
            try {
                await sendDelete(memo, pending);
            } finally {
                if (pendingDeletes[memo.id] === pending) delete pendingDeletes[memo.id];
            }
        }).catch(function (error) {
            console.error("[SAM3 Notebook] memo delete failed:", error);
            setStatus("메모 삭제 실패: " + errorText(error), "error");
            refresh();   // 서버에 남아 있으면 목록에 다시 보인다
        });
    }

    async function sendDelete(memo, pending) {
        if (pending.cancelled) return;
        // 지우기로 한 판(마지막으로 본 서버 시각)을 붙인다 — 그 뒤 다른 곳에서 고쳤으면 409 로 지우지 않는다.
        // 앞선 저장이 성공했으면 memo.base 는 이미 그 저장의 시각이다.
        var url = itemUrl(memo.id);
        if (memo.base) url += "?base_updated_at=" + encodeURIComponent(memo.base);
        var response = await request("DELETE", url);
        if (response.status === 409) {
            var payload = await readJson(response);
            if (payload && payload.memo && typeof payload.memo === "object") {
                keepChangedElsewhere(memo.id, payload.memo);
                return;
            }
            throw new Error(payload && payload.detail ? String(payload.detail) : "HTTP 409");
        }
        if (!response.ok && response.status !== 404) throw await responseError(response);
        setStatus("삭제됨", "saved");
    }

    // 지우려던 메모를 다른 곳에서 먼저 고쳤다 — 지우지 않고 그쪽 내용을 목록에 되돌린다(지운 쪽 편집만 버린다).
    function keepChangedElsewhere(id, stored) {
        if (pendingDeletes[id]) pendingDeletes[id].cancelled = true;
        if (stored.deleted === true) {   // 그쪽에서도 지웠다
            setStatus("삭제됨", "saved");
            return;
        }
        var restored = fromServer(Object.assign({}, stored, { id: id }));
        var existing = findMemo(id);
        if (!existing) memos.push(restored);
        else if (!isDirty(existing)) memos[memos.indexOf(existing)] = restored;
        memos = sortMemos(memos);
        if (!selectedId) selectedId = id;
        writeDrafts();
        writeUiState();
        showNotice(
            "'" + displayTitle(findMemo(id)) + "' 메모는 다른 곳에서 먼저 바뀌어 지우지 않았습니다. "
            + "바뀐 내용을 다시 보여 줍니다 — 그래도 지우려면 한 번 더 삭제하세요."
        );
        setStatus("삭제하지 않음", "warning");
        renderAll();
    }

    function onTitleInput() {
        var memo = findMemo(selectedId);
        if (!memo) return;
        memo.title = clip(ui.title.value, MAX_TITLE_LENGTH);
        scheduleSave(memo);
        syncListItem(memo);
    }

    function onTextInput() {
        var memo = findMemo(selectedId);
        if (!memo) return;
        memo.text = clip(ui.text.value, MAX_TEXT_LENGTH);
        scheduleSave(memo);
        if (!memo.title.trim()) syncListItem(memo);
    }

    function onEditorKeydown(event) {
        var key = event.key;
        var modifier = event.ctrlKey || event.metaKey;
        if (modifier && !event.altKey && (key === "s" || key === "S")) {
            event.preventDefault();
            event.stopPropagation();
            if (selectedId) queueSave(selectedId);
            return;
        }
        // Forge 전역 단축키(Ctrl+Enter 생성, Alt+Enter 건너뛰기, Esc 중단)가 메모 입력에서 불리지 않게 한다.
        if (key === "Escape" || (key === "Enter" && (modifier || event.altKey))) {
            event.stopPropagation();
            return;
        }
        if (key === "Enter" && event.target === ui.title && !event.isComposing) {
            event.preventDefault();
            ui.text.focus();
        }
    }

    function setTab(next, persist) {
        tab = next === "memo" ? "memo" : "presets";
        if (ui.panel) ui.panel.setAttribute("data-sam3-notebook-tab", tab);
        if (ui.tabs) {
            Array.prototype.forEach.call(
                ui.tabs.querySelectorAll("[data-memo-tab]"),
                function (button) {
                    var active = button.getAttribute("data-memo-tab") === tab;
                    button.setAttribute("aria-selected", active ? "true" : "false");
                    button.tabIndex = active ? 0 : -1;
                }
            );
        }
        if (persist) writeUiState();
        if (tab === "memo") maybeRefresh();
    }

    function showNotice(message) {
        if (!ui.notice) return;
        ui.noticeText.textContent = message;
        ui.notice.hidden = false;
    }

    function setStatus(message, tone) {
        if (!ui.status) return;
        ui.status.textContent = message || "";
        ui.status.setAttribute("data-tone", tone || "normal");
    }

    // ---- 그리기 -----------------------------------------------------------

    function setFieldValue(field, value) {
        if (field.value === value) return;
        var focused = document.activeElement === field;
        var start = field.selectionStart;
        var end = field.selectionEnd;
        field.value = value;
        if (focused && typeof start === "number") {
            try {
                field.setSelectionRange(
                    Math.min(start, value.length),
                    Math.min(end, value.length)
                );
            } catch (error) {}
        }
    }

    function itemMeta(memo) {
        return (isDirty(memo) ? "저장 대기 · " : "") + formatTime(memo.updated_at);
    }

    function syncListItem(memo) {
        if (!ui.list) return;
        var items = ui.list.querySelectorAll("[data-memo-id]");
        for (var index = 0; index < items.length; index++) {
            if (items[index].getAttribute("data-memo-id") !== memo.id) continue;
            items[index].querySelector(".sam3-memo-item-title").textContent = displayTitle(memo);
            items[index].querySelector(".sam3-memo-item-meta").textContent = itemMeta(memo);
            return;
        }
    }

    function renderList() {
        var list = ui.list;
        if (!list) return;
        list.textContent = "";
        if (!memos.length) {
            var empty = document.createElement("p");
            empty.className = "sam3-memo-empty";
            empty.textContent = available === false
                ? "메모 저장소가 없는 Forge 입니다. 확장을 업데이트한 뒤 Forge 를 다시 시작하세요."
                : "메모가 없습니다. ＋ 새 메모로 시작하세요.";
            list.appendChild(empty);
            return;
        }
        memos.forEach(function (memo) {
            var item = document.createElement("button");
            item.type = "button";
            item.className = "sam3-memo-item";
            item.setAttribute("role", "option");
            item.setAttribute("data-memo-id", memo.id);
            item.setAttribute("aria-selected", memo.id === selectedId ? "true" : "false");
            var title = document.createElement("span");
            title.className = "sam3-memo-item-title";
            title.textContent = displayTitle(memo);
            var meta = document.createElement("span");
            meta.className = "sam3-memo-item-meta";
            meta.textContent = itemMeta(memo);
            item.appendChild(title);
            item.appendChild(meta);
            list.appendChild(item);
        });
    }

    function renderEditor() {
        if (!ui.title) return;
        var memo = findMemo(selectedId);
        ui.title.disabled = !memo;
        ui.text.disabled = !memo;
        ui.remove.disabled = !memo;
        ui.create.disabled = available === false;
        setFieldValue(ui.title, memo ? memo.title : "");
        setFieldValue(ui.text, memo ? memo.text : "");
        ui.text.placeholder = memo
            ? "메모를 입력하세요 — 자동으로 저장됩니다 (Ctrl+S 바로 저장)"
            : "목록에서 메모를 고르거나 ＋ 새 메모를 누르세요";
    }

    function renderAll() {
        renderList();
        renderEditor();
    }

    // ---- 붙이기 -----------------------------------------------------------

    var STYLE = [
        "#sam3_notebook_panel > .sam3-notebook-tabs {",
        "    display: flex; gap: .3rem; padding: 0 .5rem .45rem;",
        "}",
        "#sam3_notebook_panel > .sam3-notebook-tabs > button[aria-selected=\"true\"] {",
        "    border-color: var(--color-accent, var(--sam3-color-accent));",
        "    color: var(--color-accent, var(--sam3-color-accent));",
        "    font-weight: 600;",
        "}",
        "#sam3_notebook_panel[data-sam3-notebook-tab=\"memo\"] > .sam3-notebook-head-actions,",
        "#sam3_notebook_panel[data-sam3-notebook-tab=\"memo\"] > .sam3-notebook-toolbar,",
        "#sam3_notebook_panel[data-sam3-notebook-tab=\"memo\"] > .sam3-notebook-presets {",
        "    display: none !important;",
        "}",
        "#sam3_notebook_panel:not([data-sam3-notebook-tab=\"memo\"]) > .sam3-memo {",
        "    display: none !important;",
        "}",
        ".sam3-memo { display: grid; gap: .45rem; min-width: 0; padding: 0 .5rem .55rem; }",
        ".sam3-memo-bar { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; }",
        ".sam3-memo-status {",
        "    flex: 1 1 6rem; min-width: 0; overflow: hidden;",
        "    color: var(--body-text-color-subdued, var(--sam3-color-muted));",
        "    font-size: .76rem; text-align: right; text-overflow: ellipsis; white-space: nowrap;",
        "}",
        ".sam3-memo-status[data-tone=\"saved\"] { color: var(--color-accent, var(--sam3-color-accent)); }",
        ".sam3-memo-status[data-tone=\"warning\"],",
        ".sam3-memo-status[data-tone=\"error\"] { color: var(--error-text-color, var(--sam3-color-error-hover)); }",
        ".sam3-memo-notice {",
        "    display: flex; align-items: flex-start; gap: .4rem; padding: .45rem .55rem;",
        "    border: 1px solid var(--color-accent, var(--sam3-color-accent)); border-radius: .45rem;",
        "    background: var(--background-fill-secondary, var(--sam3-color-paper-2));",
        "    font-size: .8rem; line-height: 1.4;",
        "}",
        ".sam3-memo-notice[hidden] { display: none; }",
        ".sam3-memo-notice > span {",
        "    flex: 1 1 auto; min-width: 0; word-break: keep-all; overflow-wrap: anywhere;",
        "}",
        ".sam3-memo-body { display: flex; flex-wrap: wrap; align-items: stretch; gap: .45rem; min-width: 0; }",
        ".sam3-memo-list {",
        "    display: grid; flex: 1 1 9rem; align-content: start; gap: .25rem; min-width: 0;",
        "    max-height: min(28rem, 50vh); overflow-y: auto; scrollbar-width: thin;",
        "}",
        "#sam3_notebook_panel .sam3-memo-item {",
        "    display: grid; gap: .1rem; width: 100%; min-width: 0; text-align: left;",
        "}",
        "#sam3_notebook_panel .sam3-memo-item[aria-selected=\"true\"] {",
        "    border-color: var(--color-accent, var(--sam3-color-accent));",
        "    background: var(--background-fill-secondary, var(--sam3-color-paper-2));",
        "}",
        ".sam3-memo-item-title { overflow: hidden; font-weight: 600; text-overflow: ellipsis; white-space: nowrap; }",
        ".sam3-memo-item-meta {",
        "    color: var(--body-text-color-subdued, var(--sam3-color-muted)); font-size: .72rem;",
        "}",
        ".sam3-memo-empty {",
        "    margin: 0; padding: .8rem; border: 1px dashed var(--border-color-primary, var(--sam3-color-rule));",
        "    border-radius: .45rem; color: var(--body-text-color-subdued, var(--sam3-color-muted));",
        "    font-size: .82rem; text-align: center;",
        "}",
        ".sam3-memo-editor {",
        "    display: grid; flex: 3 1 16rem; grid-template-rows: auto 1fr; gap: .4rem; min-width: 0;",
        "}",
        ".sam3-memo-editor > input, .sam3-memo-editor > textarea {",
        "    box-sizing: border-box; width: 100%; min-width: 0; padding: .4rem .48rem;",
        "    border: 1px solid var(--border-color-primary, var(--sam3-color-rule)); border-radius: .4rem;",
        "    color: inherit; background: var(--input-background-fill, var(--sam3-color-paper-3)); font: inherit;",
        "}",
        ".sam3-memo-editor > input { font-weight: 600; }",
        ".sam3-memo-editor > textarea { min-height: 16rem; resize: vertical; line-height: 1.5; }",
        ".sam3-memo-editor > :disabled, #sam3_notebook_panel .sam3-memo button:disabled {",
        "    opacity: .55; cursor: not-allowed;",
        "}"
    ].join("\n");

    function injectStyle() {
        if (document.getElementById(STYLE_ID)) return;
        var style = document.createElement("style");
        style.id = STYLE_ID;
        style.textContent = STYLE;
        (document.head || document.documentElement).appendChild(style);
    }

    function build() {
        var tabs = document.createElement("div");
        tabs.className = "sam3-notebook-tabs";
        tabs.setAttribute("role", "tablist");
        tabs.setAttribute("aria-label", "Notebook 보기");
        tabs.setAttribute("data-sam3-memo-tabs", "");
        tabs.innerHTML = [
            '<button type="button" role="tab" data-memo-tab="presets">프리셋</button>',
            '<button type="button" role="tab" data-memo-tab="memo">메모</button>'
        ].join("");

        var section = document.createElement("section");
        section.className = "sam3-memo";
        section.setAttribute("data-sam3-memo", "");
        section.setAttribute("role", "tabpanel");
        section.setAttribute("aria-label", "메모");
        section.innerHTML = [
            '<div class="sam3-memo-bar">',
            ' <button type="button" data-memo-new title="새 메모">＋ 새 메모</button>',
            ' <button type="button" class="danger" data-memo-delete title="선택한 메모 삭제">삭제</button>',
            ' <button type="button" data-memo-refresh title="다시 불러오기" aria-label="다시 불러오기">↻</button>',
            ' <span class="sam3-memo-status" data-memo-status aria-live="polite"></span>',
            '</div>',
            '<div class="sam3-memo-notice" data-memo-notice role="status" hidden>',
            ' <span data-memo-notice-text></span>',
            ' <button type="button" data-memo-notice-close title="알림 닫기" aria-label="알림 닫기">×</button>',
            '</div>',
            '<div class="sam3-memo-body">',
            ' <div class="sam3-memo-list" data-memo-list role="listbox" aria-label="메모 목록"></div>',
            ' <div class="sam3-memo-editor">',
            '  <input type="text" data-memo-title maxlength="' + MAX_TITLE_LENGTH + '"',
            '   placeholder="제목" aria-label="메모 제목" autocomplete="off" spellcheck="false">',
            '  <textarea data-memo-text maxlength="' + MAX_TEXT_LENGTH + '"',
            '   aria-label="메모 내용" spellcheck="false"></textarea>',
            ' </div>',
            '</div>'
        ].join("");

        ui.tabs = tabs;
        ui.section = section;
        ui.list = section.querySelector("[data-memo-list]");
        ui.title = section.querySelector("[data-memo-title]");
        ui.text = section.querySelector("[data-memo-text]");
        ui.status = section.querySelector("[data-memo-status]");
        ui.notice = section.querySelector("[data-memo-notice]");
        ui.noticeText = section.querySelector("[data-memo-notice-text]");
        ui.create = section.querySelector("[data-memo-new]");
        ui.remove = section.querySelector("[data-memo-delete]");

        tabs.addEventListener("click", function (event) {
            var button = event.target.closest && event.target.closest("[data-memo-tab]");
            if (!button) return;
            event.preventDefault();
            setTab(button.getAttribute("data-memo-tab"), true);
        });
        ui.create.addEventListener("click", createMemo);
        ui.remove.addEventListener("click", deleteSelected);
        section.querySelector("[data-memo-refresh]").addEventListener("click", function () {
            refresh();
        });
        section.querySelector("[data-memo-notice-close]").addEventListener("click", function () {
            ui.notice.hidden = true;
        });
        ui.list.addEventListener("click", function (event) {
            var item = event.target.closest && event.target.closest("[data-memo-id]");
            if (item) select(item.getAttribute("data-memo-id"));
        });
        ui.title.addEventListener("input", onTitleInput);
        ui.text.addEventListener("input", onTextInput);
        ui.title.addEventListener("keydown", onEditorKeydown);
        ui.text.addEventListener("keydown", onEditorKeydown);
    }

    function findPanel() {
        return app().querySelector("#sam3_notebook_panel")
            || document.querySelector("#sam3_notebook_panel");
    }

    function ensureMounted() {
        var panel = findPanel();
        if (!panel) return false;
        if (ui.panel === panel && ui.section && ui.section.parentElement === panel
                && ui.tabs && ui.tabs.parentElement === panel) {
            return true;
        }
        injectStyle();
        Array.prototype.slice.call(panel.children).forEach(function (child) {
            if (child.hasAttribute("data-sam3-memo-tabs") || child.hasAttribute("data-sam3-memo")) {
                panel.removeChild(child);
            }
        });
        build();
        var summary = null;
        for (var index = 0; index < panel.children.length; index++) {
            if (panel.children[index].tagName === "SUMMARY") {
                summary = panel.children[index];
                break;
            }
        }
        panel.insertBefore(ui.tabs, summary ? summary.nextSibling : panel.firstChild);
        panel.appendChild(ui.section);
        if (ui.panel !== panel) {
            panel.addEventListener("toggle", maybeRefresh);
        }
        ui.panel = panel;
        if (!uiStateRead) {
            uiStateRead = true;
            var saved = readJsonKey(UI_KEY, {});
            tab = saved.tab === "memo" ? "memo" : "presets";
            if (!selectedId && typeof saved.selectedId === "string") selectedId = saved.selectedId;
        }
        setTab(tab, false);
        renderAll();
        if (available === null && !refreshPromise) refresh();
        return true;
    }

    function boot() {
        ensureMounted();
        // Gradio 가 패널을 새로 만들거나 notebook.js 가 늦게 붙여도 다시 찾는다.
        if (!mountTimer) mountTimer = setInterval(ensureMounted, MOUNT_POLL_MS);
    }

    window.addEventListener("sam3:notebook-mounted", function () { ensureMounted(); });
    window.addEventListener("focus", maybeRefresh);
    document.addEventListener("visibilitychange", function () {
        if (heartbeatTimer) writeDrafts();   // 숨은 탭은 더 오래 기다려 준다(draftOwnerGone)
        if (!document.hidden) maybeRefresh();
    });
    // 닫히거나 새로 고쳐진다 — 남은 초안은 다음에 열린 탭이 바로 이어받아도 된다.
    window.addEventListener("pagehide", markDraftsClosed);
    window.addEventListener("pageshow", function (event) {
        if (event.persisted && heartbeatTimer) writeDrafts();   // bfcache 에서 돌아왔다 — 다시 열려 있다
    });

    // 테스트 전용 입구. 실제 페이지는 이 객체를 만들지 않는다.
    if (window.__sam3NotebookMemoTestHooks
            && typeof window.__sam3NotebookMemoTestHooks === "object") {
        var hooks = window.__sam3NotebookMemoTestHooks;
        hooks.mount = ensureMounted;
        hooks.refresh = refresh;
        hooks.memos = function () { return memos; };
        hooks.selectedId = function () { return selectedId; };
        // 큐가 빌 때까지 기다린다(작업이 새 작업을 넣어도).
        hooks.idle = async function () {
            var seen;
            do {
                seen = queue;
                await seen;
            } while (seen !== queue);
        };
        // 디바운스를 기다리지 않고 대기 중인 저장을 바로 보낸다.
        hooks.flush = function () {
            Object.keys(saveTimers).forEach(function (id) { queueSave(id); });
            return hooks.idle();
        };
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () { setTimeout(boot, 300); });
    } else {
        document.addEventListener("DOMContentLoaded", function () {
            setTimeout(boot, 1300);
        });
    }
})();
