const hasPersistentStorage = (() => {
  try {
    const key = "__localagent_storage_probe__";
    window.localStorage.setItem(key, "1");
    window.localStorage.removeItem(key);
    return true;
  } catch (_) {
    return false;
  }
})();

function storageGet(key, fallback = "") {
  if (!hasPersistentStorage) return fallback;
  try {
    return window.localStorage.getItem(key) || fallback;
  } catch (_) {
    return fallback;
  }
}

function storageSet(key, value) {
  if (!hasPersistentStorage) return;
  try {
    window.localStorage.setItem(key, value);
  } catch (_) {}
}

function isDesignPreview() {
  if (window.self === window.top) return false;
  const path = window.location.pathname || "";
  return (
    !hasPersistentStorage ||
    window.location.href === "about:srcdoc" ||
    path.includes("/api/projects/") ||
    path.includes("/raw/")
  );
}

const state = {
  user: null,
  sessions: [],
  messages: [],
  activeSessionId: null,
  files: { uploads: [] },
  pendingAttachments: [],
  openSessionMenuId: null,
  theme: storageGet("localagent_theme", "dark"),
  sidebarCollapsed: storageGet("localagent_sidebar_collapsed") === "1",
  runningPollTimer: null,
  settingsChatUserId: null,
  busy: false,
  voiceRecorder: null,
  voiceStream: null,
  voiceRecording: false,
  streamingResponse: false,
  autoFollowStream: true,
  voiceSupported: typeof navigator !== "undefined"
    && Boolean(navigator.mediaDevices?.getUserMedia)
    && typeof window !== "undefined"
    && Boolean(window.AudioContext || window.webkitAudioContext),
};

const $ = (id) => document.getElementById(id);
const MESSAGE_BOTTOM_STICKINESS_PX = 96;
const MAX_RENDERED_MESSAGE_CHARS = 120000;
const MAX_TEXT_PREVIEW_CHARS = 200000;
const MAX_SSE_BUFFER_CHARS = 1000000;
const MAX_PENDING_TRACE_EVENTS = 500;
const MAX_TRACE_FIELD_DISPLAY_CHARS = 800;
const MAX_CLIENT_MESSAGES = 100;
const MAX_UPLOAD_BATCH_FILES = 10;
const MAX_UPLOAD_FILE_BYTES = 25 * 1024 * 1024;
const MAX_VOICE_RECORDING_SECONDS = 75;
const STREAM_RENDER_INTERVAL_MS = 100;
const STREAM_IDLE_TIMEOUT_MS = 180000;
const STREAM_TOTAL_TIMEOUT_MS = 600000;
const TRUNCATED_OUTPUT_NOTICE = "\n\n[Output truncated in the browser because it exceeded the display safety limit.]";
const TRUNCATED_PREVIEW_NOTICE = "\n\n[Preview truncated in the browser because it exceeded the display safety limit.]";
const TRUNCATED_TRACE_NOTICE = "\n[Trace field truncated.]";

function isMessagesNearBottom() {
  const el = $("messages");
  if (!el) return true;
  return el.scrollHeight - el.scrollTop - el.clientHeight <= MESSAGE_BOTTOM_STICKINESS_PX;
}

function scrollMessagesToBottom() {
  const el = $("messages");
  if (!el) return;
  el.scrollTop = el.scrollHeight;
}

function maybeScrollMessagesToBottom(shouldStick) {
  if (shouldStick) scrollMessagesToBottom();
}

function shouldFollowMessagesBottom() {
  return (!state.streamingResponse || state.autoFollowStream) && isMessagesNearBottom();
}

function stopStreamingAutoFollow() {
  if (!state.streamingResponse) return;
  state.autoFollowStream = false;
}

function displayMessageContent(content) {
  const text = String(content || "");
  if (text.length <= MAX_RENDERED_MESSAGE_CHARS) return text;
  return text.slice(0, MAX_RENDERED_MESSAGE_CHARS - TRUNCATED_OUTPUT_NOTICE.length) + TRUNCATED_OUTPUT_NOTICE;
}

function displayPreviewContent(content) {
  const text = String(content || "");
  if (text.length <= MAX_TEXT_PREVIEW_CHARS) return text;
  return text.slice(0, MAX_TEXT_PREVIEW_CHARS - TRUNCATED_PREVIEW_NOTICE.length) + TRUNCATED_PREVIEW_NOTICE;
}

function truncateUiText(value, limit, notice = TRUNCATED_TRACE_NOTICE) {
  const text = String(value || "");
  if (text.length <= limit) return text;
  return text.slice(0, Math.max(0, limit - notice.length)).trimEnd() + notice;
}

function capTraceEvents(events) {
  if (!Array.isArray(events) || events.length <= MAX_PENDING_TRACE_EVENTS) {
    return events || [];
  }
  return events.slice(-MAX_PENDING_TRACE_EVENTS);
}

function sanitizeTraceEventForUi(event) {
  const source = event || {};
  return {
    ts: source.ts || "",
    kind: truncateUiText(source.kind || "status", 80, ""),
    label: truncateUiText(source.label || "agent", 120, ""),
    tool_name: truncateUiText(source.tool_name || "", 160, ""),
    tool_call_id: truncateUiText(source.tool_call_id || "", 160, ""),
    args: truncateUiText(source.args || "", MAX_TRACE_FIELD_DISPLAY_CHARS),
    output: truncateUiText(source.output || "", MAX_TRACE_FIELD_DISPLAY_CHARS),
  };
}

function sanitizeTurnLogForUi(log) {
  const source = log || {};
  return {
    ...source,
    objective: truncateUiText(source.objective || "", 2000, TRUNCATED_OUTPUT_NOTICE),
    summary: truncateUiText(source.summary || "", 2000, TRUNCATED_OUTPUT_NOTICE),
    error: truncateUiText(source.error || "", 2000, TRUNCATED_OUTPUT_NOTICE),
  };
}

function sanitizeMessageForUi(message) {
  const metadata = message?.metadata ? { ...message.metadata } : undefined;
  if (metadata?.trace_events) {
    metadata.trace_events = capTraceEvents(metadata.trace_events.map(sanitizeTraceEventForUi));
  }
  if (metadata?.turn_logs) {
    metadata.turn_logs = metadata.turn_logs.slice(0, 12).map(sanitizeTurnLogForUi);
  }
  return {
    ...message,
    content: displayMessageContent(message?.content),
    metadata,
  };
}

function sanitizeMessagesForUi(messages) {
  return (messages || []).slice(-MAX_CLIENT_MESSAGES).map(sanitizeMessageForUi);
}

function stableHash(value) {
  const text = String(value || "");
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function csrfToken() {
  const match = document.cookie
    .split("; ")
    .find((part) => part.startsWith("localagent_csrf="));
  return match ? decodeURIComponent(match.split("=")[1]) : "";
}

async function api(path, options = {}) {
  const isFormData = options.body instanceof FormData;
  const headers = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...(options.headers || {}),
  };
  if (!["GET", "HEAD"].includes((options.method || "GET").toUpperCase())) {
    headers["X-CSRF-Token"] = csrfToken();
  }
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers,
  });
  if (response.status === 401 && !options.noAuthRedirect) {
    showLogin();
    throw new Error("Not authenticated");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg || String(item)).join(" ")
      : data.detail;
    const error = new Error(detail || data.error || "Request failed");
    error.status = response.status;
    throw error;
  }
  return data;
}

async function streamApi(path, payload, onEvent) {
  const controller = new AbortController();
  let idleTimer = null;
  const totalTimer = window.setTimeout(() => controller.abort(), STREAM_TOTAL_TIMEOUT_MS);
  const resetIdleTimer = () => {
    if (idleTimer) window.clearTimeout(idleTimer);
    idleTimer = window.setTimeout(() => controller.abort(), STREAM_IDLE_TIMEOUT_MS);
  };
  const clearStreamTimers = () => {
    window.clearTimeout(totalTimer);
    if (idleTimer) window.clearTimeout(idleTimer);
  };
  resetIdleTimer();
  let response = null;
  try {
    response = await fetch(path, {
      method: "POST",
      credentials: "same-origin",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrfToken(),
      },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    clearStreamTimers();
    if (error.name === "AbortError") throw new Error("Agent stream timed out.");
    throw error;
  }
  if (response.status === 401) {
    clearStreamTimers();
    showLogin();
    throw new Error("Not authenticated");
  }
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => ({}));
    clearStreamTimers();
    throw new Error(data.detail || data.error || "Request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalData = null;

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      resetIdleTimer();
      buffer += decoder.decode(value, { stream: true });
      if (buffer.length > MAX_SSE_BUFFER_CHARS) {
        throw new Error("Streaming response exceeded the browser safety limit.");
      }
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        const line = part.split("\n").find((item) => item.startsWith("data: "));
        if (!line) continue;
        const event = JSON.parse(line.slice(6));
        if (event.type === "trace") {
          onEvent(event.event);
        } else if (event.type === "text_delta") {
          onEvent({ kind: "answer_delta", content: event.content || "" });
        } else if (event.type === "replace") {
          onEvent({ kind: "answer_replace", content: event.content || "" });
        } else if (event.type === "done") {
          finalData = event.data;
        } else if (event.type === "error") {
          throw new Error(event.error || "Request failed");
        }
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Agent stream timed out.");
    }
    throw error;
  } finally {
    clearStreamTimers();
    await reader.cancel().catch(() => {});
  }

  if (!finalData) {
    return null;
  }
  return finalData;
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeAttr(text) {
  return escapeHtml(text).replace(/"/g, "&quot;");
}

function formatMessage(content) {
  const lines = String(content || "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let index = 0;

  const readParagraph = () => {
    const paragraph = [];
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim() || isBlockStart(line, index, lines)) break;
      paragraph.push(line);
      index += 1;
    }
    return `<p>${paragraph.map(formatInlineMarkdown).join("<br>")}</p>`;
  };

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = line.match(/^```([\w-]*)\s*$/);
    if (fence) {
      index += 1;
      const code = [];
      while (index < lines.length && !/^```\s*$/.test(lines[index])) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      const language = fence[1] ? ` data-language="${escapeAttr(fence[1])}"` : "";
      blocks.push(`<pre${language}><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*$/);
    if (heading) {
      const level = Math.min(heading[1].length, 4);
      blocks.push(`<h${level}>${formatInlineMarkdown(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
      blocks.push("<hr>");
      index += 1;
      continue;
    }

    if (isTableStart(index, lines)) {
      const table = readTable(index, lines);
      blocks.push(renderTable(table));
      index = table.nextIndex;
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*[-*]\s+/, ""));
        index += 1;
      }
      blocks.push(`<ul>${items.map((item) => `<li>${formatInlineMarkdown(item)}</li>`).join("")}</ul>`);
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      const firstNumber = Number(line.match(/^\s*(\d+)\.\s+/)?.[1] || 1);
      const start = Number.isFinite(firstNumber) && firstNumber !== 1 ? ` start="${firstNumber}"` : "";
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*\d+\.\s+/, ""));
        index += 1;
      }
      blocks.push(`<ol${start}>${items.map((item) => `<li>${formatInlineMarkdown(item)}</li>`).join("")}</ol>`);
      continue;
    }

    blocks.push(readParagraph());
  }

  return linkifyVirtualPaths(blocks.join(""));
}

function isBlockStart(line, index = -1, lines = []) {
  return (
    /^```/.test(line) ||
    /^(#{1,6})\s+/.test(line) ||
    /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line) ||
    isTableStart(index, lines) ||
    /^\s*[-*]\s+/.test(line) ||
    /^\s*\d+\.\s+/.test(line)
  );
}

function isTableStart(index, lines) {
  if (index < 0 || index + 1 >= lines.length) return false;
  const header = parseTableRow(lines[index]);
  const delimiter = parseTableDelimiter(lines[index + 1]);
  return Boolean(header && delimiter && header.length > 1 && delimiter.length === header.length);
}

function readTable(startIndex, lines) {
  const headers = parseTableRow(lines[startIndex]);
  const alignments = parseTableDelimiter(lines[startIndex + 1]);
  const rows = [];
  let nextIndex = startIndex + 2;

  while (nextIndex < lines.length) {
    const line = lines[nextIndex];
    if (!line.trim()) break;
    const row = parseTableRow(line);
    if (!row || row.length < 2) break;
    rows.push(normalizeTableRow(row, headers.length));
    nextIndex += 1;
  }

  return {
    headers: normalizeTableRow(headers, headers.length),
    alignments,
    rows,
    nextIndex,
  };
}

function renderTable(table) {
  const renderCell = (tag, cell, alignment) => {
    const align = alignment ? ` style="text-align: ${alignment}"` : "";
    return `<${tag}${align}>${formatInlineMarkdown(cell)}</${tag}>`;
  };
  const head = table.headers
    .map((cell, cellIndex) => renderCell("th", cell, table.alignments[cellIndex]))
    .join("");
  const body = table.rows
    .map((row) => {
      const cells = table.headers
        .map((_, cellIndex) => renderCell("td", row[cellIndex] || "", table.alignments[cellIndex]))
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  return `<div class="markdown-table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function parseTableRow(line) {
  const trimmed = line.trim();
  if (!trimmed.includes("|")) return null;

  let source = trimmed;
  if (source.startsWith("|")) source = source.slice(1);
  if (source.endsWith("|") && !source.endsWith("\\|")) source = source.slice(0, -1);

  return splitTableCells(source).map((cell) => cell.trim().replace(/\\\|/g, "|"));
}

function parseTableDelimiter(line) {
  const cells = parseTableRow(line);
  if (!cells || cells.length < 2) return null;
  const alignments = [];

  for (const cell of cells) {
    if (!/^:?-{3,}:?$/.test(cell.trim())) return null;
    const trimmed = cell.trim();
    if (trimmed.startsWith(":") && trimmed.endsWith(":")) {
      alignments.push("center");
    } else if (trimmed.endsWith(":")) {
      alignments.push("right");
    } else {
      alignments.push("");
    }
  }

  return alignments;
}

function splitTableCells(source) {
  const cells = [];
  let cell = "";
  let inCode = false;

  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    const previous = source[index - 1];
    if (char === "`" && previous !== "\\") {
      inCode = !inCode;
    }
    if (char === "|" && previous !== "\\" && !inCode) {
      cells.push(cell);
      cell = "";
    } else {
      cell += char;
    }
  }
  cells.push(cell);
  return cells;
}

function normalizeTableRow(row, width) {
  return Array.from({ length: width }, (_, index) => row[index] || "");
}

function formatInlineMarkdown(text) {
  const codeTokens = [];
  let html = escapeHtml(text).replace(/`([^`]+)`/g, (_, code) => {
    const token = `@@CODE${codeTokens.length}@@`;
    codeTokens.push(`<code>${code}</code>`);
    return token;
  });
  html = html
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  codeTokens.forEach((code, tokenIndex) => {
    html = html.replace(`@@CODE${tokenIndex}@@`, code);
  });
  return html;
}

function linkifyVirtualPaths(html) {
  return html.replace(/(\/docs\/web_uploads\/[^\s<>"'`,;()[\]{}]+)/g, (match) => {
    const cleanPath = match.replace(/[.,;:!?]+$/, "");
    const trailing = match.slice(cleanPath.length);
    return `<code class="inline-file-path">${escapeHtml(cleanPath)}</code>${trailing}`;
  });
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function displayRole(role) {
  return role === "admin" ? "admin" : "normal user";
}

function clearRunningPoll() {
  if (!state.runningPollTimer) return;
  clearInterval(state.runningPollTimer);
  state.runningPollTimer = null;
}

function clearAppRuntimeState() {
  clearRunningPoll();
  stopVoiceStream();
  state.voiceRecorder = null;
  state.voiceRecording = false;
  state.streamingResponse = false;
  state.autoFollowStream = true;
  state.messages = [];
  state.files = { uploads: [] };
  state.pendingAttachments = [];
  $("messages").textContent = "";
  $("admin-chat-list").textContent = "";
  $("admin-chat-detail").textContent = "";
  $("file-preview-body").textContent = "";
}

function showLogin(message = "") {
  clearAppRuntimeState();
  $("login-view").hidden = false;
  $("app-view").hidden = true;
  $("login-error").textContent = message;
}

function showApp() {
  const initials = (state.user.username || "?").slice(0, 2).toUpperCase();
  const avatar = document.getElementById("user-avatar-initials");
  if (avatar) avatar.textContent = initials;
  $("login-view").hidden = true;
  $("app-view").hidden = false;
  $("current-user").textContent = state.user.username;
  $("current-role").textContent = displayRole(state.user.role);
  $("manage-users").hidden = false;
}

function renderSessions() {
  const list = $("session-list");
  list.textContent = "";
  for (const session of state.sessions) {
    const item = document.createElement("div");
    item.className = "session-item";
    if (session.id === state.activeSessionId) item.classList.add("active");

    const titleButton = document.createElement("button");
    titleButton.className = "session-title";
    titleButton.type = "button";
    titleButton.textContent = session.title;
    titleButton.title = session.title;
    titleButton.addEventListener("click", () => {
      closeSessionMenu();
      loadSession(session.id);
    });

    const actions = document.createElement("div");
    actions.className = "session-actions";

    const menuButton = document.createElement("button");
    menuButton.className = "session-menu-trigger";
    menuButton.type = "button";
    menuButton.setAttribute("aria-label", `Chat options for ${session.title}`);
    menuButton.setAttribute("aria-haspopup", "menu");
    menuButton.setAttribute("aria-expanded", state.openSessionMenuId === session.id ? "true" : "false");
    menuButton.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
        <circle cx="3.5" cy="8" r="1.2" fill="currentColor"></circle>
        <circle cx="8" cy="8" r="1.2" fill="currentColor"></circle>
        <circle cx="12.5" cy="8" r="1.2" fill="currentColor"></circle>
      </svg>
    `;
    menuButton.addEventListener("click", (event) => {
      event.stopPropagation();
      state.openSessionMenuId = state.openSessionMenuId === session.id ? null : session.id;
      renderSessions();
    });
    actions.appendChild(menuButton);

    if (state.openSessionMenuId === session.id) {
      const menu = document.createElement("div");
      menu.className = "session-menu";
      menu.setAttribute("role", "menu");

      const renameButton = document.createElement("button");
      renameButton.type = "button";
      renameButton.setAttribute("role", "menuitem");
      renameButton.textContent = "Rename";
      renameButton.addEventListener("click", async (event) => {
        event.stopPropagation();
        closeSessionMenu(true);
        await renameSession(session.id, session.title);
      });

      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "danger";
      deleteButton.setAttribute("role", "menuitem");
      deleteButton.textContent = "Delete";
      deleteButton.addEventListener("click", async (event) => {
        event.stopPropagation();
        closeSessionMenu(true);
        await deleteSession(session.id, session.title);
      });

      menu.append(renameButton, deleteButton);
      actions.appendChild(menu);
    }

    item.append(titleButton, actions);
    list.appendChild(item);
  }
}

function closeSessionMenu(render = false) {
  if (!state.openSessionMenuId) return;
  state.openSessionMenuId = null;
  if (render) renderSessions();
}

async function renameSession(id, currentTitle = "") {
  const title = prompt("Rename chat", currentTitle);
  if (title === null) return;
  const nextTitle = title.trim();
  if (!nextTitle) return;
  await api(`/api/chat/sessions/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify({ title: nextTitle }),
  });
  await refreshSessions();
}

async function deleteSession(id, title = "this chat") {
  if (!confirm(`Delete "${title || "this chat"}" permanently? This removes it from the database.`)) return;
  await api(`/api/chat/sessions/${encodeURIComponent(id)}`, {
    method: "DELETE",
    body: "{}",
  });
  await refreshSessions();
  if (id === state.activeSessionId) {
    if (state.sessions.length) {
      await loadSession(state.sessions[0].id);
    } else {
      await createSession();
    }
  }
}

function createMessageElement(message) {
  message = sanitizeMessageForUi(message);
  const row = document.createElement("article");
  row.className = `message ${message.role}`;
  row.dataset.messageId = message.id || "";
  row.__messageSignature = messageSignature(message);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const isRunning = message.role === "assistant" && message.metadata?.status === "running" && !message.content;
  if (isRunning) {
    bubble.classList.add("streaming");
    bubble.textContent = "Working...";
  } else {
    bubble.innerHTML = formatMessage(displayMessageContent(message.content));
  }

  const attachments = message.attachments || message.metadata?.attachments || [];
  if (message.role === "user" && attachments.length) {
    const strip = document.createElement("div");
    strip.className = "attachment-strip message-attachments";
    renderAttachmentCards(strip, attachments);
    row.appendChild(strip);
  }
  row.append(bubble);
  if (message.role === "user" && message.id) {
    row.appendChild(createUserMessageControls(message, bubble));
  }
  renderActivity(row, message.metadata || {});
  return row;
}

function messageSignature(message) {
  const metadata = message.metadata || {};
  const trace = metadata.trace_events || [];
  const lastTrace = trace.length ? traceEventKey(trace[trace.length - 1]) : "";
  const logs = (metadata.turn_logs || []).slice(0, 12).map(sanitizeTurnLogForUi);
  return [
    message.id || "",
    message.role || "",
    message.content_truncated ? "truncated" : "",
    String(message.content || "").length,
    stableHash(message.content || ""),
    metadata.status || "",
    trace.length,
    lastTrace,
    logs.length,
    stableHash(logs.map((log) => `${log.status || ""}:${log.objective || ""}:${log.summary || ""}:${log.error || ""}`).join("|")),
  ].join("|");
}

function createUserMessageControls(message, bubble) {
  const controls = document.createElement("div");
  controls.className = "message-controls";

  const editButton = document.createElement("button");
  editButton.type = "button";
  editButton.className = "message-control-button icon-only";
  editButton.setAttribute("aria-label", "Edit message");
  editButton.title = "Edit message";
  editButton.innerHTML = pencilIcon();
  editButton.addEventListener("click", () => {
    showUserMessageEditor(message, bubble, controls);
  });
  controls.appendChild(editButton);

  const variants = message.branch_variants || [];
  if (variants.length > 1) {
    const switcher = document.createElement("div");
    switcher.className = "branch-switcher";
    for (const variant of variants) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "branch-switch-button";
      button.textContent = String(variant.number);
      button.disabled = Boolean(variant.active);
      button.setAttribute("aria-label", `Switch to branch ${variant.number}`);
      button.addEventListener("click", () => activateBranch(variant.branch_id));
      switcher.appendChild(button);
    }
    controls.appendChild(switcher);
  }

  return controls;
}

function showUserMessageEditor(message, bubble, controls) {
  if (bubble.querySelector(".message-edit-form")) return;
  const originalHtml = bubble.innerHTML;
  bubble.classList.add("editing");
  controls.hidden = true;
  const form = document.createElement("form");
  form.className = "message-edit-form";
  const input = document.createElement("textarea");
  input.value = message.content || "";
  input.rows = Math.min(10, Math.max(3, input.value.split("\n").length));
  input.addEventListener("input", () => resizeEditTextarea(input));
  const actions = document.createElement("div");
  actions.className = "message-edit-actions";
  const save = document.createElement("button");
  save.type = "submit";
  save.className = "btn-primary sm";
  save.textContent = "Save fork";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "btn-ghost-sm";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => {
    bubble.classList.remove("editing");
    bubble.innerHTML = originalHtml;
    controls.hidden = false;
  });
  actions.append(save, cancel);
  form.append(input, actions);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const content = input.value.trim();
    if (!content) return;
    if (content === (message.content || "").trim()) {
      bubble.classList.remove("editing");
      bubble.innerHTML = originalHtml;
      controls.hidden = false;
      return;
    }
    await forkFromUserMessage(message.id, content);
  });
  bubble.textContent = "";
  bubble.appendChild(form);
  resizeEditTextarea(input);
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
}

function resizeEditTextarea(input) {
  input.style.height = "auto";
  input.style.height = `${Math.max(92, input.scrollHeight)}px`;
}

function pencilIcon() {
  return `
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
      <path d="M8.9 2.1l3 3M2 12l3.1-.7 6.2-6.2a2.1 2.1 0 00-3-3L2.2 8.3 2 12z" stroke="currentColor" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round"></path>
    </svg>
  `;
}

async function activateBranch(branchId) {
  if (!state.activeSessionId || !branchId) return;
  setBusy(true);
  try {
    const data = await api(
      `/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/branches/${encodeURIComponent(branchId)}/activate`,
      {
        method: "POST",
        body: "{}",
      }
    );
    renderMessages(data.messages || [], {
      historyTruncated: data.messages_truncated,
      totalMessages: data.total_messages,
    });
    await refreshSessions();
    await refreshFiles();
  } finally {
    setBusy(false);
  }
}

async function forkFromUserMessage(messageId, content) {
  if (!state.activeSessionId || !messageId) return;
  setBusy(true);
  try {
    await api(
      `/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/messages/${encodeURIComponent(messageId)}/fork`,
      {
        method: "POST",
        body: JSON.stringify({ content }),
      }
    );
    await loadSession(state.activeSessionId);
    await refreshSessions();
  } finally {
    setBusy(false);
  }
}

function renderActivity(row, metadata) {
  const logs = (metadata.turn_logs || []).slice(0, 12).map(sanitizeTurnLogForUi);
  const traceEvents = capTraceEvents((metadata.trace_events || []).map(sanitizeTraceEventForUi));
  const visibleLogs = logs.filter((log) => log.objective || log.summary || log.error);
  const visibleTrace = traceEvents.filter((event) => {
    return ["model_request", "model_tools", "tool_call", "tool_result", "tool_call_start"].includes(event.kind);
  });

  if (!row.__activityTraceEvents) row.__activityTraceEvents = new Map();
  const traceMap = row.__activityTraceEvents;
  for (const event of visibleTrace) {
    traceMap.set(traceEventKey(event), event);
    while (traceMap.size > MAX_PENDING_TRACE_EVENTS) {
      traceMap.delete(traceMap.keys().next().value);
    }
  }

  const existing = row.querySelector(".activity");
  if (!visibleLogs.length && !traceMap.size) {
    if (existing) existing.remove();
    return;
  }

  let details = existing;
  if (!details) {
    details = document.createElement("details");
    details.className = "activity";
    details.open = metadata.status === "running";
    const summary = document.createElement("summary");
    const trace = document.createElement("div");
    trace.className = "tool-trace";
    const logList = document.createElement("div");
    logList.className = "activity-logs";
    details.append(summary, trace, logList);
    row.appendChild(details);
  }

  const summary = details.querySelector("summary");
  const done = visibleLogs.filter((log) => log.status === "done").length;
  const toolCalls = [...traceMap.values()].filter((event) => event.kind === "tool_call").length;
  summary.textContent = `Activity: ${toolCalls} tool call${toolCalls === 1 ? "" : "s"}${visibleLogs.length ? `, ${done}/${visibleLogs.length} tasks` : ""}`;

  const trace = details.querySelector(".tool-trace");
  trace.hidden = traceMap.size === 0;
  for (const [key, event] of traceMap.entries()) {
    if (!trace.querySelector(`[data-trace-key="${cssEscape(key)}"]`)) {
      trace.appendChild(createTraceEventElement(event, key));
    }
  }

  const logList = details.querySelector(".activity-logs");
  logList.textContent = "";
  logList.hidden = visibleLogs.length === 0;
  for (const log of visibleLogs) {
    const item = document.createElement("div");
    item.className = `activity-item ${log.status === "done" ? "done" : "failed"}`;
    const title = document.createElement("strong");
    title.textContent = log.objective || log.task_id || "Agent task";
    const body = document.createElement("p");
    body.textContent = log.error || log.summary || "";
    item.append(title, body);
    logList.appendChild(item);
  }
}

function traceEventKey(event) {
  const identity = [
    event.ts || "",
    event.kind || "",
    event.label || "",
    event.tool_call_id || "",
    event.tool_name || "",
  ].join("|");
  return `${identity}|${stableHash(`${event.args || ""}|${event.output || ""}`)}`;
}

function cssEscape(value) {
  if (window.CSS?.escape) return window.CSS.escape(value);
  return String(value).replace(/["\\]/g, "\\$&");
}

function createTraceEventElement(event, key = "") {
  const item = document.createElement("div");
  item.className = `trace-event ${event.kind}`;
  if (key) item.dataset.traceKey = key;

  const label = document.createElement("span");
  label.className = "trace-label";
  label.textContent = event.label || "agent";

  const title = document.createElement("strong");
  if (event.kind === "model_request") {
    title.textContent = "model request";
  } else if (event.kind === "model_tools") {
    title.textContent = `model selected ${event.tool_name}`;
  } else if (event.kind === "tool_result") {
    title.textContent = `${event.tool_name || "tool"} returned`;
  } else {
    title.textContent = event.tool_name || "tool call";
  }

  const body = document.createElement("code");
  body.textContent = truncateUiText(event.output || event.args || "", MAX_TRACE_FIELD_DISPLAY_CHARS);

  item.append(label, title);
  if (body.textContent) item.appendChild(body);
  return item;
}

function mergeTraceEvents(...lists) {
  const merged = new Map();
  for (const list of lists) {
    for (const event of list || []) {
      const compact = sanitizeTraceEventForUi(event);
      merged.set(traceEventKey(compact), compact);
      while (merged.size > MAX_PENDING_TRACE_EVENTS) {
        merged.delete(merged.keys().next().value);
      }
    }
  }
  return [...merged.values()];
}

function renderMessages(messages, options = {}) {
  const stickToBottom = options.stickToBottom ?? true;
  const safeMessages = sanitizeMessagesForUi(messages);
  state.messages = safeMessages;
  const el = $("messages");
  el.textContent = "";
  if (!safeMessages.length) {
    el.appendChild(createMessageElement({
      role: "assistant",
      content: "Start a new conversation, or upload files to add local context.",
    }));
    updateRunningPoll([]);
    return;
  }
  syncHistoryNotice(el, options, safeMessages.length);
  for (const message of safeMessages) {
    el.appendChild(createMessageElement(message));
  }
  maybeScrollMessagesToBottom(stickToBottom);
  updateRunningPoll(safeMessages);
}

function syncHistoryNotice(container, options, renderedCount) {
  const existing = container.querySelector(".history-truncation-notice");
  if (!options.historyTruncated) {
    if (existing) existing.remove();
    return;
  }
  const total = Number(options.totalMessages || 0);
  const hidden = Math.max(0, total - renderedCount);
  const text = hidden
    ? `${hidden} older message${hidden === 1 ? "" : "s"} hidden to keep this tab responsive.`
    : "Older messages hidden to keep this tab responsive.";
  const notice = existing || document.createElement("div");
  notice.className = "history-truncation-notice";
  notice.textContent = text;
  if (!existing) container.prepend(notice);
}

function updateMessagesInPlace(messages, options = {}) {
  const safeMessages = sanitizeMessagesForUi(messages);
  const el = $("messages");
  const rows = [...el.querySelectorAll(".message[data-message-id]")];
  const rowsById = new Map(rows.map((row) => [row.dataset.messageId, row]));
  const canPatch = safeMessages.length === rows.length
    && safeMessages.every((message) => message.id && rowsById.has(String(message.id)));
  if (!canPatch) {
    renderMessages(safeMessages, { ...options, stickToBottom: shouldFollowMessagesBottom() });
    return;
  }

  const stickToBottom = shouldFollowMessagesBottom();
  state.messages = safeMessages;
  syncHistoryNotice(el, options, safeMessages.length);
  for (const message of safeMessages) {
    updateMessageElement(rowsById.get(String(message.id)), message);
  }
  maybeScrollMessagesToBottom(stickToBottom);
  updateRunningPoll(safeMessages);
}

function updateRunningPoll(messages = []) {
  const hasRunning = messages.some((message) => message.role === "assistant" && message.metadata?.status === "running");
  if (!hasRunning) {
    if (state.runningPollTimer && !hasRunning) {
      clearInterval(state.runningPollTimer);
      state.runningPollTimer = null;
    }
    return;
  }
  if (state.runningPollTimer) return;
  state.runningPollTimer = setInterval(async () => {
    if (!state.activeSessionId) return;
    try {
      const data = await api(`/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}`);
      updateMessagesInPlace(data.messages, {
        historyTruncated: data.messages_truncated,
        totalMessages: data.total_messages,
      });
      const stillRunning = data.messages.some((message) => message.role === "assistant" && message.metadata?.status === "running");
      if (!stillRunning && state.runningPollTimer) {
        clearInterval(state.runningPollTimer);
        state.runningPollTimer = null;
        await refreshSessions();
        await refreshFiles();
      }
    } catch (_) {
      clearInterval(state.runningPollTimer);
      state.runningPollTimer = null;
    }
  }, 2500);
}

function appendLocalMessage(role, content, attachments = []) {
  const row = createMessageElement({ role, content, attachments });
  $("messages").appendChild(row);
  scrollMessagesToBottom();
  return row;
}

function updateMessageElement(row, message) {
  if (!row || !message) return;
  message = sanitizeMessageForUi(message);
  const signature = messageSignature(message);
  if (row.__messageSignature === signature) return;
  row.__messageSignature = signature;
  const bubble = row.querySelector(".bubble");
  if (bubble) {
    if (message.metadata?.status === "running" && !message.content) {
      bubble.classList.add("streaming");
      bubble.textContent = "Working...";
    } else {
      bubble.classList.remove("streaming");
      bubble.classList.remove("streaming-text");
      bubble.innerHTML = formatMessage(displayMessageContent(message.content || ""));
    }
  }
  renderActivity(row, message.metadata || {});
}

async function refreshSessions() {
  const data = await api("/api/chat/sessions");
  state.sessions = data.sessions;
  renderSessions();
}

function isEmptyNewChat(session) {
  const messageCount = session?.message_count ?? 0;
  const fileCount = session?.file_count ?? 0;
  return session
    && session.title === "New chat"
    && (
      session.is_empty === true
      || (session.is_empty === undefined && messageCount === 0 && fileCount === 0)
    );
}

async function createSession() {
  await refreshSessions();
  const existingEmpty = state.sessions.find(isEmptyNewChat);
  if (existingEmpty) {
    state.activeSessionId = existingEmpty.id;
    await loadSession(existingEmpty.id);
    return;
  }

  const data = await api("/api/chat/sessions", {
    method: "POST",
    body: JSON.stringify({ title: "New chat" }),
  });
  state.activeSessionId = data.session.id;
  await refreshSessions();
  await loadSession(data.session.id);
}

async function loadSession(id) {
  const data = await api(`/api/chat/sessions/${encodeURIComponent(id)}`);
  state.activeSessionId = id;
  state.pendingAttachments = [];
  renderSessions();
  renderMessages(data.messages, {
    historyTruncated: data.messages_truncated,
    totalMessages: data.total_messages,
  });
  await refreshFiles();
  renderComposerAttachments();
}

function setBusy(value) {
  state.busy = value;
  const sendButton = $("send-message");
  if (sendButton) sendButton.disabled = value;
  $("message-input").disabled = value;
  $("voice-input").disabled = (value && !state.voiceRecording) || !state.voiceSupported;
}

function closeUploadMenu() {
  $("upload-menu").hidden = true;
  $("upload-menu-trigger").setAttribute("aria-expanded", "false");
}

function toggleUploadMenu() {
  const menu = $("upload-menu");
  const nextOpen = menu.hidden;
  menu.hidden = !nextOpen;
  $("upload-menu-trigger").setAttribute("aria-expanded", nextOpen ? "true" : "false");
}

async function sendMessage(content) {
  if (!state.activeSessionId) {
    await createSession();
  }
  const attachments = getCurrentUploadAttachments();
  appendLocalMessage("user", content, attachments);
  state.pendingAttachments = [];
  $("upload-status").textContent = "";
  renderComposerAttachments();
  const pending = appendLocalMessage("assistant", "");
  const pendingBubble = pending.querySelector(".bubble");
  pendingBubble.classList.add("streaming");
  pendingBubble.textContent = "Working...";
  const pendingMetadata = { trace_events: [] };
  let streamedContent = "";
  let streamTruncated = false;
  let streamRenderTimer = null;
  const clearStreamRenderTimer = () => {
    if (!streamRenderTimer) return;
    window.clearTimeout(streamRenderTimer);
    streamRenderTimer = null;
  };
  const appendStreamDelta = (delta) => {
    if (!delta || streamTruncated) return;
    const nextLength = streamedContent.length + delta.length;
    if (nextLength > MAX_RENDERED_MESSAGE_CHARS) {
      const contentLimit = Math.max(0, MAX_RENDERED_MESSAGE_CHARS - TRUNCATED_OUTPUT_NOTICE.length);
      const base = streamedContent.slice(0, contentLimit).trimEnd();
      const remaining = Math.max(0, contentLimit - base.length);
      streamedContent = `${base}${delta.slice(0, remaining)}${TRUNCATED_OUTPUT_NOTICE}`;
      streamTruncated = true;
      return;
    }
    streamedContent += delta;
  };
  const renderStreamContent = (force = false) => {
    if (!force && streamRenderTimer) return;
    const render = () => {
      streamRenderTimer = null;
      const stickToBottom = shouldFollowMessagesBottom();
      pendingBubble.classList.remove("streaming");
      pendingBubble.classList.add("streaming-text");
      pendingBubble.textContent = streamedContent || "Working...";
      maybeScrollMessagesToBottom(stickToBottom);
    };
    if (force) {
      clearStreamRenderTimer();
      render();
      return;
    }
    streamRenderTimer = window.setTimeout(render, STREAM_RENDER_INTERVAL_MS);
  };
  state.streamingResponse = true;
  state.autoFollowStream = true;
  setBusy(true);
  try {
    const data = await streamApi(
      `/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/messages/stream`,
      { content },
      (event) => {
        if (event.kind === "answer_delta") {
          appendStreamDelta(event.content || "");
          renderStreamContent();
          return;
        }
        if (event.kind === "answer_replace") {
          const stickToBottom = shouldFollowMessagesBottom();
          streamedContent = displayMessageContent(event.content || streamedContent);
          streamTruncated = streamedContent.endsWith(TRUNCATED_OUTPUT_NOTICE);
          clearStreamRenderTimer();
          pendingBubble.classList.remove("streaming");
          pendingBubble.classList.remove("streaming-text");
          pendingBubble.innerHTML = formatMessage(streamedContent);
          maybeScrollMessagesToBottom(stickToBottom);
          return;
        }
        const stickToBottom = shouldFollowMessagesBottom();
        pendingMetadata.trace_events.push(sanitizeTraceEventForUi(event));
        pendingMetadata.trace_events = capTraceEvents(pendingMetadata.trace_events);
        renderActivity(pending, pendingMetadata);
        maybeScrollMessagesToBottom(stickToBottom);
      }
    );
    renderStreamContent(true);
    if (data?.message) {
      const safeMessage = sanitizeMessageForUi(data.message);
      safeMessage.metadata = {
        ...(safeMessage.metadata || {}),
        trace_events: mergeTraceEvents(
          pendingMetadata.trace_events,
          safeMessage.metadata?.trace_events || []
        ),
      };
      const replacement = createMessageElement(safeMessage);
      const stickToBottom = shouldFollowMessagesBottom();
      pending.replaceWith(replacement);
      maybeScrollMessagesToBottom(stickToBottom);
    } else {
      await loadSession(state.activeSessionId);
    }
    await refreshSessions();
    await refreshFiles();
  } catch (error) {
    clearStreamRenderTimer();
    await loadSession(state.activeSessionId).catch(() => {
      const bubble = pending.querySelector(".bubble");
      bubble.textContent = error.message;
    });
  } finally {
    clearStreamRenderTimer();
    state.streamingResponse = false;
    state.autoFollowStream = true;
    setBusy(false);
  }
}

async function refreshFiles({ clearStatus = true } = {}) {
  if (clearStatus) $("upload-status").textContent = "";
  if (!state.activeSessionId) return;
  try {
    const data = await api(`/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/files`);
    state.files = { uploads: data.uploads || [] };
    renderComposerAttachments();
  } catch (error) {
    $("upload-status").textContent = error.message;
  }
}

function normalizeAttachment(item) {
  return {
    id: item.id,
    filename: item.filename || item.name || "Uploaded file",
    size_bytes: item.size_bytes || 0,
    virtual_path: item.virtual_path || "",
    content_type: item.content_type || "",
  };
}

function getCurrentUploadAttachments() {
  return state.pendingAttachments.map(normalizeAttachment);
}

function attachmentMeta(item) {
  const bits = [formatBytes(item.size_bytes)];
  if (item.virtual_path) bits.push(item.virtual_path);
  return bits.filter(Boolean).join(" - ");
}

function renderAttachmentCards(container, items) {
  container.textContent = "";
  if (!items.length) {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  for (const item of items) {
    const card = document.createElement("button");
    card.className = "attachment-card";
    card.type = "button";
    card.setAttribute("aria-label", `Preview ${item.filename || item.name || "uploaded file"}`);
    card.addEventListener("click", () => openAttachmentPreview(item));
    const icon = document.createElement("span");
    icon.className = "attachment-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = fileIcon(item.filename || item.name || "");
    const text = document.createElement("span");
    text.className = "attachment-text";
    const name = document.createElement("strong");
    name.textContent = item.filename || item.name || "Uploaded file";
    const meta = document.createElement("span");
    meta.textContent = attachmentMeta(item);
    text.append(name, meta);
    card.append(icon, text);
    container.appendChild(card);
  }
}

function renderComposerAttachments() {
  renderAttachmentCards($("composer-attachments"), getCurrentUploadAttachments());
}

function fileIcon(filename) {
  const ext = filename.split(".").pop()?.toLowerCase() || "";
  if (["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(ext)) return "IMG";
  if (ext === "pdf") return "PDF";
  if (["md", "txt", "json", "csv", "log"].includes(ext)) return ext.toUpperCase();
  return "FILE";
}

async function uploadSelectedFile(file) {
  if (!file) return;
  if (file.size > MAX_UPLOAD_FILE_BYTES) {
    $("upload-status").textContent = `${file.name} is larger than ${formatBytes(MAX_UPLOAD_FILE_BYTES)}.`;
    return;
  }
  if (!state.activeSessionId) {
    await createSession();
  }
  const status = $("upload-status");
  status.textContent = `Uploading ${file.name}...`;
  const formData = new FormData();
  formData.append("file", file);
  try {
    const data = await api(`/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/files`, {
      method: "POST",
      body: formData,
    });
    status.textContent = `Uploaded ${data.file.filename}`;
    state.pendingAttachments.push(normalizeAttachment(data.file));
    renderComposerAttachments();
    await refreshFiles({ clearStatus: false });
  } catch (error) {
    status.textContent = error.message;
  } finally {
    $("file-input").value = "";
  }
}

async function uploadSelectedFiles(files) {
  const selected = [...files].slice(0, MAX_UPLOAD_BATCH_FILES);
  if (files.length > MAX_UPLOAD_BATCH_FILES) {
    $("upload-status").textContent = `Only the first ${MAX_UPLOAD_BATCH_FILES} files were queued.`;
  }
  for (const file of selected) {
    await uploadSelectedFile(file);
  }
}

function filesFromTransfer(fileList) {
  return [...(fileList || [])].filter((file) => file && file.size > 0);
}

function setVoiceRecording(value) {
  state.voiceRecording = value;
  const button = $("voice-input");
  button.classList.toggle("is-recording", value);
  button.setAttribute("aria-label", value ? "Stop voice input" : "Start voice input");
  button.title = value ? "Stop voice input" : "Start voice input";
  button.disabled = !state.voiceSupported || (state.busy && !value);
}

function stopVoiceStream() {
  if (!state.voiceStream) return;
  state.voiceStream.getTracks().forEach((track) => track.stop());
  state.voiceStream = null;
}

function audioContextConstructor() {
  return window.AudioContext || window.webkitAudioContext;
}

function createWavVoiceRecorder(stream) {
  const AudioContextClass = audioContextConstructor();
  if (!AudioContextClass) {
    throw new Error("Voice input is not available in this browser.");
  }

  const audioContext = new AudioContextClass();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const silentGain = audioContext.createGain();
  const chunks = [];
  const maxSamples = Math.floor(audioContext.sampleRate * MAX_VOICE_RECORDING_SECONDS);
  let capturedSamples = 0;
  let limitReached = false;
  let stopped = false;
  silentGain.gain.value = 0;

  processor.onaudioprocess = (event) => {
    if (stopped || limitReached) return;
    const input = event.inputBuffer.getChannelData(0);
    const remaining = maxSamples - capturedSamples;
    if (remaining <= 0) {
      limitReached = true;
      window.setTimeout(() => stopVoiceInput(), 0);
      return;
    }
    const samples = input.length > remaining ? input.slice(0, remaining) : input;
    chunks.push(new Float32Array(samples));
    capturedSamples += samples.length;
    if (capturedSamples >= maxSamples) {
      limitReached = true;
      $("upload-status").textContent = `Recording stopped at ${MAX_VOICE_RECORDING_SECONDS} seconds.`;
      window.setTimeout(() => stopVoiceInput(), 0);
    }
  };
  source.connect(processor);
  processor.connect(silentGain);
  silentGain.connect(audioContext.destination);
  audioContext.resume?.().catch(() => {});

  return {
    async stop() {
      const sampleRate = audioContext.sampleRate;
      stopped = true;
      processor.onaudioprocess = null;
      source.disconnect();
      processor.disconnect();
      silentGain.disconnect();
      await audioContext.close().catch(() => {});
      const blob = encodeWavBlob(chunks, sampleRate);
      chunks.length = 0;
      return blob;
    },
  };
}

function mergeAudioChunks(chunks) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function encodeWavBlob(chunks, sampleRate) {
  const samples = mergeAudioChunks(chunks);
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

function resizeMessageInput() {
  const input = $("message-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

function insertVoiceTranscript(text) {
  const transcript = String(text || "").trim();
  if (!transcript) return;
  const input = $("message-input");
  const existing = input.value;
  const joiner = existing.trim() && !/\s$/.test(existing) ? " " : "";
  input.value = `${existing}${joiner}${transcript}`;
  resizeMessageInput();
  input.focus();
}

async function transcribeVoiceBlob(blob) {
  const status = $("upload-status");
  if (!blob.size) {
    status.textContent = "No audio recorded.";
    return;
  }
  status.textContent = "Transcribing voice...";
  const formData = new FormData();
  formData.append("file", blob, "voice-input.wav");
  try {
    const data = await api("/api/speech/asr", {
      method: "POST",
      body: formData,
    });
    insertVoiceTranscript(data.text);
    status.textContent = data.text ? "" : "No speech detected.";
  } catch (error) {
    status.textContent = error.message;
  }
}

async function startVoiceInput() {
  if (!state.voiceSupported) {
    $("upload-status").textContent = "Voice input is not available in this browser.";
    return;
  }
  if (state.busy || state.voiceRecording) return;
  closeUploadMenu();
  $("upload-status").textContent = "Recording...";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = createWavVoiceRecorder(stream);
    state.voiceStream = stream;
    state.voiceRecorder = recorder;
    setVoiceRecording(true);
  } catch (error) {
    stopVoiceStream();
    state.voiceRecorder = null;
    setVoiceRecording(false);
    $("upload-status").textContent = error?.message || "Could not start voice input.";
  }
}

async function stopVoiceInput() {
  const recorder = state.voiceRecorder;
  if (!recorder) {
    setVoiceRecording(false);
    stopVoiceStream();
    return;
  }
  $("upload-status").textContent = "Transcribing voice...";
  state.voiceRecorder = null;
  try {
    const blob = await recorder.stop();
    await transcribeVoiceBlob(blob);
  } catch (error) {
    $("upload-status").textContent = error?.message || "Could not transcribe voice input.";
  } finally {
    stopVoiceStream();
    setVoiceRecording(false);
  }
}

function closeAttachmentPreview() {
  $("file-preview-modal").hidden = true;
  $("file-preview-title").textContent = "File preview";
  $("file-preview-meta").textContent = "";
  $("file-preview-body").textContent = "";
}

function renderPreviewLoading(item) {
  $("file-preview-title").textContent = item.filename || item.name || "Uploaded file";
  $("file-preview-meta").textContent = attachmentMeta(item);
  const body = $("file-preview-body");
  body.textContent = "";
  const loading = document.createElement("p");
  loading.className = "file-preview-note";
  loading.textContent = "Loading preview...";
  body.appendChild(loading);
  $("file-preview-modal").hidden = false;
}

function renderPreviewBody(item, data) {
  const body = $("file-preview-body");
  body.textContent = "";
  const file = data.file || item;
  const rawUrl = data.raw_url || "";
  const contentType = file.content_type || item.content_type || "";
  const filename = file.filename || item.filename || item.name || "Uploaded file";
  const ext = filename.split(".").pop()?.toLowerCase() || "";
  const isImage = contentType.startsWith("image/") || ["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(ext);
  const isPdf = contentType === "application/pdf" || ext === "pdf";

  $("file-preview-title").textContent = filename;
  $("file-preview-meta").textContent = attachmentMeta(file);

  if (data.is_text) {
    const pre = document.createElement("pre");
    pre.className = "file-preview-text";
    pre.textContent = displayPreviewContent(data.content || "");
    body.appendChild(pre);
    return;
  }

  if (isImage && rawUrl) {
    const img = document.createElement("img");
    img.className = "file-preview-image";
    img.src = rawUrl;
    img.alt = filename;
    body.appendChild(img);
    return;
  }

  if (isPdf && rawUrl) {
    const frame = document.createElement("iframe");
    frame.className = "file-preview-frame";
    frame.src = rawUrl;
    frame.title = filename;
    body.appendChild(frame);
    return;
  }

  const note = document.createElement("p");
  note.className = "file-preview-note";
  note.textContent = "This file type cannot be previewed inline.";
  body.appendChild(note);
  if (rawUrl) {
    const link = document.createElement("a");
    link.className = "file-preview-link";
    link.href = rawUrl;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = "Open file";
    body.appendChild(link);
  }
}

async function openAttachmentPreview(item) {
  const attachment = normalizeAttachment(item);
  renderPreviewLoading(attachment);
  if (isDesignPreview() || !state.activeSessionId || !attachment.id) {
    renderPreviewBody(attachment, {
      file: attachment,
      is_text: true,
      content: attachment.virtual_path
        ? `Preview unavailable in design mode.\n\n${attachment.virtual_path}`
        : "Preview unavailable for this file.",
    });
    return;
  }
  try {
    const data = await api(
      `/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/files/${encodeURIComponent(attachment.id)}/content`
    );
    renderPreviewBody(attachment, data);
  } catch (error) {
    const body = $("file-preview-body");
    body.textContent = "";
    const note = document.createElement("p");
    note.className = "file-preview-note error";
    note.textContent = error.message;
    body.appendChild(note);
  }
}

function showDesignPreview() {
  const now = new Date().toISOString();
  state.user = { username: "design", role: "admin" };
  state.sessions = [
    { id: "preview-1", title: "Research planning" },
    { id: "preview-2", title: "Upload review" },
    { id: "preview-3", title: "Agent workflow notes" },
  ];
  state.activeSessionId = "preview-1";
  state.files = {
    uploads: [
      {
        id: "preview-upload",
        filename: "product-context.md",
        virtual_path: "/docs/web_uploads/preview/product-context.md",
        size_bytes: 1840,
      },
    ],
  };
  state.pendingAttachments = state.files.uploads.map(normalizeAttachment);
  showApp();
  renderSessions();
  renderMessages([
    {
      role: "user",
      content: "Review the uploaded product context and outline the next research steps.",
      attachments: getCurrentUploadAttachments(),
    },
    {
      role: "assistant",
      content: "I found three useful threads: clarify the target users, compare the current onboarding flow, and collect examples from similar local-first tools.",
      metadata: {
        turn_logs: [
          {
            status: "done",
            objective: "Read uploaded context",
            summary: "Extracted product goals, constraints, and open design questions.",
          },
          {
            status: "done",
            objective: "Plan next steps",
            summary: "Grouped the work into user research, interface audit, and implementation tasks.",
          },
        ],
      },
    },
  ]);
  renderComposerAttachments();
}

async function boot() {
  applyTheme(state.theme);
  if (isDesignPreview()) {
    showDesignPreview();
    return;
  }
  try {
    const data = await api("/api/me", { noAuthRedirect: true });
    state.user = data.user;
    showApp();
    await refreshSessions();
    if (state.sessions.length) {
      await loadSession(state.sessions[0].id);
    } else {
      await createSession();
    }
  } catch (_) {
    showLogin();
  }
}

function applyTheme(theme) {
  state.theme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = state.theme;
  storageSet("localagent_theme", state.theme);
  const icon = $("theme-toggle-icon");
  if (icon) icon.textContent = state.theme === "light" ? "●" : "○";
  const toggle = $("theme-toggle");
  if (toggle) {
    toggle.setAttribute("aria-label", state.theme === "light" ? "Switch to dark theme" : "Switch to light theme");
    toggle.title = "Switch dark / light theme";
  }
}

function applySidebarCollapsed(collapsed) {
  state.sidebarCollapsed = Boolean(collapsed);
  document.documentElement.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
  storageSet("localagent_sidebar_collapsed", state.sidebarCollapsed ? "1" : "0");
  const icon = $("sidebar-collapse-icon");
  if (icon) icon.textContent = "‹";
  const button = $("sidebar-collapse");
  if (button) {
    button.setAttribute("aria-label", "Collapse sidebar");
    button.title = "Collapse sidebar";
  }
  const logoButton = $("sidebar-logo-trigger");
  if (logoButton) {
    logoButton.disabled = !state.sidebarCollapsed;
    logoButton.setAttribute("aria-label", state.sidebarCollapsed ? "Expand sidebar" : "Local Agent logo");
    logoButton.setAttribute("title", state.sidebarCollapsed ? "Expand sidebar" : "");
  }
}

function setupSidebarState() {
  applySidebarCollapsed(state.sidebarCollapsed);
}

$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const data = await api("/api/auth/login", {
      method: "POST",
      noAuthRedirect: true,
      body: JSON.stringify({
        username: $("login-username").value.trim(),
        password: $("login-password").value,
      }),
    });
    state.user = data.user;
    await boot();
  } catch (error) {
    const message = error.status === 401
      ? "Invalid username or password. Register first if you do not have an account."
      : error.message;
    showLogin(message);
  }
});

$("register-button").addEventListener("click", async () => {
  if (!$("login-form").reportValidity()) return;
  try {
    const data = await api("/api/auth/register", {
      method: "POST",
      noAuthRedirect: true,
      body: JSON.stringify({
        username: $("login-username").value.trim(),
        password: $("login-password").value,
      }),
    });
    state.user = data.user;
    await boot();
  } catch (error) {
    showLogin(error.message);
  }
});

$("logout").addEventListener("click", async () => {
  await api("/api/auth/logout", { method: "POST", body: "{}" }).catch(() => {});
  state.user = null;
  state.sessions = [];
  state.activeSessionId = null;
  state.files = { uploads: [] };
  state.pendingAttachments = [];
  showLogin();
});

async function openAdmin() {
  $("admin-panel").hidden = false;
  $("admin-error").textContent = "";
  $("profile-password-status").textContent = "";
  $("settings-owner").textContent = `${state.user.username} - ${displayRole(state.user.role)}`;
  $("admin-chat-detail").textContent = "";
  state.settingsChatUserId = null;
  const isAdmin = state.user.role === "admin";
  $("profile-password-section").hidden = isAdmin;
  $("admin-settings-section").hidden = !isAdmin;
  if (isAdmin) {
    await refreshUsers();
    await refreshAdminChats();
  }
}

async function refreshUsers() {
  const data = await api("/api/admin/users");
  const list = $("user-list");
  list.textContent = "";
  if (!data.users.length) {
    const empty = document.createElement("p");
    empty.className = "admin-empty";
    empty.textContent = "No users yet.";
    list.appendChild(empty);
    return;
  }
  for (const user of data.users) {
    const row = document.createElement("div");
    row.className = "user-row";
    const identity = document.createElement("div");
    identity.className = "user-identity";
    const name = document.createElement("strong");
    name.textContent = user.username;
    const meta = document.createElement("span");
    meta.textContent = `${displayRole(user.role)}${user.is_active ? "" : " - inactive"}`;
    identity.append(name, meta);

    const passwordForm = document.createElement("form");
    passwordForm.className = "user-password-form";
    const passwordInput = document.createElement("input");
    passwordInput.type = "password";
    passwordInput.minLength = 8;
    passwordInput.placeholder = "New password";
    passwordInput.autocomplete = "new-password";
    const resetButton = document.createElement("button");
    resetButton.className = "btn-ghost-sm";
    resetButton.type = "submit";
    resetButton.textContent = "Reset";
    const rowStatus = document.createElement("span");
    rowStatus.className = "user-row-status";
    passwordForm.append(passwordInput, resetButton);
    passwordForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      rowStatus.textContent = "";
      const nextPassword = passwordInput.value;
      if (nextPassword.length < 8) {
        rowStatus.textContent = "Use at least 8 characters.";
        return;
      }
      try {
        await api(`/api/admin/users/${user.id}`, {
          method: "PATCH",
          body: JSON.stringify({ password: nextPassword }),
        });
        passwordInput.value = "";
        rowStatus.textContent = "Password reset.";
      } catch (error) {
        rowStatus.textContent = error.message;
      }
    });

    const actions = document.createElement("div");
    actions.className = "user-row-actions";
    const chats = document.createElement("button");
    chats.className = "btn-ghost-sm";
    chats.type = "button";
    chats.textContent = "Chats";
    chats.addEventListener("click", async () => {
      state.settingsChatUserId = user.id;
      await refreshAdminChats();
    });
    const toggle = document.createElement("button");
    toggle.className = "btn-ghost-sm";
    toggle.type = "button";
    toggle.textContent = user.is_active ? "Disable" : "Enable";
    toggle.disabled = user.id === state.user.id;
    toggle.addEventListener("click", async () => {
      await api(`/api/admin/users/${user.id}`, {
        method: "PATCH",
        body: JSON.stringify({ is_active: !user.is_active }),
      });
      await refreshUsers();
    });
    actions.append(chats, toggle);
    row.append(identity, passwordForm, actions, rowStatus);
    list.appendChild(row);
  }
}

async function refreshAdminChats() {
  const data = await api("/api/admin/chat/sessions");
  const list = $("admin-chat-list");
  list.textContent = "";
  const sessions = state.settingsChatUserId
    ? data.sessions.filter((session) => session.user_id === state.settingsChatUserId)
    : data.sessions;
  if (!sessions.length) {
    const empty = document.createElement("p");
    empty.className = "admin-empty";
    empty.textContent = state.settingsChatUserId ? "No chat history for this user." : "No chat history yet.";
    list.appendChild(empty);
    $("admin-chat-detail").textContent = "";
    return;
  }
  if (data.truncated) {
    const notice = document.createElement("p");
    notice.className = "admin-empty";
    notice.textContent = "Showing the most recent chat sessions.";
    list.appendChild(notice);
  }
  for (const session of sessions) {
    const button = document.createElement("button");
    button.className = "admin-chat-item";
    button.type = "button";
    const title = document.createElement("strong");
    title.textContent = session.title;
    const meta = document.createElement("span");
    meta.textContent = `${session.username} - ${new Date(session.updated_at).toLocaleString()}`;
    button.append(title, meta);
    button.addEventListener("click", () => loadAdminChat(session.id));
    list.appendChild(button);
  }
}

async function loadAdminChat(id) {
  const data = await api(`/api/admin/chat/sessions/${encodeURIComponent(id)}`);
  const detail = $("admin-chat-detail");
  detail.textContent = "";

  const heading = document.createElement("div");
  heading.className = "admin-chat-heading";
  const title = document.createElement("strong");
  title.textContent = data.session.title;
  const owner = document.createElement("span");
  owner.textContent = data.session.username;
  heading.append(title, owner);
  detail.appendChild(heading);

  if (!data.messages.length) {
    const empty = document.createElement("p");
    empty.className = "admin-empty";
    empty.textContent = "No messages in this chat.";
    detail.appendChild(empty);
    return;
  }

  if (data.messages_truncated) {
    const notice = document.createElement("p");
    notice.className = "admin-empty";
    const hidden = Math.max(0, Number(data.total_messages || 0) - data.messages.length);
    notice.textContent = hidden
      ? `Showing newest messages. ${hidden} older message${hidden === 1 ? "" : "s"} hidden.`
      : "Showing newest messages.";
    detail.appendChild(notice);
  }

  for (const message of data.messages) {
    const row = document.createElement("article");
    row.className = "admin-chat-message";
    const role = document.createElement("span");
    role.textContent = message.role === "user" ? "User" : "Local Agent";
    const content = document.createElement("p");
    content.textContent = displayMessageContent(message.content);
    row.append(role, content);
    detail.appendChild(row);
  }
}

$("manage-users").addEventListener("click", openAdmin);
$("close-admin").addEventListener("click", () => {
  $("admin-panel").hidden = true;
});

$("change-password-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = $("profile-password-status");
  status.textContent = "";
  const currentPassword = $("current-password").value;
  const newPassword = $("own-new-password").value;
  const confirmPassword = $("own-confirm-password").value;
  if (newPassword !== confirmPassword) {
    status.textContent = "New passwords do not match.";
    return;
  }
  try {
    await api("/api/me/password", {
      method: "PATCH",
      body: JSON.stringify({
        current_password: currentPassword,
        new_password: newPassword,
      }),
    });
    $("change-password-form").reset();
    status.textContent = "Password updated.";
  } catch (error) {
    status.textContent = error.message;
  }
});

$("create-user-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("admin-error").textContent = "";
  try {
    await api("/api/admin/users", {
      method: "POST",
      body: JSON.stringify({
        username: $("new-user-username").value,
        password: $("new-user-password").value,
        role: $("new-user-role").value,
      }),
    });
    $("create-user-form").reset();
    await refreshUsers();
    await refreshAdminChats();
  } catch (error) {
    $("admin-error").textContent = error.message;
  }
});

$("new-chat").addEventListener("click", createSession);

$("messages").addEventListener("wheel", stopStreamingAutoFollow, { passive: true });
$("messages").addEventListener("touchstart", stopStreamingAutoFollow, { passive: true });
$("messages").addEventListener("pointerdown", stopStreamingAutoFollow);

$("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("message-input");
  const content = input.value.trim();
  if (!content || state.busy) return;
  input.value = "";
  resizeMessageInput();
  await sendMessage(content);
});

$("message-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});

$("message-input").addEventListener("input", () => {
  resizeMessageInput();
});

$("file-input").addEventListener("change", (event) => {
  uploadSelectedFiles(filesFromTransfer(event.target.files));
});

$("message-input").addEventListener("paste", (event) => {
  const files = filesFromTransfer(event.clipboardData?.files);
  if (!files.length) return;
  event.preventDefault();
  uploadSelectedFiles(files);
});

$("composer").addEventListener("dragover", (event) => {
  if (!event.dataTransfer?.types?.includes("Files")) return;
  event.preventDefault();
});

$("composer").addEventListener("drop", (event) => {
  const files = filesFromTransfer(event.dataTransfer?.files);
  if (!files.length) return;
  event.preventDefault();
  uploadSelectedFiles(files);
});

$("upload-menu-trigger").addEventListener("click", (event) => {
  event.stopPropagation();
  toggleUploadMenu();
});

$("add-local-file").addEventListener("click", () => {
  closeUploadMenu();
  $("file-input").click();
});

$("voice-input").addEventListener("click", () => {
  if (state.voiceRecording) {
    stopVoiceInput();
    return;
  }
  startVoiceInput();
});

$("file-preview-close").addEventListener("click", closeAttachmentPreview);

$("file-preview-modal").addEventListener("click", (event) => {
  if (event.target === $("file-preview-modal")) closeAttachmentPreview();
});

$("theme-toggle").addEventListener("click", () => {
  applyTheme(state.theme === "light" ? "dark" : "light");
});

$("sidebar-collapse").addEventListener("click", () => {
  applySidebarCollapsed(true);
});

$("sidebar-logo-trigger").addEventListener("click", () => {
  if (state.sidebarCollapsed) applySidebarCollapsed(false);
});

document.addEventListener("click", (event) => {
  if (event.target.closest(".session-actions")) {
    closeUploadMenu();
    return;
  }
  if (!event.target.closest(".composer-upload")) closeUploadMenu();
  closeSessionMenu(true);
});

document.addEventListener("keydown", (event) => {
  if (["PageUp", "PageDown", "Home", "End", "ArrowUp", "ArrowDown", " "].includes(event.key)) {
    stopStreamingAutoFollow();
  }
  if (event.key === "Escape") {
    closeAttachmentPreview();
    closeUploadMenu();
    closeSessionMenu(true);
  }
});

setupSidebarState();
setVoiceRecording(false);
boot();
