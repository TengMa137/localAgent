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
  files: { uploads: [], reports: [] },
  pendingAttachments: [],
  openSessionMenuId: null,
  theme: storageGet("localagent_theme", "dark"),
  sidebarCollapsed: storageGet("localagent_sidebar_collapsed") === "1",
  runningPollTimer: null,
  activeGenerationPollTimer: null,
  settingsChatUserId: null,
  busy: false,
};

const $ = (id) => document.getElementById(id);

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
  const response = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": csrfToken(),
    },
    body: JSON.stringify(payload),
  });
  if (response.status === 401) {
    showLogin();
    throw new Error("Not authenticated");
  }
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || data.error || "Request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalData = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
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

  if (!finalData) {
    return null;
  }
  return finalData;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text || "";
  return div.innerHTML;
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

function showLogin(message = "") {
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
  const row = document.createElement("article");
  row.className = `message ${message.role}`;
  row.dataset.messageId = message.id || "";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const isRunning = message.role === "assistant" && message.metadata?.status === "running" && !message.content;
  if (isRunning) {
    bubble.classList.add("streaming");
    bubble.textContent = "Working...";
  } else {
    bubble.innerHTML = formatMessage(message.content);
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
    renderMessages(data.messages || []);
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
  const old = row.querySelector(".activity");
  const wasOpen = old?.open ?? metadata.status === "running";
  if (old) old.remove();
  const logs = metadata.turn_logs || [];
  const traceEvents = metadata.trace_events || [];
  const visibleLogs = logs.filter((log) => log.objective || log.summary || log.error);
  const visibleTrace = traceEvents.filter((event) => {
    return ["model_request", "model_tools", "tool_call", "tool_result", "tool_call_start"].includes(event.kind);
  });
  if (!visibleLogs.length && !visibleTrace.length) return;

  const details = document.createElement("details");
  details.className = "activity";
  details.open = wasOpen;
  const summary = document.createElement("summary");
  const done = visibleLogs.filter((log) => log.status === "done").length;
  const toolCalls = visibleTrace.filter((event) => event.kind === "tool_call").length;
  summary.textContent = `Activity: ${toolCalls} tool call${toolCalls === 1 ? "" : "s"}${visibleLogs.length ? `, ${done}/${visibleLogs.length} tasks` : ""}`;
  details.appendChild(summary);

  if (visibleTrace.length) {
    const trace = document.createElement("div");
    trace.className = "tool-trace";
    for (const event of visibleTrace) {
      trace.appendChild(createTraceEventElement(event));
    }
    details.appendChild(trace);
  }

  for (const log of visibleLogs) {
    const item = document.createElement("div");
    item.className = `activity-item ${log.status === "done" ? "done" : "failed"}`;
    const title = document.createElement("strong");
    title.textContent = log.objective || log.task_id || "Agent task";
    const body = document.createElement("p");
    body.textContent = log.error || log.summary || "";
    item.append(title, body);
    details.appendChild(item);
  }
  row.appendChild(details);
}

function createTraceEventElement(event) {
  const item = document.createElement("div");
  item.className = `trace-event ${event.kind}`;

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
  body.textContent = event.output || event.args || "";

  item.append(label, title);
  if (body.textContent) item.appendChild(body);
  return item;
}

function renderMessages(messages) {
  state.messages = messages;
  const el = $("messages");
  el.textContent = "";
  if (!messages.length) {
    el.appendChild(createMessageElement({
      role: "assistant",
      content: "Start a new conversation, or upload files to add local context.",
    }));
    updateRunningPoll([]);
    return;
  }
  for (const message of messages) {
    el.appendChild(createMessageElement(message));
  }
  el.scrollTop = el.scrollHeight;
  updateRunningPoll(messages);
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
      renderMessages(data.messages);
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
  $("messages").scrollTop = $("messages").scrollHeight;
  return row;
}

function updateMessageElement(row, message) {
  if (!row || !message) return;
  const bubble = row.querySelector(".bubble");
  if (bubble) {
    if (message.metadata?.status === "running" && !message.content) {
      bubble.classList.add("streaming");
      bubble.textContent = "Working...";
    } else {
      bubble.classList.remove("streaming");
      bubble.innerHTML = formatMessage(message.content || "");
    }
  }
  renderActivity(row, message.metadata || {});
}

function latestAssistantMessage(messages, afterId = 0) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === "assistant" && Number(messages[index].id || 0) > afterId) return messages[index];
  }
  return null;
}

function stopActiveGenerationPoll() {
  if (state.activeGenerationPollTimer) {
    clearInterval(state.activeGenerationPollTimer);
    state.activeGenerationPollTimer = null;
  }
}

function startActiveGenerationPoll(pendingRow, afterAssistantId = 0) {
  stopActiveGenerationPoll();
  state.activeGenerationPollTimer = setInterval(async () => {
    if (!state.activeSessionId || !document.body.contains(pendingRow)) {
      stopActiveGenerationPoll();
      return;
    }
    try {
      const data = await api(`/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}`);
      const latest = latestAssistantMessage(data.messages, afterAssistantId);
      if (!latest) return;
      updateMessageElement(pendingRow, latest);
      $("messages").scrollTop = $("messages").scrollHeight;
      if (latest.metadata?.status !== "running") {
        stopActiveGenerationPoll();
        await refreshSessions();
        await refreshFiles();
      }
    } catch (_) {
      stopActiveGenerationPoll();
    }
  }, 1000);
}

async function refreshSessions() {
  const data = await api("/api/chat/sessions");
  state.sessions = data.sessions;
  renderSessions();
}

async function createSession() {
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
  renderMessages(data.messages);
  await refreshFiles();
  renderComposerAttachments();
}

function setBusy(value) {
  state.busy = value;
  const sendButton = $("send-message");
  if (sendButton) sendButton.disabled = value;
  $("message-input").disabled = value;
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
  const previousAssistantId = Math.max(
    0,
    ...state.messages
      .filter((message) => message.role === "assistant")
      .map((message) => Number(message.id || 0))
  );
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
  startActiveGenerationPoll(pending, previousAssistantId);
  setBusy(true);
  try {
    const data = await streamApi(
      `/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/messages/stream`,
      { content },
      (event) => {
        if (event.kind === "answer_delta") {
          streamedContent += event.content || "";
          pendingBubble.classList.remove("streaming");
          pendingBubble.innerHTML = formatMessage(streamedContent);
          $("messages").scrollTop = $("messages").scrollHeight;
          return;
        }
        if (event.kind === "answer_replace") {
          streamedContent = event.content || streamedContent;
          pendingBubble.classList.remove("streaming");
          pendingBubble.innerHTML = formatMessage(streamedContent);
          $("messages").scrollTop = $("messages").scrollHeight;
          return;
        }
        pendingMetadata.trace_events.push(event);
        renderActivity(pending, pendingMetadata);
      }
    );
    if (data?.message) {
      const replacement = createMessageElement(data.message);
      pending.replaceWith(replacement);
    } else {
      await loadSession(state.activeSessionId);
    }
    stopActiveGenerationPoll();
    await refreshSessions();
    await refreshFiles();
  } catch (error) {
    await loadSession(state.activeSessionId).catch(() => {
      const bubble = pending.querySelector(".bubble");
      bubble.textContent = error.message;
    });
    startActiveGenerationPoll(pending, previousAssistantId);
  } finally {
    setBusy(false);
  }
}

async function refreshFiles({ clearStatus = true } = {}) {
  if (clearStatus) $("upload-status").textContent = "";
  if (!state.activeSessionId) return;
  try {
    const data = await api(`/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/files`);
    state.files = { uploads: data.uploads || [], reports: data.reports || [] };
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
    pre.textContent = data.content || "";
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
    reports: [
      { name: "research-planning.md", size_bytes: 2912, updated_at: now },
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
  state.files = { uploads: [], reports: [] };
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

  for (const message of data.messages) {
    const row = document.createElement("article");
    row.className = "admin-chat-message";
    const role = document.createElement("span");
    role.textContent = message.role === "user" ? "User" : "Local Agent";
    const content = document.createElement("p");
    content.textContent = message.content;
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

$("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("message-input");
  const content = input.value.trim();
  if (!content || state.busy) return;
  input.value = "";
  input.style.height = "auto";
  await sendMessage(content);
});

$("message-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});

$("message-input").addEventListener("input", () => {
  const input = $("message-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

$("file-input").addEventListener("change", (event) => {
  uploadSelectedFile(event.target.files[0]);
});

$("upload-menu-trigger").addEventListener("click", (event) => {
  event.stopPropagation();
  toggleUploadMenu();
});

$("add-local-file").addEventListener("click", () => {
  closeUploadMenu();
  $("file-input").click();
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
  if (event.key === "Escape") {
    closeAttachmentPreview();
    closeUploadMenu();
    closeSessionMenu(true);
  }
});

setupSidebarState();
boot();
