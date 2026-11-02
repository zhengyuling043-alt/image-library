"use strict";
const $ = (selector) => document.querySelector(selector);
const token = $('meta[name="app-token"]').content;
const state = {items: [], selected: new Set(), groups: null, query: null, queryURL: null, busy: false, imports: [], visibleLimit: 60, recoverable: []};
const labels = {library: "素材库", search: "以图搜图", groups: "重复组与对比"};
let toastTimer;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function notify(message, error = false) {
  const element = $("#toast");
  element.textContent = message; element.className = "toast" + (error ? " error" : ""); element.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => element.hidden = true, error ? 11000 : 5000);
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json", "X-App-Token": token}, body: JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败，请重试");
  return data;
}
function setBusy(busy, text = "正在处理…") {
  state.busy = busy; $("#task-banner").hidden = !busy; $("#task-text").textContent = text;
  document.querySelectorAll("button, input, select").forEach(element => {
    if (element.matches(".nav-item, [data-goto], .icon-button, [data-mode]")) return;
    if (busy) {element.dataset.previousDisabled = String(element.disabled); element.disabled = true;}
    else if ("previousDisabled" in element.dataset) {element.disabled = element.dataset.previousDisabled === "true"; delete element.dataset.previousDisabled;}
  });
  if (!busy) updateSelection();
}
async function pollJob(identifier) {
  while (true) {
    const job = await api("/api/jobs/" + identifier);
    if (job.state === "done") return job.result;
    if (job.state === "failed") throw new Error(job.error || "任务失败");
    await new Promise(resolve => setTimeout(resolve, 600));
  }
}
async function task(path, body, message) {
  const response = await api(path, body);
  $("#task-text").textContent = message;
  return pollJob(response.job);
}
async function guarded(message, work) {
  if (state.busy) return notify("已有任务正在运行，请稍候");
  setBusy(true, message);
  const status = $("#operation-status"); status.hidden = true; status.textContent = "";
  try {await work();} catch (error) {notify(error.message, true);}
  finally {setBusy(false);}
}
function changePage(page) {
  document.querySelectorAll(".page").forEach(element => element.hidden = element.id !== "page-" + page);
  document.querySelectorAll(".nav-item").forEach(element => element.classList.toggle("active", element.dataset.page === page));
  $("#breadcrumb").textContent = labels[page];
  window.scrollTo({top: 0});
}
document.querySelectorAll("[data-page]").forEach(element => element.addEventListener("click", () => changePage(element.dataset.page)));
document.querySelectorAll("[data-goto]").forEach(element => element.addEventListener("click", () => changePage(element.dataset.goto)));

function imageURL(name) {
  const item = state.items.find(item => item.name === name);
  return "/api/thumbnail?name=" + encodeURIComponent(name) + "&v=" + encodeURIComponent(item?.version || "");
}
function sizeText(bytes) {return bytes >= 1024 * 1024 ? (bytes / 1024 / 1024).toFixed(1) + " MB" : Math.round(bytes / 1024) + " KB";}
function card(item, {selectable = false, score, group, badge, reason, category, interpretation} = {}) {
  const article = node("article", "image-card");
  article.classList.toggle("selected", state.selected.has(item.name) && selectable);
  const visual = node("div", "image-visual");
  const image = node("img"); image.src = imageURL(item.name); image.alt = item.name; image.loading = "lazy";
  image.tabIndex = 0; image.setAttribute("role", "button"); image.setAttribute("aria-label", "查看 " + item.name);
  image.addEventListener("click", () => showDetail(item, group));
  image.addEventListener("keydown", event => {if (event.key === "Enter") showDetail(item, group);});
  visual.append(image);
  if (selectable) {
    const label = node("label", "card-check"), checkbox = node("input");
    checkbox.type = "checkbox"; checkbox.checked = state.selected.has(item.name); checkbox.setAttribute("aria-label", "选择 " + item.name);
    checkbox.addEventListener("change", () => {
      checkbox.checked ? state.selected.add(item.name) : state.selected.delete(item.name);
      article.classList.toggle("selected", checkbox.checked); updateSelection();
    }); label.append(checkbox); visual.append(label);
  }
  if (score !== undefined) visual.append(node("span", "card-badge score", score.toFixed(4)));
  else if (group) visual.append(node("span", "card-badge", group.members.length + " 个版本"));
  else if (badge) visual.append(node("span", "card-badge", badge));
  const info = node("div", "image-info"), name = node("p", "image-name", item.name); name.title = item.name;
  const meta = node("div", "image-meta");
  meta.append(node("span", "", item.width ? `${item.width} × ${item.height}` : "无法读取"));
  meta.append(node("span", "status-dot " + item.status, {ready: "● 已入库", pending: "● 待更新", failed: "● 读取失败"}[item.status] || ""));
  info.append(name, meta);
  if (reason) info.append(node("p", "match-explanation", `${category || "依据"} · ${reason}`));
  if (interpretation) info.append(node("p", "match-interpretation", interpretation));
  article.append(visual, info); return article;
}
function updateSelection() {
  $("#export").textContent = `导出清单 (${state.selected.size})`;
  $("#export").disabled = state.busy || state.selected.size === 0;
}
function visibleItems() {
  const filter = $("#filename-filter").value.toLowerCase().trim();
  let items = state.items.filter(item => item.name.toLowerCase().includes(filter));
  const groups = new Map();
  if ($("#collapse-groups").checked && state.groups) {
    const hidden = new Set();
    for (const group of state.groups.groups) {
      const matches = group.members.filter(name => items.some(item => item.name === name));
      if (!matches.length) continue;
      const representative = matches.includes(group.suggested_keep) ? group.suggested_keep : matches[0];
      groups.set(representative, group);
      matches.filter(name => name !== representative).forEach(name => hidden.add(name));
    }
    items = items.filter(item => !hidden.has(item.name));
  }
  const order = $("#sort-order").value;
  items.sort((a, b) => order === "pixels" ? b.width * b.height - a.width * a.height : order === "recent" ? b.version - a.version : a.name.localeCompare(b.name));
  return {items, groups};
}
function renderLibrary() {
  const {items, groups} = visibleItems();
  const visible = items.slice(0, state.visibleLimit);
  $("#library-grid").replaceChildren(...visible.map(item => card(item, {selectable: true, group: groups.get(item.name)})));
  $("#load-more").hidden = visible.length >= items.length;
  $("#library-empty").hidden = items.length > 0;
  $("#library-empty h3").textContent = state.items.length ? "没有匹配的素材" : "这里还没有素材";
  $("#library-empty p").textContent = state.items.length ? "试试其他文件名，或清空搜索条件。" : "导入一些图片，开始整理。";
  $("#grid-summary").textContent = `显示 ${visible.length} / ${items.length} 项 · 共 ${state.items.length} 张素材`;
  $("#collapse-note").hidden = !$("#collapse-groups").checked;
  updateSelection();
}
async function refreshLibrary() {
  const data = await api("/api/library"); state.items = data.items;
  state.recoverable = data.recoverable || [];
  $("#restore-last").hidden = !state.recoverable.length;
  const names = new Set(data.items.map(item => item.name));
  state.selected = new Set([...state.selected].filter(name => names.has(name)));
  $("#stat-total").textContent = data.total; $("#stat-ready").textContent = data.ready; $("#stat-pending").textContent = data.pending;
  $("#collection-count").textContent = data.total; $("#nav-count").textContent = data.total;
  renderLibrary(); return data;
}
$("#filename-filter").addEventListener("input", () => {state.visibleLimit = 60; renderLibrary();});
$("#sort-order").addEventListener("change", () => {state.visibleLimit = 60; renderLibrary();});
$("#load-more").addEventListener("click", () => {state.visibleLimit += 60; renderLibrary();});
$("#restore-last").addEventListener("click", () => guarded("正在恢复最近一次清理…", async () => {
  const result = await task("/api/restore", {batch: state.recoverable[0].id}, "正在恢复图片与索引…");
  invalidateResults(); await refreshLibrary(); notify(`已恢复 ${result.restored.length} 张图片。`);
}));
$("#select-all").addEventListener("click", () => {
  const names = visibleItems().items.map(item => item.name);
  const all = names.every(name => state.selected.has(name));
  names.forEach(name => all ? state.selected.delete(name) : state.selected.add(name)); renderLibrary();
});
$("#collapse-groups").addEventListener("change", () => {
  if (!$("#collapse-groups").checked) return renderLibrary();
  guarded("正在整理重复组视图…", async () => {
    try {state.groups = await task("/api/groups", {preset: "balanced", similarity: .86, distance: 8}, "正在比较组内版本…");}
    catch (error) {$("#collapse-groups").checked = false; throw error;}
    finally {renderLibrary();}
  });
});
function invalidateResults() {
  state.groups = null; $("#collapse-groups").checked = false; lastSearch = null;
  $("#group-list").replaceChildren(); $("#group-summary").textContent = "素材已更新，请重新分析。"; $("#group-empty").hidden = false;
  $("#compare-left").replaceChildren(node("p", "muted", "素材已更新，请重新对比。")); $("#compare-right").replaceChildren(node("p", "muted", "素材已更新，请重新对比。"));
  $("#search-grid").replaceChildren(); $("#search-empty").hidden = false; $("#search-summary").textContent = "最多展示 20 张";
  $("#search-compare-left").replaceChildren(node("p", "muted", "素材已更新，请重新对比。")); $("#search-compare-right").replaceChildren(node("p", "muted", "素材已更新，请重新对比。"));
  resetSearchView();
}
$("#update-index").addEventListener("click", () => guarded("正在检查素材变化，首次处理可能需要稍候…", async () => {
  const result = await task("/api/update", {}, "正在更新索引，请保持页面开启…");
  invalidateResults(); await refreshLibrary();
  const c = result.counts; notify(`更新完成：新增 ${c.added}，修改 ${c.modified}，复用 ${c.reused}，删除 ${c.deleted}，失败 ${c.failed}。`, c.failed > 0);
}));
$("#export").addEventListener("click", () => guarded("正在生成素材清单…", async () => {
  const response = await fetch("/api/export", {method: "POST", headers: {"Content-Type": "application/json", "X-App-Token": token}, body: JSON.stringify({names: [...state.selected]})});
  if (!response.ok) throw new Error((await response.json()).error);
  const url = URL.createObjectURL(await response.blob()), link = node("a");
  link.href = url; link.download = "素材清单.csv"; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  notify(`已导出 ${state.selected.size} 张素材的信息清单。`);
}));

function showDetail(item, group) {
  $("#detail-title").textContent = group ? `候选重复组 · ${group.members.length} 个版本` : item.name;
  const content = $("#detail-content"); content.replaceChildren();
  if (group) {
    const grid = node("div", "group-images");
    group.members.forEach(name => {const member = state.items.find(i => i.name === name); if (member) grid.append(card(member, {badge: name === group.suggested_keep ? "像素数优先" : "版本"}));});
    content.append(grid, node("p", "detail-meta", "这里只合并显示，所有图片都仍然保留。导出组内其他版本时可关闭合并视图后选择。"));
  } else {
    const image = node("img", "detail-image"); image.src = imageURL(item.name); image.alt = item.name;
    content.append(image, node("p", "detail-meta", `${item.width} × ${item.height} · ${sizeText(item.bytes)} · 此处显示缩略预览`));
  }
  if (!$("#image-dialog").open) $("#image-dialog").showModal();
}
$("#detail-close").addEventListener("click", () => $("#image-dialog").close());
async function fileData(file) {
  if (file.size > 20 * 1024 * 1024) throw new Error("单张图片不能超过 20 MB");
  return new Promise((resolve, reject) => {const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(",")[1]); reader.onerror = () => reject(new Error("文件读取失败")); reader.readAsDataURL(file);});
}
function chooseQuery(file) {
  if (!file || state.busy) return;
  if (file.size > 20 * 1024 * 1024) return notify("单张图片不能超过 20 MB", true);
  state.query = file; if (state.queryURL) URL.revokeObjectURL(state.queryURL); state.queryURL = URL.createObjectURL(file);
  lastSearch = null;
  resetSearchView();
  $("#query-preview").src = state.queryURL; $("#query-preview").hidden = false; $("#query-placeholder").hidden = true;
  $("#query-name").textContent = file.name; $("#search-submit").disabled = false;
  $("#search-grid").replaceChildren(); $("#search-empty").hidden = false; $("#search-summary").textContent = "点击查找以获取新结果";
}
$("#query-file").addEventListener("change", event => chooseQuery(event.target.files[0]));
$("#query-drop").addEventListener("dragover", event => {event.preventDefault(); $("#query-drop").classList.add("dragover");});
$("#query-drop").addEventListener("dragleave", () => $("#query-drop").classList.remove("dragover"));
$("#query-drop").addEventListener("drop", event => {event.preventDefault(); $("#query-drop").classList.remove("dragover"); chooseQuery(event.dataTransfer.files[0]);});
$("#search-submit").addEventListener("click", () => guarded("正在查找相似素材，首次加载模型可能需要稍候…", async () => {
  const data = await fileData(state.query);
  lastSearch = await task("/api/search", {data}, "正在计算查询图片的特征并搜索…");
  await refreshLibrary();
  renderSearch();
  if (lastSearch.skipped) notify(`跳过 ${lastSearch.skipped} 张已变化素材，请更新索引。`);
}));
$("#text-query").addEventListener("input", event => {$("#text-search-submit").disabled = !event.target.value.trim();});
$("#text-search-submit").addEventListener("click", () => guarded("正在执行以文搜图…", async () => {
  const text = $("#text-query").value.trim();
  lastSearch = await task("/api/search-text", {text}, "正在编码文本并搜索共享向量空间…");
  await refreshLibrary(); renderSearch();
}));

let lastSearch = null, searchMode = "single";
function resetSearchView() {
  $("#search-notice").hidden = true;
  $("#search-grid").replaceChildren(); $("#search-grid").hidden = searchMode !== "single";
  $("#search-compare").hidden = true;
  $("#search-compare-left").replaceChildren(); $("#search-compare-right").replaceChildren();
  $("#search-empty").hidden = false;
  $("#search-empty h3").textContent = "从一张图片开始";
  $("#search-empty p").textContent = "选择参考图并点击查找，获取当前图片的结果。";
}
function searchCard(row) {const item = state.items.find(i => i.name === row.name); return item ? card(item, {score: row.score, reason: row.reason, category: row.category, interpretation: row.interpretation}) : node("span");}
function clampThreshold(value) {return Math.min(1, Math.max(0, Number(value) || 0));}
function renderSearch() {
  if (!lastSearch) return resetSearchView();
  if (searchMode === "single") {
    const threshold = clampThreshold($("#search-sim").value);
    const matches = lastSearch.results.filter(row => row.score >= threshold);
    const rows = matches.slice(0, 20);
    $("#search-grid").hidden = false; $("#search-compare").hidden = true;
    $("#search-grid").replaceChildren(...rows.map(searchCard));
    $("#search-empty").hidden = rows.length > 0;
    $("#search-summary").textContent = `检索 ${lastSearch.searched} 张 · 命中 ${matches.length} 张 · 展示前 ${rows.length} 张`;
    renderSearchWarnings();
    if (!rows.length) {$("#search-empty h3").textContent = "当前阈值下没有结果"; $("#search-empty p").textContent = "可以降低相似度下限后再查看。";}
  } else {
    $("#search-grid").hidden = true; $("#search-compare").hidden = false; $("#search-empty").hidden = true;
    for (const side of ["left", "right"]) {
      const threshold = clampThreshold($("#search-" + side + "-sim").value);
      const matches = lastSearch.results.filter(row => row.score >= threshold);
      const rows = matches.slice(0, 20);
      const container = $("#search-compare-" + side);
      container.replaceChildren(node("p", "result-summary", `相似度 ≥ ${threshold} · 命中 ${matches.length} 张 · 展示前 ${rows.length} 张`), ...rows.map(searchCard));
      if (!rows.length) container.append(node("p", "muted", "该阈值下没有结果。"));
    }
    $("#search-summary").textContent = `检索 ${lastSearch.searched} 张 · 双阈值对比`;
    renderSearchWarnings();
  }
}
function renderSearchWarnings() {
  const failures = Object.entries(lastSearch.failures || {});
  const notice = $("#search-notice");
  if (!failures.length) {notice.hidden = true; notice.textContent = ""; return;}
  notice.hidden = false;
  notice.textContent = `以下 ${failures.length} 张未入库：` + failures.map(([name, reason]) => `${name}：${reason}`).join("；");
}
$("#search-sim").addEventListener("input", renderSearch);
$("#search-left-sim").addEventListener("input", renderSearch);
$("#search-right-sim").addEventListener("input", renderSearch);
document.querySelectorAll("[data-search-mode]").forEach(button => button.addEventListener("click", () => {
  searchMode = button.dataset.searchMode;
  document.querySelectorAll("[data-search-mode]").forEach(item => item.classList.toggle("active", item === button));
  $("#search-thresholds").hidden = searchMode !== "compare";
  renderSearch();
}));

const thresholdPresets = {strict: [.94, 6], balanced: [.86, 8], broad: [.78, 28]};
$("#group-preset").addEventListener("change", event => {
  const values = thresholdPresets[event.target.value];
  if (values) {$("#group-sim").value = values[0]; $("#group-dist").value = values[1];}
});
for (const id of ["group-sim", "group-dist"]) $("#" + id).addEventListener("input", () => {$("#group-preset").value = "custom";});

function parameters(prefix) {
  const sim = $("#" + prefix + "-sim"), dist = $("#" + prefix + "-dist");
  if (!sim.reportValidity() || !dist.reportValidity() || sim.value === "" || dist.value === "") throw new Error("请填写有效的阈值");
  return {similarity: Number(sim.value), distance: Number(dist.value)};
}
function groupCard(group, allowCleanup = true) {
  const box = node("article", "group-card"), header = node("div", "group-header");
  header.append(node("h3", "", `组 ${group.id} · ${group.members.length} 个版本`), node("span", "", group.category === "exact" ? "完全重复" : "含近重复候选"));
  const keeper = node("select"); keeper.setAttribute("aria-label", `组 ${group.id} 保留版本`);
  group.members.forEach(name => {const option = node("option", "", name); option.value = name; keeper.append(option);});
  keeper.value = group.suggested_keep;
  if (group.members.length > 1 && allowCleanup) {
    const button = node("button", "cleanup-button", "清理其余版本");
    button.addEventListener("click", () => cleanupGroup(group, keeper.value));
    header.append(button);
  }
  const images = node("div", "group-images");
  group.members.slice(0, 20).forEach(name => {
    const item = state.items.find(i => i.name === name);
    if (item) {
      const score = group.quality_scores?.[name];
      const badge = name === group.suggested_keep ? `推荐保留 · ${score?.toFixed(1) ?? ""}` :
        (score === undefined ? undefined : `质量 ${score.toFixed(1)}`);
      images.append(card(item, {badge}));
    }
  });
  if (group.members.length > 20) {
    const more = node("button", "secondary", `展开其余 ${group.members.length - 20} 个版本`);
    more.addEventListener("click", () => {group.members.slice(20).forEach(name => {const item = state.items.find(i => i.name === name); if (item) images.append(card(item));}); more.remove();});
    images.append(more);
  }
  const evidence = node("details", "evidence"); evidence.append(node("summary", "", "查看判定依据"));
  group.pairs.forEach(pair => evidence.append(node("p", "", `${pair.a} ↔ ${pair.b}：${pair.reason || (pair.category === "exact" ? "文件内容相同" : "近重复候选")}；相似度 ${pair.cosine.toFixed(4)}；哈希距离 ${pair.dhash_distance}`)));
  const keep = group.suggested_keep, keepScore = group.quality_scores?.[keep];
  const kd = group.keeper_detail || {};
  const keepMetrics = [
    `质量分 ${keepScore?.toFixed(2) ?? "—"}`,
    kd.width && kd.height ? `${kd.width}×${kd.height}` : "",
    kd.sharpness != null ? `清晰度 ${kd.sharpness.toFixed(2)}` : "",
    kd.brightness != null ? `亮度 ${kd.brightness.toFixed(2)}` : "",
  ].filter(Boolean).join("；");
  const keepInfo = node("p", "group-reason",
    `推荐保留：${keep}（${keepMetrics}，为组内综合质量最高）；组代表图：${group.representative || keep}。` +
    `选择依据：综合质量分优先，最终由人工确认。`);
  box.append(header, images, keepInfo);
  if (allowCleanup) {const label = node("label", "keeper-label", "本次保留："); label.append(keeper); box.append(label);}
  if (group.pair_count > group.pairs.length) evidence.append(node("p", "", `共 ${group.pair_count} 对关系，此处展示前 ${group.pairs.length} 对依据。`));
  box.append(evidence); return box;
}
function renderGroupList(container, groups, allowCleanup = true) {
  let shown = 0;
  const more = node("button", "secondary", "加载更多重复组");
  function append() {
    more.remove(); groups.slice(shown, shown + 12).forEach(group => container.append(groupCard(group, allowCleanup)));
    shown += 12; if (shown < groups.length) container.append(more);
  }
  more.addEventListener("click", append); append();
}
function summaryText(result) {
  const backend = result.candidate_backend === "faiss-range-search" ? "FAISS范围召回" : "NumPy分块召回";
  const candidates = Number.isFinite(result.candidate_pair_count) ? ` · ${result.candidate_pair_count} 个向量候选对` : "";
  return `${result.total} 张参与分析 · ${result.groups.length} 个候选组 · ${result.singletons.length} 张未归组 · ${Object.keys(result.skipped || {}).length} 张跳过 · ${backend}${candidates}`;
}
function skippedDetails(result) {
  const details = node("details", "evidence");
  const entries = Object.entries(result.skipped || {});
  details.hidden = !entries.length;
  details.append(node("summary", "", `查看 ${entries.length} 张未参与分析的图片及原因`));
  entries.forEach(([name, reason]) => details.append(node("p", "", `${name}：${reason}`)));
  return details;
}
async function runGroups(parameters) {
  const result = await task("/api/groups", parameters, "正在检查组内相似关系…");
  state.groups = result;
  await refreshLibrary();
  $("#group-summary").textContent = summaryText(result);
  $("#group-list").replaceChildren(skippedDetails(result)); renderGroupList($("#group-list"), result.groups); $("#group-empty").hidden = result.groups.length > 0;
  if (!result.groups.length) {$("#group-empty h3").textContent = "当前条件下没有发现重复组"; $("#group-empty p").textContent = "可以对比其他阈值，或导入已知重复的测试图片。";}
  return result;
}
function cleanupGroup(group, keep) {
  if (state.busy) return notify("已有任务正在运行，请稍候");
  const extra = group.members.filter(name => name !== keep);
  const message = `保留「${keep}」，将以下 ${extra.length} 个版本移入本地回收区，可从素材库恢复：\n\n${extra.join("\n")}\n\n确定清理吗？`;
  if (!window.confirm(message)) return;
  guarded("正在清理多余版本…", async () => {
    const result = await task("/api/cleanup", {review: group.review, signature: group.signature, keep, names: extra}, "正在移入回收区并同步索引…");
    invalidateResults(); await refreshLibrary();
    const rejected = result.rejected.length;
    notify(`已将 ${result.deleted.length} 个版本移入回收区，可在素材库恢复。`, rejected > 0);
    if (result.deleted.length) await runGroups(parameters("group"));
  });
}
$("#group-run").addEventListener("click", () => guarded("正在比较图片并生成候选组…", async () => { await runGroups(parameters("group")); }));
document.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll("[data-mode]").forEach(item => item.classList.toggle("active", item === button));
  $("#group-view").hidden = button.dataset.mode !== "groups"; $("#compare-view").hidden = button.dataset.mode !== "compare";
}));
$("#compare-run").addEventListener("click", () => guarded("正在用两套条件比较同一素材库…", async () => {
  const result = await task("/api/compare", {left: parameters("left"), right: parameters("right")}, "正在生成并排对比结果…");
  await refreshLibrary();
  for (const side of ["left", "right"]) {
    const data = result[side], container = $("#compare-" + side);
    container.replaceChildren(node("p", "result-summary", summaryText(data)), skippedDetails(data)); renderGroupList(container, data.groups, false);
    if (!data.groups.length) container.append(node("p", "muted", "这套条件下没有候选重复组。"));
  }
}));

$("#import-open").addEventListener("click", () => $("#import-dialog").showModal());
$("#import-close").addEventListener("click", () => {if (!state.busy) $("#import-dialog").close(); else notify("导入任务仍在运行，请稍候");});
$("#import-dialog").addEventListener("cancel", event => {if (state.busy) event.preventDefault();});
$("#import-files").addEventListener("change", event => {
  state.imports = [...event.target.files].map(file => ({file, state: "等待导入"})); renderImports();
  $("#import-start").disabled = !state.imports.length;
});
function renderImports() {
  $("#import-list").replaceChildren(...state.imports.map(item => {
    const row = node("div", "import-row" + (item.error ? " error" : ""));
    row.append(node("span", "", item.file.name), node("span", "", item.error ? `${item.state}：${item.error}` : item.state)); row.title = item.error || item.name || item.file.name; return row;
  }));
}
$("#import-start").addEventListener("click", () => guarded("正在导入图片…", async () => {
  for (const [index, item] of state.imports.entries()) {
    if (item.name) continue;
    $("#task-text").textContent = `正在导入 ${index + 1}/${state.imports.length}：${item.file.name}`;
    try {
      const response = await api("/api/import", {name: item.file.name, data: await fileData(item.file)});
      item.name = response.name; item.state = "已保存 · 等待入库"; item.error = null;
    } catch (error) {item.state = "导入失败"; item.error = error.message;}
    renderImports();
  }
  invalidateResults(); await refreshLibrary();
  if (state.imports.some(item => item.name)) {
    try {
      const result = await task("/api/update", {}, "图片已保存，正在建立索引…");
      state.imports.forEach(item => {if (item.name) {item.error = result.failures[item.name]; item.state = item.error ? "入库失败" : "入库成功";}});
    } catch (error) {
      state.imports.forEach(item => {if (item.name) {item.state = "待更新索引"; item.error = error.message;}});
      throw error;
    } finally {renderImports(); await refreshLibrary();}
  }
  const failed = state.imports.filter(item => item.error).length;
  notify(failed ? `${failed} 张图片未完成，请查看列表中的失败原因。` : "全部图片已导入并建立索引。", failed > 0);
}));

refreshLibrary().then(async data => {
  if (data.active_job) await guarded("上次任务仍在运行，正在恢复状态…", async () => {await pollJob(data.active_job); await refreshLibrary(); notify("任务已完成，可重新查看结果。");});
}).catch(error => notify("加载素材库失败：" + error.message, true));
