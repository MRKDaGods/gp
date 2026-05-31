# Kaggle Workflow — MTMC Tracker

Pipeline chain, auth, disk hygiene, and the safety rules around `kaggle kernels push`. Read before pushing any kernel.

---

## Pipeline Chain

- **10a** (GPU): stages 0–2 — ingestion, detection+tracking, feature extraction
- **10b** (CPU): stage 3 — FAISS indexing
- **10c** (CPU): stages 4–5 — association, evaluation

Backend/frontend integration is local orchestration only; GPU-heavy stages still run on Kaggle.

### Push
```pwsh
kaggle kernels push -p notebooks/kaggle/10X_stagesNN/
```

### Logs
```pwsh
python scripts/kaggle_logs.py <kernel_slug> --tail N
```

### Auth tokens (`~/.kaggle/`)
Per-account token files live in `~/.kaggle/` (never committed). The canonical account→token-file
map is `ACCOUNT_TOKENS` in `scripts/dump_kaggle_kernel_summaries.py`. Tooling selects an account by
copying its token file over `~/.kaggle/kaggle.json` before each CLI call (hot-swap).

| Account | Token file(s) (first match wins) | Owner |
|---------|----------------------------------|-------|
| gumfreddy | `gumfreddy_access_token` | abdo |
| mrkdagods | `mrkdagods_access_token`, `MRKDaGods__access_token` | mrk |
| ali369 | `ali369_access_token`, `ali_369_access_token` | lolo |
| yahiaakhalafallah | `yahiaakhalafallah_access_token` | yahia |

Current default active account: **gumfreddy**. As of 2026-05-31 all four token files are present
in `~/.kaggle/`. Max **2 concurrent GPU sessions per account** — the multiple accounts parallelize
GPU work across slots.

---

## Push Safety Rules (CRITICAL)

1. **NEVER push a kernel more than once without confirming the previous version is fully running or complete.** Rapid re-pushes create duplicate GPU sessions that consume both slots and block all other work.
2. After every push, check for warning lines like `The following are not valid dataset sources` — these indicate the run started but with missing inputs. **Immediately cancel** the bad run via `kaggle kernels cancel <owner/slug>` before attempting a fix-and-repush.
3. If `kaggle kernels cancel` CLI command fails or is unavailable (older CLI versions lack the subcommand), **post the kernel URL to the user, then keep polling `kaggle kernels status <slug>` every ~60s in a loop** until the status reaches `cancelled` / `error` / `complete`. Do NOT stop the workflow — once cancellation is confirmed, resume planned work automatically.
4. **Kaggle allows a maximum of 2 concurrent GPU sessions per account** — always check active sessions before pushing a GPU-enabled notebook.
5. When iterating on `kernel-metadata.json` fixes, validate metadata locally first, then push **once**.

---

## Disk Hygiene (CRITICAL — disk is tight)

After every `kaggle kernels output` download, immediately delete useless artifacts:

- **DELETE `last.pth`** (never useful — only `best_mAP.pth` / `best_R1.pth` are kept)
- For FAILed runs, **DELETE all `.pth` files**; keep only `eval_results.json`, `recipe.json`, `train_log.json`, summary JSONs
- **DELETE empty/0-byte log files**
- Keep `best_mAP.pth` only if the run has plausible ensemble value (close to baseline R1 even on FAIL)

Always run before/after:
```pwsh
Get-ChildItem | Measure-Object -Sum Length
```
…and report GB reclaimed.

Old `tmp_*_outputs/` directories from completed verdict runs should be **pruned, not accumulated**.

---

## Session Lifecycle (CRITICAL — never exit a turn waiting)

- **NEVER end a turn without queueing the next action** — every turn ends with a tool call (sleep/poll/subagent) or `vscode_askQuestions` if blocked.
- **NEVER use `mode=async` for `Start-Sleep` waits.** Always use `mode=sync` with a generous `timeout` so the sleep + poll completes within the turn.
- If the user says "monitor", "check back in N hours", "wait Xh" — execute `Start-Sleep -Seconds <N>` synchronously in `mode=sync`. Do NOT exit the turn waiting for an async system notification.
- Between subagent invocations, immediately start the next sleep/poll or ask the user via `vscode_askQuestions` — never end the conversation in an idle state.
