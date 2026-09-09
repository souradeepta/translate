-- Book project state schema v1. Applied transactionally by BookStore.
CREATE TABLE IF NOT EXISTS project_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS units (
  block_id TEXT PRIMARY KEY,
  source_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  selected_attempt_id INTEGER,
  approved_attempt_id INTEGER,
  lease_owner TEXT,
  lease_expires_at TEXT,
  updated_at TEXT NOT NULL,
  approval_needs_revalidation INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  block_id TEXT NOT NULL REFERENCES units(block_id), stage TEXT NOT NULL,
  source_hash TEXT NOT NULL, context_hash TEXT NOT NULL DEFAULT '', config_hash TEXT NOT NULL,
  model TEXT, model_revision TEXT, prompt_version TEXT, target_text TEXT NOT NULL,
  raw_response TEXT, status TEXT NOT NULL, error_type TEXT, error_message TEXT,
  started_at TEXT, finished_at TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempt_dependencies (
  attempt_id INTEGER NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
  dependency_kind TEXT NOT NULL, dependency_key TEXT NOT NULL, dependency_hash TEXT NOT NULL,
  PRIMARY KEY (attempt_id, dependency_kind, dependency_key)
);
CREATE TABLE IF NOT EXISTS context_assets (
  kind TEXT NOT NULL, asset_key TEXT NOT NULL, value_json TEXT NOT NULL,
  locked INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL DEFAULT 'machine', updated_at TEXT NOT NULL,
  PRIMARY KEY (kind, asset_key)
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, stage TEXT NOT NULL, config_hash TEXT NOT NULL, status TEXT NOT NULL,
  started_at TEXT NOT NULL, finished_at TEXT, summary_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS qa_findings (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT REFERENCES runs(run_id), rule TEXT NOT NULL,
  severity TEXT NOT NULL, block_ids_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL
);
