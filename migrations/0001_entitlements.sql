CREATE TABLE IF NOT EXISTS entitlements (
  session_id TEXT PRIMARY KEY,
  paid INTEGER NOT NULL DEFAULT 0,
  stripe_checkout_session_id TEXT,
  stripe_payment_intent_id TEXT,
  customer_email TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);