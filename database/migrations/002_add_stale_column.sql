-- =============================================
-- Add is_stale column to chunk tables
-- Run this migration to enable stale chunk tracking
-- =============================================

-- Add is_stale column to stripe_docs
ALTER TABLE stripe_docs 
ADD COLUMN IF NOT EXISTS is_stale BOOLEAN DEFAULT FALSE;

-- Add is_stale column to stripe_github_issues
ALTER TABLE stripe_github_issues 
ADD COLUMN IF NOT EXISTS is_stale BOOLEAN DEFAULT FALSE;

-- Add is_stale column to stripe_stackoverflow
ALTER TABLE stripe_stackoverflow 
ADD COLUMN IF NOT EXISTS is_stale BOOLEAN DEFAULT FALSE;

-- Create indexes for efficient stale filtering
CREATE INDEX IF NOT EXISTS stripe_docs_is_stale_idx ON stripe_docs (is_stale);
CREATE INDEX IF NOT EXISTS stripe_github_issues_is_stale_idx ON stripe_github_issues (is_stale);
CREATE INDEX IF NOT EXISTS stripe_stackoverflow_is_stale_idx ON stripe_stackoverflow (is_stale);
