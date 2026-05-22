-- =============================================
-- Neon pgvector setup
-- Run this in the Neon SQL Editor
-- =============================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Stripe Documentation table
CREATE TABLE IF NOT EXISTS stripe_docs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    url TEXT,
    title TEXT DEFAULT '',
    content TEXT NOT NULL,
    embedding vector(384),
    fts_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Stripe GitHub Issues table
CREATE TABLE IF NOT EXISTS stripe_github_issues (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    url TEXT,
    title TEXT DEFAULT '',
    content TEXT NOT NULL,
    embedding vector(384),
    fts_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Stripe StackOverflow table
CREATE TABLE IF NOT EXISTS stripe_stackoverflow (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    url TEXT,
    title TEXT DEFAULT '',
    content TEXT NOT NULL,
    embedding vector(384),
    fts_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Vector similarity indexes (HNSW - better for Neon than IVFFlat)
CREATE INDEX IF NOT EXISTS stripe_docs_embedding_idx
    ON stripe_docs USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS stripe_github_issues_embedding_idx
    ON stripe_github_issues USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS stripe_stackoverflow_embedding_idx
    ON stripe_stackoverflow USING hnsw (embedding vector_cosine_ops);

-- 6. Full-text search indexes
CREATE INDEX IF NOT EXISTS stripe_docs_fts_idx ON stripe_docs USING GIN (fts_vector);
CREATE INDEX IF NOT EXISTS stripe_github_issues_fts_idx ON stripe_github_issues USING GIN (fts_vector);
CREATE INDEX IF NOT EXISTS stripe_stackoverflow_fts_idx ON stripe_stackoverflow USING GIN (fts_vector);
