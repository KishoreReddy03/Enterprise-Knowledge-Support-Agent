-- ============================================================================
-- Stripe Support Agent - Initial Database Schema
-- Migration: 001_initial_schema.sql
-- Description: Creates all core tables for the support agent system
-- ============================================================================

-- ============================================================================
-- TABLE: tickets
-- Purpose: Stores incoming customer support tickets
-- ============================================================================
CREATE TABLE tickets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    content TEXT NOT NULL,
    customer_id TEXT,
    customer_tier TEXT DEFAULT 'standard',
    complexity TEXT,  -- 'simple' | 'moderate' | 'complex'
    urgency TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'open',
    metadata JSONB DEFAULT '{}'
);

-- ============================================================================
-- TABLE: agent_responses
-- Purpose: Stores AI-generated responses to tickets with metadata
-- ============================================================================
CREATE TABLE agent_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticket_id UUID REFERENCES tickets(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    draft_reply TEXT,
    confidence_score FLOAT,
    sources_used JSONB,    -- list of source references
    agent_path JSONB,      -- which agents ran, in order
    tokens_used INTEGER,
    latency_ms INTEGER,
    was_escalated BOOLEAN DEFAULT FALSE,
    escalation_reason TEXT
);

-- ============================================================================
-- TABLE: rep_feedback
-- Purpose: Captures human rep edits and ratings for learning loop
-- ============================================================================
CREATE TABLE rep_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID REFERENCES agent_responses(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    original_reply TEXT,
    edited_reply TEXT,
    edit_type TEXT,  -- 'factual_correction' | 'tone' | 'completeness' | 'accuracy' | 'none'
    was_sent BOOLEAN DEFAULT TRUE,
    rep_rating INTEGER CHECK (rep_rating BETWEEN 1 AND 5)
);

-- ============================================================================
-- TABLE: knowledge_updates
-- Purpose: Tracks detected changes in source documentation
-- ============================================================================
CREATE TABLE knowledge_updates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    source TEXT,  -- 'stripe_docs' | 'github' | 'stackoverflow'
    content_hash TEXT UNIQUE,
    summary TEXT,
    affected_features TEXT[],
    is_breaking_change BOOLEAN DEFAULT FALSE
);

-- ============================================================================
-- TABLE: pattern_reports
-- Purpose: Stores weekly pattern analysis reports
-- ============================================================================
CREATE TABLE pattern_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    report_data JSONB,
    tickets_analyzed INTEGER,
    documentation_gaps TEXT[],
    at_risk_customers TEXT[]
);

-- ============================================================================
-- INDEXES
-- Purpose: Optimize common query patterns
-- ============================================================================
CREATE INDEX idx_tickets_customer_id ON tickets(customer_id);
CREATE INDEX idx_tickets_created_at ON tickets(created_at);
CREATE INDEX idx_agent_responses_ticket_id ON agent_responses(ticket_id);
CREATE INDEX idx_rep_feedback_response_id ON rep_feedback(response_id);
CREATE INDEX idx_knowledge_updates_content_hash ON knowledge_updates(content_hash);

-- ============================================================================
-- ROW LEVEL SECURITY
-- Purpose: Enable RLS on all tables (policies to be added separately)
-- ============================================================================
ALTER TABLE tickets ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE rep_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_updates ENABLE ROW LEVEL SECURITY;
ALTER TABLE pattern_reports ENABLE ROW LEVEL SECURITY;
