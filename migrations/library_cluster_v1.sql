-- Library Cluster v1 — schema for Supabase / PostgreSQL backend.
-- Apply with:  psql $LIBRARY_DB_DSN -f library_cluster_v1.sql
-- Spec: knowledge_base/02_architecture/features/_blueprints/library_cluster/02_data_model.md

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- topics
CREATE TABLE IF NOT EXISTS topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    current_stage INTEGER NOT NULL DEFAULT 0,
    stage_history JSONB DEFAULT '[]'::JSONB,
    article_count INTEGER DEFAULT 0,
    skill_count INTEGER DEFAULT 0,
    contributor_count INTEGER DEFAULT 0,
    total_iterations INTEGER DEFAULT 0,
    estimated_total_iterations INTEGER DEFAULT 100,
    progress_ratio FLOAT DEFAULT 0.0,
    parent_topic_id UUID REFERENCES topics(id) ON DELETE SET NULL,
    subtopic_ids UUID[] DEFAULT '{}',
    related_topic_ids UUID[] DEFAULT '{}',
    embedding VECTOR(768),
    keywords TEXT[] DEFAULT '{}',
    layer TEXT NOT NULL DEFAULT 'open',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_topics_slug ON topics(slug);
CREATE INDEX IF NOT EXISTS idx_topics_layer ON topics(layer);
CREATE INDEX IF NOT EXISTS idx_topics_embedding ON topics USING hnsw(embedding vector_cosine_ops);

-- articles
CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_id UUID NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    abstract TEXT DEFAULT '',
    body_md TEXT NOT NULL,
    authored_by TEXT NOT NULL,
    user_id UUID,
    llm_model TEXT,
    quality_score FLOAT DEFAULT 0.0,
    hallucination_score FLOAT DEFAULT 0.0,
    contributes_to_stage INTEGER DEFAULT 0,
    layer TEXT NOT NULL DEFAULT 'closed',
    published_to_open_at TIMESTAMPTZ,
    license TEXT DEFAULT 'CC-BY-NC-4.0',
    embedding VECTOR(768),
    is_ai_marked BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_articles_topic ON articles(topic_id);
CREATE INDEX IF NOT EXISTS idx_articles_layer ON articles(layer);
CREATE INDEX IF NOT EXISTS idx_articles_embedding ON articles USING hnsw(embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_articles_fts ON articles USING gin(to_tsvector('english', body_md));

CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    title TEXT,
    accessed_at TIMESTAMPTZ,
    reliability_score FLOAT DEFAULT 0.5,
    layer TEXT DEFAULT 'secondary'
);

CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    hallucination_flag BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS claim_sources (
    claim_id UUID REFERENCES claims(id) ON DELETE CASCADE,
    source_id UUID REFERENCES sources(id) ON DELETE CASCADE,
    PRIMARY KEY (claim_id, source_id)
);

-- skills
CREATE TABLE IF NOT EXISTS skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    topic_id UUID REFERENCES topics(id) ON DELETE CASCADE,
    signature JSONB NOT NULL,
    prompt_template TEXT NOT NULL,
    few_shot_examples JSONB DEFAULT '[]',
    tools_used TEXT[] DEFAULT '{}',
    version TEXT NOT NULL DEFAULT '0.1.0',
    parent_version TEXT,
    success_rate FLOAT DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    layer TEXT DEFAULT 'closed',
    published_to_open_at TIMESTAMPTZ,
    install_mode TEXT DEFAULT 'pass_through',
    storage_location TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID NOT NULL
);

-- research sessions
CREATE TABLE IF NOT EXISTS research_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_id UUID REFERENCES topics(id) ON DELETE SET NULL,
    user_id UUID NOT NULL,
    query TEXT NOT NULL,
    mode TEXT NOT NULL,
    pathway_steps JSONB DEFAULT '[]',
    articles_generated UUID[] DEFAULT '{}',
    skills_updated UUID[] DEFAULT '{}',
    sources_consulted UUID[] DEFAULT '{}',
    contribution_pct_to_topic FLOAT DEFAULT 0.0,
    user_specific_pct FLOAT DEFAULT 1.0,
    tokens_consumed INTEGER DEFAULT 0,
    duration_seconds INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    layer TEXT DEFAULT 'closed'
);

-- hypotheses
CREATE TABLE IF NOT EXISTS hypotheses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_id UUID NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    derived_from_articles UUID[] DEFAULT '{}',
    generated_by TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    validation_evidence UUID[] DEFAULT '{}',
    initial_confidence FLOAT DEFAULT 0.5,
    current_confidence FLOAT DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    layer TEXT DEFAULT 'closed'
);

-- RLS policies — enforce OPEN-vs-CLOSED + own-row access.
ALTER TABLE articles ENABLE ROW LEVEL SECURITY;
ALTER TABLE skills ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE hypotheses ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS articles_open_read ON articles;
CREATE POLICY articles_open_read ON articles FOR SELECT USING (layer = 'open');

DROP POLICY IF EXISTS articles_own_read ON articles;
CREATE POLICY articles_own_read ON articles FOR SELECT
    USING (user_id = current_setting('app.current_user', true)::uuid);

DROP POLICY IF EXISTS skills_open_read ON skills;
CREATE POLICY skills_open_read ON skills FOR SELECT USING (layer = 'open');

DROP POLICY IF EXISTS skills_own_read ON skills;
CREATE POLICY skills_own_read ON skills FOR SELECT
    USING (created_by = current_setting('app.current_user', true)::uuid);

DROP POLICY IF EXISTS sessions_own_read ON research_sessions;
CREATE POLICY sessions_own_read ON research_sessions FOR SELECT
    USING (user_id = current_setting('app.current_user', true)::uuid);
