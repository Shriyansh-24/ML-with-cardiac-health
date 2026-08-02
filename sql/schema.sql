-- ═══════════════════════════════════════════════════════════════════
-- CardioGenome — Phase 2 schema (accounts + report history)
--
-- Run this in the Supabase SQL Editor:
--     Dashboard → SQL Editor → New query → paste → Run
--
-- Creates two tables:
--   1. reports        — one row per saved risk report, scoped to a user.
--                       Row Level Security: users can only select/insert/
--                       delete their OWN rows (auth.uid() = user_id).
--   2. research_data  — fully anonymised, opt-in research contributions.
--                       NO user_id column, by design. Only the service-role
--                       key (server side) may write; no anon/authenticated
--                       read or write policies exist.
-- ═══════════════════════════════════════════════════════════════════

-- ── 1) reports ────────────────────────────────────────────────────────
create table if not exists public.reports (
    id            uuid primary key default gen_random_uuid(),
    user_id       uuid not null references auth.users(id) on delete cascade,
    created_at    timestamptz not null default now(),
    summary       jsonb not null,      -- overview: probability, risk level, model source
    conditions    jsonb not null,      -- per-condition scores + reasons
    form_snapshot jsonb not null       -- the raw intake answers, for re-render
);

alter table public.reports enable row level security;

drop policy if exists "reports_select_own" on public.reports;
create policy "reports_select_own" on public.reports
    for select using (auth.uid() = user_id);

drop policy if exists "reports_insert_own" on public.reports;
create policy "reports_insert_own" on public.reports
    for insert with check (auth.uid() = user_id);

drop policy if exists "reports_delete_own" on public.reports;
create policy "reports_delete_own" on public.reports
    for delete using (auth.uid() = user_id);

create index if not exists reports_user_created_idx
    on public.reports (user_id, created_at desc);

-- ── 2) research_data (anonymised, opt-in) ─────────────────────────────
create table if not exists public.research_data (
    id         uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    payload    jsonb not null
);

alter table public.research_data enable row level security;

-- Deliberately NO select/insert policies: the service-role key bypasses
-- RLS (used server-side only), and nobody else — including logged-in
-- users — can read or write this table from the client.
