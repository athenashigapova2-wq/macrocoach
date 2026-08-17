-- Trace every provider attempt and make model-routing decisions auditable.

alter table public.agent_llm_calls
  add column invocation_id uuid not null default gen_random_uuid(),
  add column attempt_number integer not null default 1
    check (attempt_number > 0),
  add column requested_model_tier text,
  add column routing_rule text not null default 'legacy',
  add column selection_reason text not null default 'legacy trace',
  add column is_fallback boolean not null default false,
  add column fallback_reason text,
  add column retry_reason text;

update public.agent_llm_calls
set requested_model_tier = model_tier
where requested_model_tier is null;

alter table public.agent_llm_calls
  alter column requested_model_tier set not null,
  add constraint agent_llm_calls_requested_model_tier_check
    check (requested_model_tier in ('small', 'main')),
  add constraint agent_llm_calls_fallback_reason_check
    check (
      (is_fallback and fallback_reason is not null)
      or (not is_fallback and fallback_reason is null)
    );

create unique index agent_llm_calls_invocation_attempt_idx
  on public.agent_llm_calls(invocation_id, attempt_number);

create index agent_llm_calls_routing_created_idx
  on public.agent_llm_calls(routing_rule, created_at desc);

create index agent_llm_calls_fallback_created_idx
  on public.agent_llm_calls(created_at desc)
  where is_fallback;

comment on column public.agent_llm_calls.invocation_id is
  'Groups all retry attempts belonging to one logical LLM invocation.';
comment on column public.agent_llm_calls.attempt_number is
  'One-based provider attempt number inside invocation_id.';
comment on column public.agent_llm_calls.requested_model_tier is
  'Tier selected by routing policy before configuration fallback.';
comment on column public.agent_llm_calls.model_tier is
  'Actual tier used for this provider attempt.';
comment on column public.agent_llm_calls.routing_rule is
  'Matched node.purpose routing rule or default.';
comment on column public.agent_llm_calls.selection_reason is
  'Human-readable deterministic reason for the model choice.';
comment on column public.agent_llm_calls.fallback_reason is
  'Reason the requested tier could not be used; null without fallback.';
comment on column public.agent_llm_calls.retry_reason is
  'Failure classification from the preceding attempt; null on attempt 1.';
