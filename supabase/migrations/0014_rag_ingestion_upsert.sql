-- Atomic, idempotent document/chunk upsert for the authenticated Python backend.
-- Source governance remains in knowledge_sources and is checked again inside the
-- transaction, even though Python validates the ingestion bundle first.

create index if not exists knowledge_chunks_embedding_hnsw_idx
  on public.knowledge_chunks using hnsw (embedding extensions.vector_cosine_ops)
  where embedding is not null;

create or replace function public.upsert_knowledge_document(
  p_source_slug text,
  p_document jsonb,
  p_chunks jsonb,
  p_force boolean default false
)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  v_source public.knowledge_sources%rowtype;
  v_document_id uuid;
  v_previous_hash text;
  v_status text;
  v_chunk_count integer;
  v_chunks_match boolean := false;
begin
  if jsonb_typeof(p_document) <> 'object' then
    raise exception 'p_document must be a JSON object';
  end if;
  if jsonb_typeof(p_chunks) <> 'array' or jsonb_array_length(p_chunks) = 0 then
    raise exception 'p_chunks must be a non-empty JSON array';
  end if;
  if nullif(btrim(p_document->>'external_id'), '') is null then
    raise exception 'document external_id is required';
  end if;
  if coalesce(p_document->>'content_hash', '') !~ '^[0-9a-f]{64}$' then
    raise exception 'document content_hash must be a lowercase SHA-256 hex digest';
  end if;
  if exists (
    select 1
    from jsonb_array_elements(p_chunks) item
    where (item->>'chunk_index')::integer < 0
       or nullif(btrim(item->>'content'), '') is null
       or coalesce(item->>'content_hash', '') !~ '^[0-9a-f]{64}$'
       or (item->>'token_count')::integer <= 0
       or nullif(btrim(item->>'embedding_model'), '') is null
       or jsonb_typeof(item->'embedding') <> 'array'
  ) then
    raise exception 'one or more chunks have an invalid ingestion contract';
  end if;

  select * into v_source
  from public.knowledge_sources
  where slug = p_source_slug
  for update;

  if not found then
    raise exception 'knowledge source % is not registered', p_source_slug;
  end if;
  if v_source.rights_status <> 'approved' or not v_source.ingestion_enabled then
    raise exception 'knowledge source % is not approved and enabled', p_source_slug;
  end if;

  select id, content_hash
    into v_document_id, v_previous_hash
  from public.knowledge_documents
  where source_id = v_source.id
    and external_id = p_document->>'external_id'
  for update;

  if found then
    select
      count(*) = jsonb_array_length(p_chunks)
      and not exists (
        select 1
        from jsonb_array_elements(p_chunks) incoming
        left join public.knowledge_chunks existing
          on existing.document_id = v_document_id
         and existing.chunk_index = (incoming->>'chunk_index')::integer
        where existing.id is null
           or existing.content_hash <> incoming->>'content_hash'
           or existing.embedding_model <> incoming->>'embedding_model'
           or existing.section_title is distinct from nullif(incoming->>'section_title', '')
           or existing.token_count <> (incoming->>'token_count')::integer
           or existing.metadata <> coalesce(incoming->'metadata', '{}'::jsonb)
           or existing.embedding is null
      )
    into v_chunks_match
    from public.knowledge_chunks
    where document_id = v_document_id;

    update public.knowledge_documents
    set title = p_document->>'title',
        canonical_url = p_document->>'canonical_url',
        language = coalesce(nullif(p_document->>'language', ''), 'en'),
        source_updated_at = nullif(p_document->>'source_updated_at', '')::timestamptz,
        fetched_at = coalesce(
          nullif(p_document->>'fetched_at', '')::timestamptz,
          now()
        ),
        content_hash = p_document->>'content_hash',
        status = 'active',
        metadata = coalesce(p_document->'metadata', '{}'::jsonb),
        updated_at = now()
    where id = v_document_id;

    if not p_force
       and v_previous_hash = p_document->>'content_hash'
       and v_chunks_match then
      return jsonb_build_object(
        'status', 'unchanged',
        'document_id', v_document_id,
        'chunks_written', 0
      );
    end if;
    v_status := 'updated';
  else
    insert into public.knowledge_documents (
      source_id, external_id, title, canonical_url, language,
      source_updated_at, fetched_at, content_hash, status, metadata
    ) values (
      v_source.id,
      p_document->>'external_id',
      p_document->>'title',
      p_document->>'canonical_url',
      coalesce(nullif(p_document->>'language', ''), 'en'),
      nullif(p_document->>'source_updated_at', '')::timestamptz,
      coalesce(nullif(p_document->>'fetched_at', '')::timestamptz, now()),
      p_document->>'content_hash',
      'active',
      coalesce(p_document->'metadata', '{}'::jsonb)
    )
    returning id into v_document_id;
    v_status := 'inserted';
  end if;

  delete from public.knowledge_chunks where document_id = v_document_id;

  insert into public.knowledge_chunks (
    document_id, chunk_index, section_title, content, content_hash,
    token_count, embedding_model, embedding, metadata
  )
  select
    v_document_id,
    (item->>'chunk_index')::integer,
    nullif(item->>'section_title', ''),
    item->>'content',
    item->>'content_hash',
    (item->>'token_count')::integer,
    item->>'embedding_model',
    (item->>'embedding')::extensions.vector(768),
    coalesce(item->'metadata', '{}'::jsonb)
  from jsonb_array_elements(p_chunks) item;

  get diagnostics v_chunk_count = row_count;
  return jsonb_build_object(
    'status', v_status,
    'document_id', v_document_id,
    'chunks_written', v_chunk_count
  );
end;
$$;

revoke all on function public.upsert_knowledge_document(text, jsonb, jsonb, boolean)
  from public, anon, authenticated;
grant execute on function public.upsert_knowledge_document(text, jsonb, jsonb, boolean)
  to service_role;
