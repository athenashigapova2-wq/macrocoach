# Athena AI

Athena AI — мобильное приложение для питания, тренировок и восстановления с
персональным AI-агентом. Интерфейс построен на React и Capacitor, серверная часть —
на FastAPI, LangGraph, Redis и Celery. Единственная облачная LLM проекта — GigaChat.

Поддерживаемые языки: русский, английский, французский, испанский и китайский.

## Возможности

- онбординг и расчёт целевых калорий и БЖУ;
- дневник питания и семантический поиск по справочнику продуктов;
- история и запись тренировок;
- данные сна, энергии, веса и цикла;
- AI-чат с маршрутизацией по specialist-агентам;
- RAG по проверенной базе знаний с каноническими ссылками;
- трассировка LLM/tool calls и offline-eval сценарии на пяти языках.

## Архитектура

```text
React / Vite / Capacitor
        │ Supabase JWT
        ▼
FastAPI :8001
        │ создаёт job
        ▼
Redis :6379 ─────► Celery worker
  │                    │
  │ status/result      ├─ LangGraph router
  │                    ├─ RAG retriever
  └────────────────────┼─ Nutrition / Workout / Recovery / General
                       ├─ GigaChat
                       └─ Supabase PostgreSQL + pgvector
```

В Docker запускаются три сервиса:

| Сервис | Назначение |
|---|---|
| `api` | FastAPI, JWT boundary, постановка заданий и выдача статуса |
| `redis` | Celery broker и краткоживущее хранилище job status/result |
| `worker` | Выполнение LangGraph, GigaChat, RAG и инструментов |

Frontend в локальной разработке запускается отдельно через Vite на порту `5175`.

### Поток сообщения

1. Frontend отправляет `POST /api/v1/agent/chat` с Supabase access token.
2. FastAPI проверяет JWT, создаёт job в Redis и возвращает `202` с `job_id`.
3. Celery worker получает задание из очереди `athena-agent`.
4. Router выбирает `nutrition`, `workout`, `recovery` или `general`.
5. Retriever добавляет релевантный RAG-контекст.
6. Specialist вызывает GigaChat и доступные ему инструменты.
7. Результат сохраняется в Redis и Supabase; frontend опрашивает job endpoint.

### Границы инструментов

`user_id` берётся только из проверенного JWT и замыкается внутри tool-функций. Он
не входит в JSON Schema, которую видит модель. Каждый specialist получает только
свой набор инструментов:

- Nutrition: профиль, поиск еды, дневной рацион, запись еды;
- Workout: профиль, история и запись тренировок;
- Recovery: профиль, сон, энергия, вес и цикл;
- General: ответ без записывающих инструментов.

`log_meal` и `log_workout` выполняются только после явной просьбы пользователя.

## Надёжность

### Безопасные retries

Повторяются только идемпотентные операции:

- вызовы GigaChat;
- чтение истории разговора;
- RAG RPC;
- инструменты, явно отмеченные `read_only`.

По умолчанию выполняется до трёх попыток с exponential backoff `0.5s → 1s`,
jitter до 25% и верхней границей задержки 4 секунды. Повторяемыми считаются
network/timeout ошибки и HTTP `408`, `425`, `429`, `500`, `502`, `503`, `504`.

Операции записи и Celery-задача целиком не повторяются: это предотвращает двойную
запись еды или тренировки при неопределённом результате сетевого запроса.

### Конфигурируемый model router

Model router выбирает только между настроенными моделями GigaChat; переключения на
другого провайдера нет. Доступны два tier:

- `small` — `LLM_ROUTER_MODEL`, а если он пустой, используется `GIGACHAT_MODEL`;
- `main` — `GIGACHAT_MODEL`.

Policy задаётся JSON-объектом в `LLM_MODEL_ROUTING_POLICY`. Правила проверяются в
порядке `node.purpose → node.* → *.purpose → *`. Конфигурация по умолчанию отправляет
классификацию маршрута и перевод названия продукта в `small`, а остальные вызовы —
в `main`:

```env
LLM_MODEL_ROUTING_ENABLED=true
LLM_ROUTER_MODEL=
LLM_MODEL_ROUTING_POLICY={"router.route_classification":"small","nutrition.food_translation":"small","*":"main"}
```

Примеры `node`: `router`, `nutrition`, `workout`, `recovery`, `general`. Примеры
`purpose`: `route_classification`, `food_translation`, `tool_planning_or_answer`,
`answer`. Неизвестный или некорректный tier останавливает backend при чтении настроек;
молчаливого provider fallback нет. Фактические `model_tier` и `model_name` каждого
вызова сохраняются в `agent_llm_calls`.

Каждая фактическая попытка обращения к GigaChat создаёт отдельную строку
`agent_llm_calls`. Все retries одного логического вызова объединяет `invocation_id`,
а `attempt_number` содержит номер попытки. Для аудита также сохраняются:

- `requested_model_tier` и фактически использованный `model_tier`;
- `routing_rule` и человекочитаемый `selection_reason`;
- `is_fallback` и `fallback_reason`;
- `retry_reason` — классификация ошибки предыдущей попытки.

Если policy выбрала `small`, но `LLM_ROUTER_MODEL` пустой, используется main-модель и
это явно записывается как fallback. Runtime/provider fallback не выполняется. Перед
запуском этого backend-кода должна быть применена миграция
`0015_agent_llm_attempt_tracing.sql`.

После миграции `agent_runs.llm_call_count` означает число фактических provider-попыток,
а не число логических вызовов. Последние решения можно проверить в SQL Editor:

```sql
select invocation_id, attempt_number, node_name, purpose,
       requested_model_tier, model_tier, model_name,
       routing_rule, selection_reason, is_fallback,
       fallback_reason, retry_reason, status
from public.agent_llm_calls
order by created_at desc
limit 50;
```

### Redis circuit breaker

Все workers используют общий circuit breaker GigaChat в Redis. После пяти логических
временных сбоев (каждый считается только после исчерпания retries) circuit переходит
из `closed` в `open`. Следующие LLM-вызовы завершаются сразу, не создавая дополнительную
нагрузку на недоступного провайдера.

Через 30 секунд circuit атомарно переходит в `half_open`: только один worker получает
lease на пробный вызов. Успешный probe закрывает circuit и сбрасывает счётчик; временная
ошибка снова открывает его. Если worker погиб во время probe, 210-секундный lease
освобождается автоматически. Lua-скрипты и Redis `TIME` обеспечивают одинаковое
состояние и единственный probe при нескольких контейнерах worker.

При недоступности Redis breaker работает fail-open: GigaChat-вызов разрешается, чтобы
защитный механизм сам не стал причиной отказа. Ошибки запроса, авторизации и другие
не-временные ошибки не увеличивают счётчик availability failures.

Текущее состояние можно посмотреть без изменения данных:

```powershell
docker compose exec -T redis redis-cli `
    HGETALL athena:circuit-breaker:gigachat
```

Отсутствующий ключ означает `closed`. Параметры порога, cooldown, probe lease и TTL
задаются переменными `LLM_CIRCUIT_BREAKER_*` из `backend/.env.example`.

### Embedding model

Worker до перехода в `ready` загружает локальную модель
`intfloat/multilingual-e5-base`. Файлы сохраняются в Docker volume `hf-cache`,
поэтому пересоздание контейнера не требует повторного скачивания. Инициализация
защищена lock и выполняется один раз при четырёх worker threads.

### Health checks

- `/health` — процесс FastAPI работает;
- `/health/ready` — заданы обязательные настройки и доступен Redis;
- worker healthcheck — Celery отвечает на `inspect ping`;
- Redis healthcheck — `redis-cli ping`.

## Стек

**Frontend:** React, Vite, Tailwind CSS, Capacitor, Supabase JS.

**Backend:** Python 3.11, FastAPI, LangChain Core, LangGraph, GigaChat, Celery.

**Data:** Supabase PostgreSQL, pgvector, Row Level Security, Redis.

**ML:** GigaChat и локальная `multilingual-e5-base` для embeddings.

## Требования

- Node.js 20+ и npm;
- Python 3.11;
- Docker Desktop с Linux Engine;
- проект Supabase;
- GigaChat Authorization key.

## Настройка окружения

### Frontend

```powershell
Copy-Item ".\.env.example" ".\.env"
npm install
```

Заполните корневой `.env`:

```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
VITE_AGENT_API_URL=https://your-production-api.example.com
AGENT_PROXY_TARGET=http://127.0.0.1:8001
```

`VITE_`-переменные попадают в клиентский bundle. Никогда не помещайте туда
`service_role` или GigaChat Authorization key.

### Backend

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install -r ".\backend\requirements.txt"
Copy-Item ".\backend\.env.example" ".\backend\.env"
```

Минимальная конфигурация `backend/.env`:

```env
GIGACHAT_AUTH_KEY=
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_MODEL=GigaChat-2
LLM_ROUTER_MODEL=
LLM_MODEL_ROUTING_ENABLED=true
LLM_MODEL_ROUTING_POLICY={"router.route_classification":"small","nutrition.food_translation":"small","*":"main"}

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_JWT_SECRET=
SUPABASE_JWT_AUDIENCE=authenticated

API_CORS_ORIGINS=http://localhost:5175,http://127.0.0.1:5175
REDIS_URL=redis://127.0.0.1:6379/0
```

`SUPABASE_JWT_SECRET` нужен только проектам с legacy HS256-токенами. Для
RS256/ES256 backend получает публичные ключи из Supabase JWKS.

Дополнительные настройки retries и RAG перечислены в
[`backend/.env.example`](backend/.env.example).

## Supabase migrations

Примените миграции из `supabase/migrations/` строго по имени:

```powershell
npx supabase login
npx supabase link --project-ref <project-ref>
npx supabase db push
```

Если Supabase CLI не используется, выполните все миграции по порядку через
Dashboard → SQL Editor.

## Запуск через Docker Compose

Рекомендуемый способ запуска backend:

```powershell
docker compose up -d --build
docker compose ps
```

Ожидаемый результат:

```text
athenaai-api-1      healthy
athenaai-redis-1    healthy
athenaai-worker-1   healthy
```

Проверка API:

```powershell
Invoke-RestMethod "http://127.0.0.1:8001/health"
Invoke-RestMethod "http://127.0.0.1:8001/health/ready"
```

Проверка worker:

```powershell
docker compose exec -T worker python -m celery `
    -A app.workers.celery_app:celery_app `
    inspect ping --timeout=5
```

Логи:

```powershell
docker compose logs -f api worker
```

Остановка без удаления данных:

```powershell
docker compose down
```

`docker compose down --volumes` удаляет очередь Redis и кэш embedding-модели;
используйте команду только для намеренного полного сброса.

Подробности ручного запуска Redis и Celery находятся в
[`backend/WORKERS.md`](backend/WORKERS.md).

## Запуск frontend

В отдельном PowerShell:

```powershell
npm run dev -- --host 127.0.0.1 --port 5175
```

Откройте:

- приложение: <http://127.0.0.1:5175/>;
- AI-чат: <http://127.0.0.1:5175/chat>;
- readiness через Vite proxy: <http://127.0.0.1:5175/agent-api/health/ready>.

В dev-режиме Vite проксирует `/agent-api` на `127.0.0.1:8001`, поэтому локальный
чат не зависит от CORS.

## Проверки

Offline-проверки не обращаются к GigaChat и не изменяют Supabase:

```powershell
python backend/scripts/test_fastapi.py
python backend/scripts/test_agent_architecture.py
python backend/scripts/test_agent_workers.py
python backend/scripts/test_agent_traces.py
python backend/scripts/test_retry_policy.py
python backend/scripts/test_circuit_breaker.py
python backend/scripts/test_model_routing.py
python backend/scripts/test_rag_retriever.py
```

Evaluation datasets:

```powershell
python backend/scripts/eval_agents.py
python backend/scripts/eval_tool_selection.py
python backend/scripts/eval_write_safety.py
python backend/scripts/eval_answer_quality.py
```

Команды с `--live` выполняют реальные вызовы GigaChat, но используют фиктивные
результаты инструментов и не должны записывать пользовательские данные.

## RAG и справочник продуктов

Runtime-путь RAG: `router → retriever → specialist`. Документация ingestion,
лицензирования источников и формата bundles находится в
[`backend/knowledge/README.md`](backend/knowledge/README.md).

Основные команды:

```powershell
python backend/scripts/ingest_knowledge.py --help
python backend/scripts/import_food_data.py --csv-dir backend/data/kaggle --dry-run
python backend/scripts/build_embeddings.py
```

Справочник содержит 2210 нормализованных продуктов. Комбинация перевода запроса,
multilingual embeddings и доменного ранжирования показала Recall@5 93% на
внутреннем наборе из 30 запросов.

## Безопасность

- GigaChat key и Supabase `service_role` существуют только на backend;
- frontend передаёт Supabase access token в `Authorization: Bearer ...`;
- FastAPI получает `user_id` только из проверенного JWT;
- пользовательские запросы Supabase фильтруются по доверенному `user_id`;
- write-tools отделены от read-only tools и не получают автоматические retries;
- access tokens и значения секретов не выводятся в readiness или логи.

## Диагностика

### Docker API недоступен

Если отсутствует `dockerDesktopLinuxEngine`, запустите Docker Desktop и дождитесь
`Engine running`, затем выполните `docker version`.

### Порт 8001 занят

```powershell
Get-NetTCPConnection -LocalPort 8001 -State Listen
```

Остановите вручную запущенный Uvicorn перед публикацией порта Docker API.

### Chat возвращает 401

Проверьте, что `VITE_SUPABASE_URL` и `SUPABASE_URL` относятся к одному проекту,
затем очистите site data и войдите заново.

### GigaChat не отвечает

Проверьте Authorization key и доступные модели диагностическим скриптом:

```powershell
python backend/scripts/check_gigachat_api.py --insecure
```

`--insecure` допустим только для локальной диагностики TLS. Backend не должен
отключать проверку сертификатов в production.

## Структура проекта

```text
backend/app/
  agents/       LangGraph router, retriever и specialists
  api/          FastAPI endpoints
  auth/         Supabase JWT validation
  model_routing.py  GigaChat model-routing policy
  rag/          retrieval и ingestion contracts
  services/     jobs, conversations, tracing, Supabase
  tools/        read/write инструменты агента
  workers/      Celery application и tasks
supabase/
  migrations/   PostgreSQL, RLS, pgvector и agent traces
src/            React frontend
compose.yaml    Redis, FastAPI и Celery worker
```

## Лицензия

Добавьте условия лицензии проекта перед публичным распространением.
