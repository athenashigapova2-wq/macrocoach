# Redis и Celery workers

Chat работает через фоновую очередь:

1. `POST /api/v1/agent/chat` проверяет Supabase JWT и возвращает `202` с `job_id`.
2. Celery worker забирает задание из Redis, запускает LangGraph и сохраняет сообщения в Supabase.
3. Клиент опрашивает `GET /api/v1/agent/chat/jobs/{job_id}` до статуса `succeeded` или `failed`.

Записи статусов живут в Redis один час (`AGENT_JOB_TTL_SECONDS`). Статус доступен
только пользователю, чей `user_id` был извлечён из проверенного JWT. Задания агента
не повторяются автоматически: повтор state-changing tool call опаснее, чем явная
повторная отправка пользователем.

## Запуск API, Redis и worker в Docker

Заполните `backend/.env`, затем из корня репозитория выполните:

```powershell
docker compose up -d --build
docker compose ps
```

Ожидаемый результат: сервисы `redis`, `api` и `worker` имеют статус `healthy`.
API доступен на `http://127.0.0.1:8001`, Redis опубликован только на
`127.0.0.1:6379`. Внутри сети Compose API и worker используют адрес
`redis://redis:6379/0`; значение из `backend/.env` переопределяется только внутри
контейнеров.

При первом запуске worker заранее загружает локальную embedding-модель. До окончания
загрузки worker остаётся в состоянии `health: starting` и не принимает задания.
Модель сохраняется в volume `hf-cache`, поэтому пересоздание контейнера не требует
повторной загрузки.

LLM-запросы и явно помеченные read-only инструменты повторяются только при временных
сетевых ошибках, timeout, HTTP 429 и выбранных HTTP 5xx. Используется ограниченный
экспоненциальный backoff с jitter. Операции записи (`log_meal`, `log_workout`) и
Celery-задача целиком не повторяются, чтобы не создавать дубликаты данных.

После исчерпания LLM retries общий Redis circuit breaker учитывает один логический
сбой. При достижении порога состояние GigaChat меняется `closed → open`; после cooldown
только один worker получает `half_open` probe lease. Успех закрывает circuit, временная
ошибка открывает повторно. Проверка состояния:

```powershell
docker compose exec -T redis redis-cli HGETALL athena:circuit-breaker:gigachat
```

Отсутствующий ключ означает `closed`. При недоступности Redis breaker работает
fail-open и не блокирует LLM сам по себе.

Model routing настраивается через `LLM_MODEL_ROUTING_POLICY`: точные правила
`node.purpose` имеют приоритет над `node.*`, затем `*.purpose` и `*`. Tier `small`
использует `LLM_ROUTER_MODEL` с безопасным возвратом к `GIGACHAT_MODEL`, tier `main`
всегда использует `GIGACHAT_MODEL`. Роутинг не переключает provider: все вызовы
остаются в GigaChat, а фактическая модель записывается в traces.

Одна строка `agent_llm_calls` теперь соответствует одной фактической provider-попытке.
Retries группируются по `invocation_id` и нумеруются через `attempt_number`; routing
rule, причина выбора, configuration fallback и причина retry сохраняются отдельными
полями. До запуска обновлённого worker примените Supabase migration
`0015_agent_llm_attempt_tracing.sql`.

Просмотр логов:

```powershell
docker compose logs -f api worker
```

Остановка сервисов без удаления данных Redis:

```powershell
docker compose down
```

Удалять постоянный volume следует только при намеренном сбросе очереди и данных Redis:

```powershell
docker compose down --volumes
```

## Ручной локальный запуск в Windows

Все команды ниже выполняются из корня репозитория в отдельных окнах PowerShell.

Сначала установите [Docker Desktop](https://www.docker.com/products/docker-desktop/),
затем зависимости проекта и запустите Redis:

```powershell
& ".\.venv\Scripts\Activate.ps1"
python -m pip install -r ".\backend\requirements.txt"
docker compose up -d redis
docker compose ps
```

Запустите API:

```powershell
& ".\.venv\Scripts\Activate.ps1"
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8001
```

Запустите worker в другом окне. Потоковый pool поддерживается Windows и позволяет
обрабатывать до четырёх I/O-bound LLM-заданий одновременно:

```powershell
& ".\.venv\Scripts\Activate.ps1"
Set-Location ".\backend"
python -m celery -A app.workers.celery_app:celery_app worker --loglevel=INFO --pool=threads --concurrency=4
```

Затем запустите Vite как обычно на `5175`. Готовность Redis видна в
`http://127.0.0.1:8001/health/ready`: поле `redis` должно быть `ready`.

Остановка локального Redis:

```powershell
docker compose stop redis
```

## Production

API, Redis и worker должны быть отдельными процессами/сервисами. Всем экземплярам
API и workers задайте одинаковые `REDIS_URL` и `AGENT_JOB_QUEUE`. Redis не следует
публиковать в интернет; используйте приватную сеть и пароль/TLS, которые предоставляет
ваш managed Redis. Масштабируйте workers количеством процессов/контейнеров, сохраняя
одинаковое имя очереди.
