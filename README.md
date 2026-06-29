# ORYND Backend — Site API

**Это бэкенд для сайта oryndai.com и Extension.**
Не путать с `ory_prod_back_1` (старый, архив) и `orynd_core` в Workspace (десктоп).

## Что делает

- Auth (Supabase JWT, email/password, Google OAuth)
- Billing (`/api/billing/me` — план, кредиты, баланс)
- Credits (quote → commit flow)
- MCP connector endpoint
- CAD / Mesh / Search / Skills — для Extension

## Стек

- FastAPI + Python 3.11
- Supabase (auth + база)
- Деплой: AWS EC2 `api.oryndai.com` → порт 8003 → nginx

## Запуск локально

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env  # заполнить ключи
.venv/bin/uvicorn orynd_core.api.main:app --reload --port 8003
```

## Деплой на EC2

```bash
# На сервере (api.oryndai.com, ubuntu@3.86.214.175)
cd /home/ubuntu/backend_v2
git pull
sudo systemctl restart orynd-back
```

## Ключевые env переменные

```
SUPABASE_URL=https://dblquhnokgpavubobfoj.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...   # только на сервере, никогда во фронте
ANTHROPIC_API_KEY=...
CRYPTOBOT_TOKEN=...
CRYPTOBOT_WEBHOOK_SECRET=...
```

## Связанные репо

| Репо | Что |
|------|-----|
| `orynd_cl_site_profe` | Фронт сайта (статика, branch `static-v3`) |
| `orynd_cl_site_profe_back` | **Этот бэк** |
| `ory_prod_back_1` | Старый бэк (архив, не трогать) |
| `orynd_app_dw_data` | GitHub Releases — DMG/EXE для скачивания |
