# Deploying RAGScope for free on Oracle Cloud

RAGScope needs roughly 6-10 GB of RAM: the 3B answer model, the reranker
(PyTorch), the embedding model and Postgres with pgvector. Free web tiers such as
Render (512 MB), Railway (no real free tier) and Vercel (frontend only) cannot run
it. Oracle Cloud's **Always Free ARM server** can, using the production files
already in this repo: `docker-compose.prod.yml` (no GPU, only Caddy published)
and `Caddyfile` (HTTPS, API key added on the server side).

Status (2026-11-09): not deployed yet. Part 1 is waiting on the Oracle account.

| What | Value |
|---|---|
| Cost | $0 on Always Free (a card is needed for identity checks only) |
| Server | Ampere A1, Ubuntu 24.04 ARM. The free allowance was reportedly cut from 4 CPUs / 24 GB to 2 CPUs / 12 GB in mid-2026; use whatever the console marks "Always Free" |
| Expected speed | Estimate: 15-30 s per answer on 2 ARM cores with no GPU (about 2 s on the laptop's GPU). Measure after deploying |
| Main risk | Oracle may reclaim an Always Free server that stays idle for 7 days (CPU, network and memory all under 20%). This stack keeps memory well above 20%, so it is unlikely, not impossible |

---

## Part 1: Oracle account and server (you, about 30 minutes)

1. **Create the account** at oracle.com/cloud/free.
   - It asks for a card to verify you; Always Free resources are not charged.
   - For **Home Region**, pick **India South (Hyderabad)** or **India West (Mumbai)**. It cannot be changed later.
2. **Create the server:** Menu → Compute → Instances → **Create instance**.
   - **Image:** Canonical **Ubuntu 24.04**, the aarch64 (ARM) build.
   - **Shape:** Ampere → **VM.Standard.A1.Flex**, with the CPUs and memory set to the highest values marked Always Free.
   - **SSH keys:** "Generate a key pair for me" → download the private key and save it as
     `C:\Users\sahil\.ssh\oracle-ragscope.key`.
   - **Boot volume:** 100 GB (up to 200 GB in total is free).
   - **"Out of capacity"** is common: try another availability domain, or retry later.
3. **Open the web ports:** instance page → subnet → **Security Lists** → Default → **Add Ingress Rules**.
   - One rule with source `0.0.0.0/0`, protocol TCP, destination port `80`.
   - A second rule, the same but with port `443`.
4. **Free domain:** at duckdns.org, sign in, create a subdomain (for example `ragscope-sahil`) and set its IP to the
   server's public IP. Keep the DuckDNS token private.
5. **Check SSH works** (PowerShell):
   ```powershell
   # Windows OpenSSH refuses a key file other users can read
   icacls C:\Users\sahil\.ssh\oracle-ragscope.key /inheritance:r /grant:r "$($env:USERNAME):(R)"
   ssh -i C:\Users\sahil\.ssh\oracle-ragscope.key ubuntu@<SERVER_IP>
   ```

Then hand over the server IP, the key path and the DuckDNS name (never the token) for Part 3.

## Part 2: two decisions before going public

1. **The deploy files are not committed.** `docker-compose.prod.yml`, `Caddyfile`, both Dockerfiles,
   `frontend/.dockerignore`, `.env.example` and `Makefile` exist only in the local working tree, so `git clone` on the
   server will not have them. Choose one:
   - commit and push them, then clone; or
   - copy them over with `scp` (Part 3, step 3).
2. **The public API is wide open.** Caddy adds the API key to every `/api/*` request. Anyone with the link can
   therefore do anything the API allows: delete documents, upload files, start evaluation runs, generate datasets or
   flush the cache. Before sharing the link, restrict `/api` so that visitors can only read, ask questions and send
   feedback. Proposed `Caddyfile` (**not tested yet**: validate it on the server first, see Part 3):

   ```caddyfile
   {$DOMAIN} {
   	handle /api/* {
   		# Visitors may read, ask questions and send feedback. Uploads, deletes,
   		# evals, dataset generation and cache flushes are refused here.
   		@blocked {
   			not method GET
   			not path /api/v1/query /api/v1/query/stream /api/v1/retrieve /api/v1/feedback
   		}
   		respond @blocked "This public demo is read-only." 403

   		reverse_proxy backend:8000 {
   			# Set, not appended: a client-supplied X-API-Key is overwritten.
   			header_up X-API-Key {$API_KEY}
   		}
   	}
   	handle {
   		reverse_proxy frontend:3000
   	}
   }
   ```
   Inside a `handle`, Caddy runs `respond` before `reverse_proxy`, so blocked requests never reach the backend. Both
   `not` lines must hold for a request to be blocked: it is not a GET, and it is not one of the four allowed paths.

## Part 3: server setup (over SSH)

Commands below run on the server unless marked **laptop**.

1. **Firewall.** Oracle's Ubuntu images block everything except SSH, even after the Security List rules:
   ```bash
   sudo iptables -I INPUT -p tcp -m multiport --dports 80,443 -m conntrack --ctstate NEW -j ACCEPT
   sudo netfilter-persistent save
   ```
2. **Docker** (Compose 2.24 or newer is needed for the `!reset` tags in the prod file):
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker ubuntu && exit   # log in again so the group applies
   docker compose version
   ```
3. **Code:**
   ```bash
   git clone https://github.com/sahilsahu4102/ragscope.git && cd ragscope
   ```
   If the deploy files were not pushed (decision 1), copy them from the **laptop**, in `F:\PROJECTS\Ragg`:
   ```powershell
   scp -i C:\Users\sahil\.ssh\oracle-ragscope.key docker-compose.prod.yml Caddyfile .env.example Makefile ubuntu@<SERVER_IP>:~/ragscope/
   scp -i C:\Users\sahil\.ssh\oracle-ragscope.key backend\Dockerfile ubuntu@<SERVER_IP>:~/ragscope/backend/
   scp -i C:\Users\sahil\.ssh\oracle-ragscope.key frontend\Dockerfile frontend\.dockerignore ubuntu@<SERVER_IP>:~/ragscope/frontend/
   ```
4. **Secrets.** These are generated on the server and written straight into `.env`, so they are never displayed. The
   backend builds its database address from the `POSTGRES_*` values, so `DATABASE_URL` in the file is not used.
   ```bash
   cp .env.example .env
   sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$(openssl rand -hex 24)|" .env
   sed -i "s|^API_KEY=.*|API_KEY=$(openssl rand -hex 32)|" .env
   sed -i "s|^DOMAIN=.*|DOMAIN=ragscope-sahil.duckdns.org|" .env   # your DuckDNS name
   alias dc='docker compose -f docker-compose.yml -f docker-compose.prod.yml'
   dc config --quiet && echo "config OK"
   ```
5. **Copy the existing database instead of rebuilding it.** This avoids re-embedding 5,869 chunks on a slow ARM CPU.
   The source PDFs are git-ignored and would not arrive with the code anyway.
   - **Laptop**, in `F:\PROJECTS\Ragg`. The dump is written inside the container and copied out, because a PowerShell
     `>` redirect would corrupt the binary file:
     ```powershell
     docker compose up -d postgres
     docker compose exec -T postgres pg_dump -U ragscope -d ragscope -Fc -f /tmp/ragscope.dump
     docker compose cp postgres:/tmp/ragscope.dump .\ragscope.dump
     scp -i C:\Users\sahil\.ssh\oracle-ragscope.key ragscope.dump ubuntu@<SERVER_IP>:~/
     ```
   - **Server:**
     ```bash
     dc up -d postgres
     dc cp ~/ragscope.dump postgres:/tmp/ragscope.dump
     dc exec -T postgres pg_restore -U ragscope -d ragscope --no-owner /tmp/ragscope.dump
     ```
6. **Models** (about 2.3 GB; the reranker downloads itself on the first question):
   ```bash
   dc up -d ollama
   dc exec ollama ollama pull llama3.2:3b
   dc exec ollama ollama pull nomic-embed-text
   ```
7. **Validate the Caddyfile** (after applying decision 2):
   ```bash
   docker run --rm -e DOMAIN=x.example.com -e API_KEY=x -v "$PWD/Caddyfile:/etc/caddy/Caddyfile:ro" \
     caddy:2-alpine caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
   ```
8. **Start everything.** The first build of the backend and frontend on ARM takes roughly 10-20 minutes:
   ```bash
   dc up -d --build
   dc ps
   curl -sI https://ragscope-sahil.duckdns.org | head -1   # expect HTTP/2 200
   ```
9. **Check it end to end.** Open the site and ask a question. Note the answer time, and check that a write request to a
   blocked path returns 403:
   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" -X DELETE https://ragscope-sahil.duckdns.org/api/v1/query/cache   # expect 403
   ```

## Updating later

```bash
cd ~/ragscope && git pull && dc up -d --build
```

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| Browser can't connect | DuckDNS IP is wrong, or ports 80/443 are closed in the Security List or in `iptables` (Part 3, step 1) |
| HTTPS certificate error | DNS hasn't updated yet, or port 80 is blocked (Let's Encrypt checks over port 80) |
| Containers restarting | Out of memory: run `docker stats`. Lower the `celery-worker` memory limit, or switch the answer model to `llama3.2:1b` |
| Very slow answers | Expected without a GPU. `llama3.2:1b` is faster but gives worse answers |
| `!reset` errors | Docker Compose is older than 2.24 |
