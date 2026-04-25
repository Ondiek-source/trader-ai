# Trader AI — Instruction Manual

This system automatically analyses forex price data, decides when to place a trade, and sends a signal to your Quotex trading bot. It runs 24/7 in the cloud. You do not need to leave your computer on.

---

## What You Need Before You Start

- An Azure account (portal.azure.com)
- A GitHub account (github.com)
- A Quotex account
- A Telegram account
- A Discord server with a channel you own
- A Twelve Data account (twelvedata.com)
- Azure CLI installed on your computer
- GitHub CLI (`gh`) installed on your computer
- Git installed on your computer

---

## Section 1 — First-Time Setup (Do This Once)

### Step 1 — Get your API keys

1. Go to **twelvedata.com** and sign up for a free account.
2. After logging in, find your **API Key** on the dashboard.
3. Copy it — you will need it in Step 3.
4. Go to **@BotFather** on Telegram and send `/newbot`.
5. Follow the prompts — give it a name and a username ending in `bot`.
6. Copy the **token** BotFather gives you.

### Step 2 — Get your Telegram Chat ID

1. Open Telegram and send any message to your new bot (e.g. "hello").
2. Open your browser and go to this URL (replace `TOKEN` with your bot token):
   `https://api.telegram.org/botTOKEN/getUpdates`
3. Find the `"chat"` section in the result.
4. Copy the `"id"` number inside it — this is your Chat ID.
5. Write it down somewhere safe.
6. You will use it in Step 3.

### Step 3 — Get your Discord webhook URL

1. Open Discord and go to the server where you want reports sent.
2. Right-click the channel you want → **Edit Channel**.
3. Click **Integrations** → **Webhooks** → **New Webhook**.
4. Give it a name (e.g. "Trader AI") then click **Copy Webhook URL**.
5. Save the URL — you will need it in Step 4.
6. Click **Save**.

### Step 4 — Create your .env file

1. Open the project folder on your computer.
2. Find the file called `.env.example` and make a copy of it.
3. Rename the copy to `.env` (no `.example` at the end).
4. Open `.env` in any text editor (Notepad is fine).
5. Fill in each value using the keys you collected in Steps 1–3.
6. Save the file. **Never share this file or commit it to GitHub** — it contains your passwords.

The values to fill in:

```txt
TWELVEDATA_API_KEY=        ← from twelvedata.com dashboard
WEBHOOK_URL=               ← the URL your Quotex bot listens on
QUOTEX_EMAIL=              ← your Quotex login email
QUOTEX_PASSWORD=           ← your Quotex login password
TELEGRAM_TOKEN=            ← from @BotFather
TELEGRAM_CHAT_ID=          ← from getUpdates URL
DISCORD_WEBHOOK_URL=       ← from Discord channel settings
WEBHOOK_KEY=Ondiek         ← leave as Ondiek
PRACTICE_MODE=true         ← set to false when you are ready to go live
```

### Step 5 — Log in to Azure and GitHub

1. Open a terminal (Git Bash on Windows).
2. Run: `az login` — a browser window will open, sign in with your Azure account.
3. Close the browser when it says you are signed in.
4. Run: `gh auth login` — follow the prompts to sign in to GitHub.
5. Choose **GitHub.com** → **HTTPS** → **Login with a web browser**.
6. Copy the code shown, open the browser link, paste the code.

### Step 6 — Provision Azure resources

1. In your terminal, navigate to the project folder:
   `cd "path/to/Trader AI"`
2. Run: `bash deploy/provision.sh`
3. Wait 3–5 minutes — it creates your storage, registry, and Windows VM.
4. When it finishes, it will print two lines starting with `AZURE_STORAGE_CONN=` and `CONTAINER_NAME=`.
5. Copy those two lines into your `.env` file.
6. Also write down the **VM IP, username, and password** printed at the end — you need these to access the bot machine.

---

## Section 2 — Deploy the AI Engine (Do This After Every Update)

### Step 7 — Add GitHub secrets

1. Go to **github.com/Ondiek-source/trader-ai**.
2. Click **Settings** → **Secrets and variables** → **Actions**.
3. Click **New repository secret** and add these one at a time:

| Secret Name | Where to find the value |
| --- | --- |
| `ACR_LOGIN_SERVER` | Printed by provision.sh (e.g. `traderaireg1234.azurecr.io`) |
| `ACR_USERNAME` | Same as the registry name (e.g. `traderaireg1234`) |
| `ACR_PASSWORD` | Run: `az acr credential show --name traderaireg1234 --query "passwords[0].value" -o tsv` |

1. After adding all three, they will appear in the list (values are hidden — that is normal).
2. Go back to your terminal.
3. Run: `git push origin main` — this triggers GitHub to build the Docker image automatically.

### Step 8 — Check the build succeeded

1. Wait 5–10 minutes for the build to finish.
2. Run: `gh run list --repo Ondiek-source/trader-ai --limit 1`
3. Look at the **STATUS** column — it should say `completed` and show a green tick.
4. If it shows a red X, run: `gh run view ID --repo Ondiek-source/trader-ai --log-failed` (replace ID with the number shown).
5. Fix the error and push again — the build retriggers automatically.
6. Once it succeeds, move to Step 9.

### Step 9 — Deploy to Azure

1. In your terminal, make sure you are in the project folder.
2. Run: `bash deploy/deploy.sh`
3. Wait 2–3 minutes.
4. When it finishes, it will print a table showing the container is running.
5. To see live logs, run:
   `az container logs --name trader-ai-engine --resource-group rg-trader-ai --follow`
6. You should see the system downloading historical data and training models — this is normal on first run and takes 20–30 minutes.

---

## Section 3 — Set Up the Trading Bot (Windows VM)

### Step 10 — Connect to the Windows VM

1. Open the **Remote Desktop Connection** app on your computer (search "RDP" in Start menu).
2. Enter the VM IP address printed by provision.sh (e.g. `4.223.132.84`).
3. Click **Connect**.
4. When asked for credentials, enter:
   - Username: `traderadmin`
   - Password: `TraderAI-2024xAz`
5. Click **OK** — the Windows desktop will appear.
6. You are now inside the cloud VM.

### Step 11 — Install Google Chrome on the VM

1. Inside the VM, open **Internet Explorer** or **Edge**.
2. Go to **google.com/chrome** and download Chrome.
3. Run the installer and complete the installation.
4. Open Chrome and make sure it works.
5. Keep Chrome up to date — the bot requires it.
6. Do not change Chrome's default settings.

### Step 12 — Install and configure the trading bot

1. Download **autobottradingsignal** — use the link or installer provided by the bot provider.
2. Run the installer and complete the setup.
3. Open the bot and log in with your Quotex account:
   - Email: your Quotex email
   - Password: your Quotex password
4. In the bot settings, set:
   - **Key**: `Ondiek`
   - **Expiry**: `60` seconds
   - Configure martingale settings as the bot requires
5. Make sure the bot is connected and showing your Quotex balance.
6. Leave the bot running — **do not close it, do not lock the screen**.

### Step 13 — Prevent the VM screen from locking

1. Inside the VM, right-click the desktop → **Personalize**.
2. Click **Lock screen** → **Screen timeout settings**.
3. Set **Screen** to **Never**.
4. Open **Control Panel** → **Power Options**.
5. Set **Turn off display** and **Sleep** both to **Never**.
6. The VM will now stay on indefinitely without locking.

---

## Section 4 — Daily Operations

### Step 14 — Monitor the system via Telegram

1. Open Telegram and find your bot.
2. Send these commands at any time:
   - `/status` — shows wins, losses, profit, current confidence level
   - `/report` — sends a full session summary right now
   - `/threshold` — shows the current confidence threshold
   - `/stop` — pauses signal generation immediately
   - `/start` — resumes signal generation
3. You will automatically receive a session report every weekday at **11:58 PM UTC**.
4. Check Discord for the same report in your configured channel.
5. If you do not receive a report, check logs:
   `az container logs --name trader-ai-engine --resource-group rg-trader-ai --follow`
6. If something looks wrong, send `/stop` in Telegram, fix the issue, then `/start`.

### Step 15 — Switch from practice to live trading

1. Open your `.env` file.
2. Change `PRACTICE_MODE=true` to `PRACTICE_MODE=false`.
3. Save the file.
4. Push the change: `git add .env && git commit -m "go live" && git push`
5. Wait for GitHub Actions to build (5–10 min), then run: `bash deploy/deploy.sh`
6. The system will now send real signals to your Quotex bot. Monitor closely for the first session.

### Step 16 — Update the system after code changes

1. Make your changes to the code.
2. In your terminal, run:
   `git add . && git commit -m "describe your change" && git push`
3. GitHub Actions will automatically build a new Docker image (5–10 min).
4. Check it succeeded: `gh run list --repo Ondiek-source/trader-ai --limit 1`
5. Once successful, run: `bash deploy/deploy.sh`
6. The running container will be replaced with the updated version.

---

## Section 5 — New Azure Tenant / Fresh Setup

### Step 17 — Deploy to a new Azure account

1. Log in to the new Azure account: `az login`
2. Run: `bash deploy/provision.sh` — creates all resources in the new account.
3. Copy the new `AZURE_STORAGE_CONN` and `CONTAINER_NAME` into your `.env`.
4. Update the three GitHub secrets (`ACR_LOGIN_SERVER`, `ACR_USERNAME`, `ACR_PASSWORD`) with the new values printed by provision.sh.
5. Run: `git push origin main` — triggers a fresh build into the new registry.
6. Once build succeeds, run: `bash deploy/deploy.sh` — deploys to the new account.

---

## Section 6 — Troubleshooting

### Step 18 — Container is not running

1. Check the container status:
   `az container show --name trader-ai-engine --resource-group rg-trader-ai --query instanceView.state`
2. If it says `Stopped` or `Failed`, check the logs:
   `az container logs --name trader-ai-engine --resource-group rg-trader-ai`
3. Look for any `ERROR` lines near the bottom of the logs.
4. Fix the issue (usually a missing or wrong `.env` value).
5. Re-run: `bash deploy/deploy.sh`
6. If the issue persists, send the error message to your developer.

### Step 19 — No signals are being sent

1. Send `/status` to your Telegram bot — check if the session is active.
2. Check that the trading bot on the VM is running and connected.
3. Check the AI engine logs for `signal_sent` or `below_threshold` events:
   `az container logs --name trader-ai-engine --resource-group rg-trader-ai | grep signal`
4. If you see `below_threshold` — the model is not confident enough yet. This is normal early in a session or after losses. Wait for it to find a good setup.
5. If you see no output at all, the stream may have disconnected — run `bash deploy/deploy.sh` to restart.
6. If signals are being sent but no trades are placed, check the bot on the VM is running.

### Step 20 — Tear down everything (nuclear option)

1. Only do this if you want to completely remove all Azure resources.
2. Run: `bash deploy/teardown.sh`
3. Type `yes` when prompted.
4. All resources (VM, storage, registry, container) will be deleted within a few minutes.
5. Your `.env` and code remain on your computer — nothing local is deleted.
6. To start again from scratch, go back to Section 1.

---

## Quick Reference

| Task | Command |
| --- | --- |
| Provision Azure (first time) | `bash deploy/provision.sh` |
| Deploy / redeploy | `bash deploy/deploy.sh` |
| Push update + redeploy | `git push` then `bash deploy/deploy.sh` |
| View live logs | `az container logs --name trader-ai-engine --resource-group rg-trader-ai --follow` |
| Check container status | `az container show --name trader-ai-engine --resource-group rg-trader-ai --query instanceView.state` |
| Tear down everything | `bash deploy/teardown.sh` |

| Telegram Command | What it does |
| --- | --- |
| `/status` | Current session wins, losses, profit, confidence |
| `/report` | Send full report now |
| `/threshold` | Show current confidence threshold |
| `/stop` | Pause trading |
| `/start` | Resume trading |
