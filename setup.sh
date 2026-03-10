#!/usr/bin/env bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# Portable sed -i helper
sed_i() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    sed -i "" "$@"
  else
    sed -i "$@"
  fi
}

echo ""
echo "========================================="
echo "  MetaBot Setup"
echo "========================================="
echo ""

# ---- 1. Check Node.js ----
info "Checking Node.js..."
if ! command -v node &>/dev/null; then
  fail "Node.js is not installed. Please install Node.js 18+ first: https://nodejs.org"
fi
NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VER" -lt 18 ]; then
  fail "Node.js 18+ is required (found: $(node -v))"
fi
ok "Node.js $(node -v)"

# ---- 2. Check Codex CLI ----
info "Checking Codex CLI..."
if ! command -v codex &>/dev/null; then
  fail "Codex CLI not found. Please install Codex first, then re-run setup."
fi
ok "Codex CLI: $(which codex)"
echo ""
warn "Make sure Codex is authenticated (run 'codex login' in a separate terminal)"
echo ""

# ---- 3. Install dependencies ----
info "Installing npm dependencies..."
npm install
ok "Dependencies installed"

# ---- 4. Config files ----
if [ ! -f .env ]; then
  cp .env.example .env
  ok "Created .env from template"
else
  ok ".env already exists"
fi

if [ ! -f bots.json ]; then
  echo ""
  echo "-----------------------------------------"
  echo "  Bot Configuration"
  echo "-----------------------------------------"
  echo ""

  read -p "Feishu App ID: " APP_ID
  read -p "Feishu App Secret: " APP_SECRET
  read -p "Working directory (absolute path): " WORK_DIR

  # Expand ~ if present
  WORK_DIR="${WORK_DIR/#\~/$HOME}"

  if [ ! -d "$WORK_DIR" ]; then
    warn "Directory '$WORK_DIR' does not exist, creating it..."
    mkdir -p "$WORK_DIR"
  fi

  cat > bots.json <<BOTEOF
[
  {
    "name": "default",
    "feishuAppId": "$APP_ID",
    "feishuAppSecret": "$APP_SECRET",
    "defaultWorkingDirectory": "$WORK_DIR",
    "allowedTools": ["Read", "Edit", "Write", "Glob", "Grep", "Bash", "WebSearch", "WebFetch"]
  }
]
BOTEOF

  # Set BOTS_CONFIG in .env
  if grep -q "^BOTS_CONFIG=" .env; then
    sed_i "s|^BOTS_CONFIG=.*|BOTS_CONFIG=./bots.json|" .env
  else
    echo "BOTS_CONFIG=./bots.json" >> .env
  fi

  ok "Created bots.json"
else
  ok "bots.json already exists"
fi

# ---- 5. Install PM2 ----
info "Checking PM2..."
if ! command -v pm2 &>/dev/null; then
  info "Installing PM2..."
  npm install -g pm2
fi
ok "PM2: $(pm2 -v)"

# ---- 6. Start service ----
echo ""
info "Starting service with PM2..."

# Stop existing instance if running
pm2 delete metabot 2>/dev/null || true
pm2 start ecosystem.config.cjs

echo ""
echo "========================================="
echo -e "  ${GREEN}Setup complete!${NC}"
echo "========================================="
echo ""
echo "  Service is running. Useful commands:"
echo ""
echo "    pm2 status          # View status"
echo "    pm2 logs            # View logs"
echo "    pm2 restart all     # Restart"
echo "    pm2 stop all        # Stop"
echo ""
echo "  Auto-start on reboot (run once):"
echo ""
echo "    pm2 startup"
echo "    pm2 save"
echo ""
echo "  Next steps:"
echo "    1. Make sure Codex CLI is authenticated (codex login)"
echo "    2. Open Feishu and send a message to your bot"
echo ""
