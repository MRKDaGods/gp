#!/bin/bash
# Startup script for MTMC Tracker (Backend + Frontend)

echo "🚀 Starting MTMC Tracker System..."
echo "=================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo -e "${YELLOW}⚠️  Python not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}⚠️  Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi

echo -e "${BLUE}📦 Installing backend dependencies...${NC}"
pip install -r backend_requirements.txt -q

echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi
cd ..

echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Check if models are downloaded
if [ ! -d "models" ]; then
    echo -e "${YELLOW}⚠️  Models not found!${NC}"
    echo "Please download models from: https://www.kaggle.com/datasets/mrkdagods/mtmc-weights"
    echo "Extract to: ./models/"
    echo ""
    read -p "Press Enter to continue without models (demo mode) or Ctrl+C to exit..."
fi

echo -e "${BLUE}🔧 Starting Backend API Server (port 8000)...${NC}"
python backend_api.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

echo -e "${BLUE}🎨 Starting Frontend Dev Server (port 3000)...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}✅ System is running!${NC}"
echo "=================================="
echo -e "Backend API:  ${BLUE}http://localhost:8000/docs${NC}"
echo -e "Frontend UI:  ${BLUE}http://localhost:3000${NC}"
echo -e "Health Check: ${BLUE}http://localhost:8000/api/health${NC}"
echo ""
echo "Press Ctrl+C to stop both servers..."

# Trap Ctrl+C and cleanup
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

# Wait for processes
wait
