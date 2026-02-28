#!/bin/bash
# Quick setup script for Phase 3: Embedding & Retrieval System

echo "=========================================="
echo "Phase 3: Embedding & Retrieval System"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠ .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
    echo "⚠ IMPORTANT: Edit .env and add your PINECONE_API_KEY"
    echo "   Get your key from: https://www.pinecone.io/"
    echo ""
    read -p "Press Enter after you've added your Pinecone API key..."
fi

# Check if catalog exists
if [ ! -f data/raw_catalog.json ]; then
    echo "⚠ No catalog found at data/raw_catalog.json"
    echo "   Run scraper first:"
    echo "   python -m src.data_pipeline.main --mode scrape --use-selenium"
    echo ""
    exit 1
fi

echo "Step 1: Testing Phase 3 components..."
echo "--------------------------------------"
python3 scripts/test_phase3.py
TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    echo ""
    echo "⚠ Tests failed. Please check the output above."
    echo "   Common issues:"
    echo "   - Missing PINECONE_API_KEY in .env"
    echo "   - Missing dependencies (run: pip install -r requirements.txt)"
    exit 1
fi

echo ""
echo "Step 2: Indexing assessments..."
echo "--------------------------------------"
python3 -m src.recommendation.indexer --catalog data/raw_catalog.json

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Phase 3 Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Test retrieval with sample queries"
    echo "2. Move to Phase 4: LLM Integration"
    echo ""
else
    echo ""
    echo "⚠ Indexing failed. Check the output above."
    exit 1
fi
