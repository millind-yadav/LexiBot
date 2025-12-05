#!/bin/bash

echo "🚀 Legal AI Frontend Testing Script"
echo "=================================="

# Navigate to frontend directory
cd /Users/milind/Documents/lexibot-agent-project/frontend

echo "✅ Current directory: $(pwd)"

# Check if package.json exists
if [ -f "package.json" ]; then
    echo "✅ package.json found"
else
    echo "❌ package.json not found"
    exit 1
fi

# Check if app directory exists
if [ -d "app" ]; then
    echo "✅ app directory found"
    ls -la app/
else
    echo "❌ app directory not found"
    exit 1
fi

# Check if legal components exist
if [ -f "app/legal/page.tsx" ]; then
    echo "✅ Legal AI page found at app/legal/page.tsx"
else
    echo "❌ Legal AI page not found"
fi

if [ -f "components/legal/contract-analyzer.tsx" ]; then
    echo "✅ Contract analyzer component found"
else
    echo "❌ Contract analyzer component not found"
fi

if [ -f "components/legal/contract-comparison.tsx" ]; then
    echo "✅ Contract comparison component found"
else
    echo "❌ Contract comparison component not found"
fi

# Check API routes
if [ -f "app/api/legal/analyze-contract/route.ts" ]; then
    echo "✅ Contract analysis API route found"
else
    echo "❌ Contract analysis API route not found"
fi

echo ""
echo "🔧 Next Steps:"
echo "1. Run: npm install (if needed)"
echo "2. Run: npm run dev"
echo "3. Open: http://localhost:3000/legal"
echo ""
echo "📋 Test Cases to Try:"
echo "1. Contract Analysis:"
echo "   - Paste sample contract text"
echo "   - Choose 'Comprehensive Analysis'"
echo "   - Click 'Analyze Contract'"
echo "   - Should see mock analysis results"
echo ""
echo "2. Contract Comparison:"
echo "   - Paste two different contract texts"
echo "   - Click 'Compare Contracts'"
echo "   - Should see detailed comparison"
echo ""
echo "✨ Features to Test:"
echo "- File upload (.txt files)"
echo "- Custom questions input"
echo "- Risk level badges"
echo "- Copy to clipboard"
echo "- Export functionality"