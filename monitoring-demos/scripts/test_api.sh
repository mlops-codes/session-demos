#!/bin/bash

echo "Testing ML Model API"
echo "==================="

API_URL="http://localhost:8000"

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s "$API_URL/health" | python3 -m json.tool
echo -e "\n"

# Test prediction endpoint
echo "2. Testing prediction endpoint..."
curl -s -X POST "$API_URL/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}' | python3 -m json.tool
echo -e "\n"

# Generate some load for monitoring
echo "3. Generating load for monitoring (10 requests)..."
for i in {1..10}; do
    # Generate random features using jot (macOS compatible) or fallback to static values
    if command -v jot >/dev/null 2>&1; then
        features=$(jot -r 10 1 10 | tr '\n' ',' | sed 's/,$//' | sed 's/,/, /g')
    else
        # Fallback to static random-like values
        features="$((i % 10 + 1)), $((i % 8 + 2)), $((i % 6 + 3)), $((i % 4 + 4)), $((i % 9 + 1)), $((i % 7 + 2)), $((i % 5 + 3)), $((i % 3 + 4)), $((i % 11 + 1)), $((i % 13 + 1))"
    fi
    
    curl -s -X POST "$API_URL/predict" \
         -H "Content-Type: application/json" \
         -d "{\"features\": [$features]}" > /dev/null
    echo "Request $i completed"
done

echo -e "\n4. Getting current stats..."
curl -s "$API_URL/stats" | python3 -m json.tool

echo -e "\n"
echo "API testing complete!"
echo "Check Grafana dashboard at http://localhost:3000 to see the metrics"