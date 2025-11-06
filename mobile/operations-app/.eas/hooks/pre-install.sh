#!/bin/bash
echo "📦 Running pre-install.sh"

if [ -z "$GOOGLE_SERVICES_JSON" ]; then
  echo "❌ GOOGLE_SERVICES_JSON not found."
  exit 1
fi

echo "$GOOGLE_SERVICES_JSON" | base64 --decode > android/app/google-services.json

echo "✅ google-services.json created successfully."
