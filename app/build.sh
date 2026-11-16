#!/bin/zsh
# Command-line build, for those who prefer it to Xcode's Run button. Builds "Prometheus Worlds.app" into app/.
# Needs Xcode installed (it does not have to be the selected developer directory). The app runs setup.sh on first launch.
set -e
cd "$(dirname "$0")/.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
xcodebuild -quiet -project "Prometheus Worlds.xcodeproj" -scheme "Prometheus Worlds" -configuration Release -derivedDataPath .build/xcode build
rm -rf "app/Prometheus Worlds.app" && cp -R ".build/xcode/Build/Products/Release/Prometheus Worlds.app" app/
echo "built app/Prometheus Worlds.app"
