#!/bin/bash

echo "Adding source files in current directory..."

git add *.py *.txt *.sh 2>/dev/null

echo ""
git status
echo ""

read -p "Commit message: " msg

git commit -m "$msg"
git push

echo ""
echo "Update complete."
