@echo off
cd report
pandoc -t markdown_strict --extract-media="../attachments" "main.tex" -o ../README.md
cd ..