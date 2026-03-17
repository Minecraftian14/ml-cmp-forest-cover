@echo off
cd report
pandoc -t markdown_strict --extract-media="attachments" "main.tex" -o README.md
move attachments ..
move README.md ..\README.md
cd ..