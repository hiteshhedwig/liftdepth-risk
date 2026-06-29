# GitHub Push Checklist

Before pushing:

```bash
du -sh .
find . -type f -size +50M
```

Do not accidentally push:

```text
data/
outputs/
external/
checkpoints/
large demo image folders
```

Recommended first push:

```bash
git init
git add README.md requirements.txt .gitignore LICENSE_NOTES.md docs scripts
git commit -m "Initial clean perception demo pipeline"
```

Optional later:

- Add a small GIF/video preview.
- Add screenshots to README.
- Add GitHub Pages demo only if data/license situation is acceptable.
