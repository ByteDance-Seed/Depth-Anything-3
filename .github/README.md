# Fork Documentation

This directory contains documentation specific to this optimized fork of Depth Anything 3.

## 📚 Documents

### [FORK_VALUE.md](FORK_VALUE.md)
**Quick decision guide**: "Should I use this fork?"

- TL;DR comparison with upstream
- Use case recommendations
- Practical examples
- Performance highlights

**Read this first** if you're deciding between this fork and upstream.

---

## 📖 Related Documentation

### Root Directory Docs

- **[FORK_HIGHLIGHTS.md](../FORK_HIGHLIGHTS.md)**: Detailed technical comparison
  - Feature-by-feature breakdown
  - Performance benchmarks
  - Architecture improvements
  - Upstream sync strategy

- **[OPTIMIZATIONS.md](../OPTIMIZATIONS.md)**: Complete performance guide
  - Platform-specific optimizations
  - Configuration options
  - Benchmarking tools
  - Troubleshooting

- **[CHANGELOG.md](../CHANGELOG.md)**: Version history
  - All changes from upstream
  - Performance improvements
  - Bug fixes
  - Breaking changes

- **[README.md](../README.md)**: Main documentation
  - Quick start
  - Installation
  - Usage examples
  - Model zoo

---

## 🎯 Navigation Guide

### "I want to..."

- **...decide if this fork is for me**
  → Read [FORK_VALUE.md](FORK_VALUE.md) (5 min)

- **...understand all optimizations**
  → Read [FORK_HIGHLIGHTS.md](../FORK_HIGHLIGHTS.md) (10 min)

- **...optimize performance for my platform**
  → Read [OPTIMIZATIONS.md](../OPTIMIZATIONS.md) (15 min)

- **...see what changed**
  → Read [CHANGELOG.md](../CHANGELOG.md) (5 min)

- **...just get started**
  → Read [README.md](../README.md) Quick Start (3 min)

---

## 🔄 Document Relationships

```
README.md (main)
    ├─ Quick comparison table
    ├─ Installation
    └─ Basic usage
         │
         ├─> FORK_VALUE.md (.github/)
         │     └─ "Why use this fork?"
         │
         ├─> FORK_HIGHLIGHTS.md
         │     ├─ Technical comparison
         │     ├─ Performance data
         │     └─ When to use
         │
         ├─> OPTIMIZATIONS.md
         │     ├─ Platform guides
         │     ├─ Configuration
         │     ├─ Benchmarking
         │     └─ Troubleshooting
         │
         └─> CHANGELOG.md
               └─ Version history
```

---

## ✨ Contributing to Docs

Found a typo or have a suggestion?

1. **Small fixes**: Direct PR to this fork
2. **Upstream features**: PR to [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
3. **Optimization ideas**: Open an issue first to discuss

---

## 📝 Maintainer Notes

### Document Guidelines

- **FORK_VALUE.md**: Marketing/decision-oriented, casual tone
- **FORK_HIGHLIGHTS.md**: Technical/comprehensive, professional tone
- **OPTIMIZATIONS.md**: Tutorial/guide, instructional tone
- **CHANGELOG.md**: Factual/concise, changelog format

### Update Triggers

Update these docs when:
- [ ] Adding new optimization
- [ ] Changing default behavior
- [ ] Merging upstream changes
- [ ] Performance benchmarks change
- [ ] New platform support

### Checklist for New Release

- [ ] Update CHANGELOG.md with version
- [ ] Update performance numbers if changed
- [ ] Update FORK_HIGHLIGHTS.md comparison table
- [ ] Update README.md badges/stats
- [ ] Tag release in git
