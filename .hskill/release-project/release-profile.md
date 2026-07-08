# Release Profile
<!-- 由 project-release skill Init 阶段生成，可随时手动编辑 -->
<!-- 生成时间：2026-07-08 -->

## 分支模型

gitflow 变体：

```
feature/* / fix/* / chore/* / doc/*
        ↓ merge into
     staging
        ↓ release 分支
   release/x.y.z
        ↓ merge into
       main          ← 打 tag 在这里
```

- **发版起点**：`staging` 分支
- **release 分支命名**：`release/x.y.z`（如 `release/0.2.0`）
- **保护分支**：`main`、`staging`（不可直接提交）
- **合并流向**：`release/x.y.z` → `staging` → `main`
- **tag**：在 `main` 上打，格式 `vX.Y.Z`，必须 annotated tag

## 版本文件

| 文件 | 位置 | 更新方式 |
|------|------|---------|
| `pyproject.toml` | `version = "x.y.z"`（第 7 行） | 直接编辑 |

只有 `pyproject.toml` 一个版本声明文件，直接编辑即可。

## 发布方式

**只打 tag，不发包。**

- 本地操作完成后，推送分支和 tag：
  ```
  git push origin staging
  git push origin main
  git push origin vX.Y.Z
  ```
- 无 CI/CD，无 PyPI 发布命令。

## 特殊规则

### 发版前检查

必须全部通过后才能进入后续步骤：

```bash
ruff check .
.venv/bin/pytest -v
```

### CHANGELOG 自动生成

从 git log（conventional commits 格式）生成，写入 `CHANGELOG.md`。

格式：每个版本一个区段，格式如下：

```
## v0.2.0 — 2026-07-08

### Features
- ...

### Bug Fixes
- ...

### Chores / Other
- ...
```

按 commit type 分组（`feat` → Features，`fix` → Bug Fixes，其余 → Chores / Other）。如果 `CHANGELOG.md` 不存在则新建；存在则将新版本区段插入文件头部（已有内容保留）。

### Commit message 格式

Conventional Commits，`type[(scope)]: subject`，允许的 type：`feat fix chore docs refactor test style perf`
