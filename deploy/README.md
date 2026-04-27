# deploy/

Templates for the configs that point at local data. The real values
live in a private backup at:

```
gdrive (rclone remote `i:`):/_rk/glms-for-ilms-in-sla/deploy/
```

Run-output artifacts (`codebase/artifacts/`) live in:

```
gdrive (rclone remote `i:`):/_rk/glms-for-ilms-in-sla/resources/
```

## Restoring the working state

```bash
# 1. Get the code
git clone https://github.com/berstearns/public-glms-for-ilms-in-sla
cd public-glms-for-ilms-in-sla

# 2. Layer real configs over placeholders (drop-in patch)
rclone copy i:/_rk/glms-for-ilms-in-sla/deploy/ .

# 3. (optional) Pull run artifacts
rclone copy i:/_rk/glms-for-ilms-in-sla/resources/ .
```
