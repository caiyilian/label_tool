# GitHub 命令行操作指南

本文档整理了使用 Git 和 GitHub CLI (`gh`) 进行代码版本控制和远程仓库管理的标准工作流。

---

## 0. 前置准备：安装与登录 GitHub CLI

在执行任何与 GitHub 远程交互的操作前，必须确保安装了 `gh` 工具并完成了账号授权。

### 安装 GitHub CLI (Windows)
在终端中执行以下命令：
```powershell
winget install --id GitHub.cli
```
*(注：安装后建议重启终端使环境变量生效。如果提示找不到命令，可直接使用绝对路径 `& "C:\Program Files\GitHub CLI\gh.exe"` 代替 `gh`)*

### 登录授权
```powershell
gh auth login
```
**交互选择推荐：**
1. 选 `GitHub.com`
2. 选 `HTTPS`
3. 选 `Yes` (Authenticate Git with your GitHub credentials)
4. 选 `Login with a web browser` (随后在弹出的浏览器页面中输入终端显示的 8 位验证码即可完成登录)

---

## 场景一：远程还没有仓库，本地是新项目（创建并上传）

这是从零开始将本地代码推送到 GitHub 的最快方式。

**1. 初始化本地 Git 仓库并提交代码**
```powershell
# 初始化本地仓库
git init

# 将当前目录下所有文件（除了 .gitignore 中排除的）加入暂存区
git add .

# 创建首次本地提交
git commit -m "Initial commit: 项目初始化"
```

**2. 使用 gh 一键创建远程仓库并推送**
```powershell
gh repo create <你的仓库名> --public --source=. --remote=origin --push
```
- `--public`：设为公开仓库（若要私有则改为 `--private`）。
- `--source=.`：将当前本地目录作为源代码源。
- `--push`：创建完远程仓库后，立刻自动把本地代码推上去。

---

## 场景二：远程已有仓库，且已与本地连接（日常更新代码）

当你的项目已经成功上传过一次（完成了场景一或三），之后你修改了代码，需要把新代码同步到 GitHub。

**执行常规的 Git 三步曲即可：**
```powershell
# 1. 将所有修改过的文件加入暂存区
git add .

# 2. 将修改提交到本地版本库（-m 后面写这次修改了什么内容）
git commit -m "feat: 增加了分辨率调整功能和进度条"

# 3. 推送到远程仓库的当前分支（通常是 master 或 main）
git push
```
*(注：如果首次推送时没有绑定上游分支，系统会提示你运行类似 `git push --set-upstream origin master` 的命令，直接复制执行即可。之后再更新就只需 `git push` 了)*

---

## 场景三：远程有同名仓库，但本地还未连接（关联并上传）

有时候你可能在 GitHub 网页端手动点击了 "New repository" 创建了一个空仓库，或者你在另一台电脑上写了代码，需要将本地代码推送到这个已经存在的远程空仓库里。

**1. 初始化并提交本地代码（如果本地还没初始化过 Git）**
```powershell
git init
git add .
git commit -m "Initial commit"
```

**2. 将本地仓库与远程仓库关联**
你需要知道远程仓库的 HTTPS 链接（在 GitHub 仓库页面的绿色 "Code" 按钮下可以复制）。
```powershell
git remote add origin https://github.com/<你的用户名>/<你的仓库名>.git
```

**3. 将本地代码推送到远程**
```powershell
# 强制将本地的 master 分支推送到远程的 origin 仓库
git push -u origin master
```

**⚠️ 常见冲突处理：**
如果在 GitHub 网页端创建仓库时，你勾选了“添加 README”或“添加 .gitignore”，导致远程仓库里**已经有文件了**，此时直接 `push` 会报错（提示远程包含本地没有的工作）。
解决方法（二选一）：
- **合并远程代码到本地再推送（推荐）**：
  ```powershell
  git pull origin master --allow-unrelated-histories
  git push -u origin master
  ```
- **暴力覆盖远程代码（警告：远程原有的文件会被本地覆盖删除）**：
  ```powershell
  git push -u origin master -f
  ```