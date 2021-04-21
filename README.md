# 毕设

常用git命令

```bash
// 在当前目录新建一个 Git 代码库
git init

// 为该本地库删除远程库
git remote rm origin
// 添加远程库
git remote add origin **.git

# 远程同步：
# 下载远程仓库的所有变动
$ git fetch [remote]

# 显示所有远程仓库
$ git remote -v

# 强行推送当前分支到远程仓库，即使有冲突
$ git push [remote] --force

# 推送所有分支到远程仓库
$ git push [remote] --all

//在 GitHub 或 码云 刚刚创建仓库第一次pull的时候，两个仓库的差别非常大，所以git拒绝合并两个不相干的东西
git pull
git pull origin master
git pull origin master --allow-unrelated-histories
```

