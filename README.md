# 毕设

常用git命令

```C++
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


//开发分支（dev）上的代码达到上线的标准后，要合并到 master 分支

 git checkout dev
 git pull
 git checkout master
 git merge dev
 git push -u origin master
 
```

![](https://pic2.zhimg.com/v2-33046daeab21be7fea0fc6a4223bad8d_b.jpg)

可以简单的概括为：

>`git fetch`是将远程主机的最新内容拉到本地，用户在检查了以后决定是否合并到工作本机分支中。
>
>而`git pull` 则是将远程主机的最新内容拉下来后直接合并，即：`git pull = git fetch + git merge`，这样可能会产生冲突，需要手动解决。

