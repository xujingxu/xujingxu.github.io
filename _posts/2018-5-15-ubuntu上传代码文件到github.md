---
layout: post
title: "ubuntu上传代码文件到github"
date: 2018-5-15
description: "ubuntu上传代码文件到github"
tag: ubuntu
--- 

### 申请github账号，创建新的repositories,命名

### 安装git

        sudo apt-get install git
        
### 生成秘钥 

        ssh-keygen -t rsa -C "youremail@mail.com"
        
使用默认，一路回车，这样会在~/下生成.ssh文件夹，按住ctrl+h,打开id_rsa.pub,复制key

回到github，进入Account Setting，左边选择SSH Keys，Add SSH，title随便填，粘贴key.

        ssh -T git@github.com,不报错的话，表示成功连上github.
        
### 设置username and email

        git config --global user.name "xxxxxx"
        git config --global user.email "xxxxxxx@xx.com"
        
### 添加远程地址

        git remote add origin git@github.com:yourName/yourRepo.git

后面的yourName和yourRepo表示你在github的用户名和刚才新建的仓库

### 提交上传

1.创建readme

        touch README
        git add README
        git commit -m 'first commit'
        
2.上传

        git push origin master
        
注意：在上面提交命令后，可能会出现“因为您当前分支的最新提交落后于其对应的远程分支”等类似错误，

需要先获取远端更新并与本地合并，再git push。

合并操作如下：

        git fetch origin   //获取远程更新
        git merge origin/master

在git push则可以成功。

如果想要添加所有文件，可以使用“git add .”代替。

### github上传出现错误总结

1. error: src refspec master does not match any

引起该错误的原因是，目录中没有任何文件(add, commit),目录是不能提交

解决办法：

        touch README
        git add README
        git commit -m 'first commit'
        git push origin master
        
2. fatal:remote origin already exists

        git remote rm origin
        git remote add origin git@github.com:yourName/yourRepo.git
        
3. Can't push to github error:pack-objects died of signal 13

主要原因是上传文件太大，即 github limits on file sizes

解决办法：

        git config http.postBuffer 52428800
        
http.postBuffer的值的单位是字节，52428800 = 1024-1024*50 50M
