# anaconda.navigator 使用


```
#查看conda版本
conda --version


#2.2 查看conda的环境配置
conda config --show


#2.6 更新Anaconda整体
#将整个Anaconda都更新到确保稳定性和兼容性的最新版本
conda update Anaconda

#2.7 查询某个命令的帮助 
conda create --help



#3.1 创建虚拟环境
#        使用conda创建虚拟环境的命令格式为:

conda create -n env_name python=3.8


#3.2 创建虚拟环境的同时安装必要的包
 #       但是并不建议这样做，简化每一条命令的任务在绝大多数时候都是明智的（一个例外是需要反复执行的脚本）

conda create -n env_name numpy matplotlib python=3.8


#3.3 查看有哪些虚拟环境
 #       以下三条命令都可以。注意最后一个是”--”，而不是“-”.
conda env list
conda info -e
conda info --envs


#3.4 激活虚拟环境
   #     使用如下命令即可激活创建的虚拟环境。
conda activate env_name


#3.5 退出虚拟环境
#        使用如下命令即可退出当前工作的虚拟环境。

conda activate
conda deactivate

#3.5 删除虚拟环境
  #      执行以下命令可以将该指定虚拟环境及其中所安装的包都删除。
conda remove --name env_name --all


#3.6 导出环境 
#       很多的软件依赖特定的环境，我们可以导出环境，这样方便自己在需要时恢复环境，也可以提供给别人用于创建完全相同的环境。

#获得环境中的所有配置
conda env export --name myenv > myenv.yml
#重新还原环境
conda env create -f  myenv.yml 
```


