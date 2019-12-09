@echo off

setlocal enabledelayedexpansion

cd /d %~dp0

REM 获取当前文件目录下文件名称
dir /B > 1.txt

REM 显示文件名和全路径
REM dir /S/B > 1.txt


REM dir /b /s /a:-D