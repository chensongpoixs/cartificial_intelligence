@echo off

setlocal

call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" x64

cd /d %~dp0
devenv machine_learning.sln /build "Debug|x86" /project machine_learning



if "%1" == "" pause