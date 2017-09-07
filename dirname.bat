@echo off
for %%I in (%1) do (
	set DIRNAME_RESULT=%%~dI%%~pI%%~nI
	set DIRNAME_EXT=%%~xI
)
echo %DIRNAME_RESULT%

rem https://stackoverflow.com/questions/3215501/batch-remove-file-extension