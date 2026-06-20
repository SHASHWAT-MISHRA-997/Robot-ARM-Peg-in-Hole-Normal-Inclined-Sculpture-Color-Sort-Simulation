Option Explicit

Dim shell, fso, projectDir, pythonExe, appPy

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe = """" & projectDir & "\venv\Scripts\pythonw.exe"""
appPy = """" & projectDir & "\app.py"""

shell.CurrentDirectory = projectDir
shell.Run pythonExe & " " & appPy, 0, False

WScript.Sleep 2500
shell.Run "http://127.0.0.1:5000", 1, False
