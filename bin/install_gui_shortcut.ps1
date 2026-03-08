# bin/install_gui_shortcut.ps1
# Creates a desktop shortcut to launch the CFB Ticket Tracker GUI.
# Run once: .\bin\install_gui_shortcut.ps1

$Root    = Split-Path $PSScriptRoot -Parent
$VBS     = "$Root\bin\launch_gui.vbs"
$Icon    = "$Root\assets\icons\cfb-tix.ico"
$Desktop = [Environment]::GetFolderPath("Desktop")
$Link    = "$Desktop\CFB Ticket Tracker.lnk"

$WS = New-Object -ComObject WScript.Shell
$SC = $WS.CreateShortcut($Link)
$SC.TargetPath       = "wscript.exe"
$SC.Arguments        = """$VBS"""
$SC.WorkingDirectory = $Root
$SC.IconLocation     = $Icon
$SC.Description      = "CFB Ticket Price Tracker GUI"
$SC.Save()

Write-Host "Shortcut created: $Link"
