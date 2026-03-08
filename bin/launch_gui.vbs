Dim objShell
Set objShell = CreateObject("WScript.Shell")

Dim root
root = "C:\Users\randi\OneDrive - Clemson University\Desktop\Randi Folders\Personal\Football Ticket Tracker\cfb-ticket-tracker"

Dim python
python = root & "\.venv_win\Scripts\pythonw.exe"

objShell.Environment("Process")("PYTHONPATH") = root & "\src"
objShell.Environment("Process")("PYTHONUTF8") = "1"
objShell.Environment("Process")("PYTHONIOENCODING") = "utf-8"

' Run GUI (pythonw = no console window)
objShell.Run """" & python & """ """ & root & "\src\gui\ticket_predictor_gui.py""", 1, False
