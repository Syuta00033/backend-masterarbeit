Beim ersten Mal verwenden: In die Konsole .\generate-cert.bat ausführen und überall Enter drücken
Dann im Browser https://localhost:8000 eingeben und Zertifikat bestätigen. 
Erst dann funktioniert das Backend

venv: 
python -m venv .venv
.venv\Scripts\Activate
python -m pip install --upgrade pip

requirements installieren: pip install -r requirements.txt