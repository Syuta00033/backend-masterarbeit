@echo off
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
echo Fertig! key.pem und cert.pem wurden erstellt.
pause
