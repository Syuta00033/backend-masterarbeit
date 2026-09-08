# Systembeschreibung — Grundlage für Kapitel 3 (Konzeption) und 4 (Umsetzung)

Stand: 08.08.2026. Alle Angaben aus dem Quelltext verifiziert.
Dateipfade relativ zu `backend-masterarbeit/` bzw. `frontend-masterarbeit/`.

---

## 1 Architektur

### 1.1 Komponenten

Das System besteht aus drei Teilen:

| Komponente | Ort | Aufgabe |
|---|---|---|
| Frontend | `frontend-masterarbeit/src/` | Videoauswahl, Technikwahl, Fortschritt, Ergebnisdarstellung, Verlauf |
| Web-API | `backend-masterarbeit/main.py` | Upload, Auftragsverwaltung, Auslieferung von Video und Bewertung |
| Analysepipeline | `backend-masterarbeit/kick_analyzer.py`, `kicks/` | Pose Estimation, Phasenerkennung, Bewertung |

Das Frontend wird im Betrieb vom Backend statisch mit ausgeliefert
(`app.mount("/", StaticFiles(...))` am Ende von `main.py`), sodass Anwendung und
API unter derselben Adresse laufen.

### 1.2 Technologie-Stack

**Backend**
- Python 3.14.3
- FastAPI 0.135.3, Uvicorn 0.44.0
- MediaPipe 0.10.35 (installiert; `requirements.txt` nennt 0.10.33 — Abweichung)
- OpenCV 4.13.0 (`opencv-contrib-python`)
- NumPy 2.4.4
- `imageio-ffmpeg` 0.5.1 für die Nachkodierung

**Frontend**
- React 19.2.4, TypeScript 6.0.2, Vite 8.0.4
- Progressive Web App (`public/manifest.json`, `display: standalone`)

> **Hinweis:** MediaPipe unterstützt offiziell nur Python bis 3.12. Der Betrieb
> unter 3.14 funktioniert, ist aber nicht abgesichert — gehört in die Limitationen.

### 1.3 Ort der Pose Estimation

Die Pose Estimation läuft **serverseitig** in Python. Das Video wird vollständig
hochgeladen und im Backend Frame für Frame verarbeitet.

> Im Frontend ist zwar `@mediapipe/tasks-vision` 0.10.34 als Abhängigkeit
> eingetragen und es existiert eine Datei `src/pages/LiveCam.tsx` mit
> browserseitiger Kameraauswertung. Diese Seite ist in `App.tsx` noch verdrahtet,
> aber **nicht mehr über die Navigation erreichbar** (`NavBar.tsx` listet nur
> Home, Pose Analysis, Verlauf). Für die Beschreibung des ausgelieferten Systems
> ist sie ohne Bedeutung; erwähne sie nur, wenn du den Entwicklungsverlauf
> darstellen willst.

### 1.4 Datenfluss

```
Nutzer wählt Technik + Modell + Videodatei
        │
        ├─► POST /upload                    (multipart)      → job_id, Datei in TEMP
        │
        ├─► POST /process/{job_id}?model=&kick_type=
        │        └─ startet Hintergrund-Thread process_video()
        │
        ├─► GET /status/{job_id}            (Polling, 800 ms) → progress / total / status
        │
        ├─► GET /analysis/{job_id}                            → Bewertungsobjekt (JSON)
        ├─► GET /download/{job_id}                            → annotiertes Video (MP4)
        │
        └─► POST /save/{job_id}?client={uuid}                 → persistiert in saved/
             GET /saved?client={uuid}                         → Verlauf des Geräts
```

Relevante Funktionen: `upload_video()`, `start_processing()`, `get_status()`,
`get_analysis()`, `download_video()`, `save_kick()`, `list_saved()` — alle in `main.py`.
Frontend-Gegenstück: `handleFileSelect()`, `startPolling()`, `handleSave()` in
`src/pages/PoseAnalysis.tsx`.

Die Auftragsverwaltung liegt in einem Prozess-lokalen Dictionary `jobs` (`main.py`).
Ein Neustart des Servers verwirft alle laufenden und noch nicht gespeicherten Analysen.

### 1.5 Verarbeitung eines Videos

`process_video()` in `main.py`:

1. Video mit OpenCV öffnen, Bildrate auslesen (`CAP_PROP_FPS`, Rückfall auf 30)
2. Ausgabegröße auf **halbe Breite und Höhe** festlegen
3. `PoseLandmarker` im Modus `VIDEO` erzeugen
4. Je Frame: verkleinern → BGR nach RGB → `detect_for_video()` → Skelett zeichnen →
   `KickAnalyzer.process_frame()` → in die Ausgabedatei schreiben
5. Nach dem letzten Frame: `kick.evaluate()` und `build_summary()`
6. Nachkodierung mit ffmpeg (libx264, `-crf 23`, `-preset fast`, `+faststart`)

---

## 2 Pose Estimation

### 2.1 Modell

MediaPipe Tasks **Pose Landmarker** (BlazePose GHUM), Laufmodus `VisionRunningMode.VIDEO`.
Drei Modellvarianten liegen unter `models/` und sind im UI wählbar:

| Variante | Datei | Größe |
|---|---|---|
| lite | `pose_landmarker_lite.task` | 5,8 MB |
| full | `pose_landmarker_full.task` | 9,4 MB |
| heavy | `pose_landmarker_heavy.task` | 30,7 MB |

Voreinstellung im Frontend ist **full** (`selectedModel` in `PoseAnalysis.tsx`).
Die Auswahl wird als Query-Parameter an `/process` übergeben.

Konfidenzschwellen (`PoseLandmarkerOptions` in `main.py`), alle drei auf 0.6:
`min_pose_detection_confidence`, `min_pose_presence_confidence`, `min_tracking_confidence`.

### 2.2 Verwendete Landmarken

Von den 33 Landmarken werden **acht** ausgewertet (`kicks/geometry.py`, Zeilen 3–6):

| Index | Bezeichner | Verwendung |
|---|---|---|
| 11, 12 | `L_SHOULDER`, `R_SHOULDER` | Hüftbeugung (Gelenkwinkel Schulter–Hüfte–Knie) |
| 23, 24 | `L_HIP`, `R_HIP` | Kniewinkel, Beckenausrichtung, Beinlänge |
| 25, 26 | `L_KNEE`, `R_KNEE` | Kniewinkel, Oberschenkelneigung, Kniehöhe |
| 27, 28 | `L_ANKLE`, `R_ANKLE` | Kickrichtung, Fußbahn, Geschwindigkeit |

Fuß-Landmarken (Ferse, Zehen) werden bewusst nicht verwendet.

Alle Berechnungen nutzen `result.pose_world_landmarks[0]` — die metrischen
Weltkoordinaten mit Ursprung in der Hüftmitte. Die normalisierten Bildkoordinaten
(`pose_landmarks`) dienen ausschließlich dem Zeichnen des Skeletts.

> Bei mehreren Personen im Bild wird stets `[0]` verwendet, also die erste erkannte
> Person. Eine Auswahl oder Warnung findet nicht statt.

### 2.3 Umgang mit Sichtbarkeitswerten

**Die Sichtbarkeitswerte werden nicht zur Filterung verwendet.** Sie werden
ausschließlich im Debug-Overlay des Ausgabevideos angezeigt (`kick_analyzer.py`,
Zeile 52 ff.): Minimum und Mittelwert über die sechs Bein-Landmarken, farblich
codiert (grün ab 0,7 / gelb ab 0,5 / rot darunter).

Frames mit niedriger Sichtbarkeit gehen also unverändert in alle Messgrößen ein.
Das ist eine bekannte und benennbare Schwäche — eine Filterung wäre der
naheliegende nächste Schritt.

### 2.4 Normalisierung

| Aspekt | Umsetzung |
|---|---|
| Translation | Durch MediaPipe selbst: Weltkoordinaten sind hüftzentriert |
| Skalierung, Winkel | Nicht nötig — Gelenkwinkel sind skaleninvariant |
| Skalierung, Fußbahn (Dwit) | Auf die Beinlänge normiert: `leg_len` = Hüfte→Knie + Knie→Knöchel des Standbeins, siehe `_compute_current_values()` in `kicks/dwit_chagi.py` |
| Skalierung, Kniehöhe | **Nicht normiert** — `_knee_drop()` liefert absolute Meter |

Die uneinheitliche Normierung ist ein bewusst zu benennender Punkt: Die
Fußbahn-Kriterien des Spinning Back Kick sind körpergrößenunabhängig, das
Kniehöhen-Kriterium von Front- und Roundhouse Kick ist es nicht.

### 2.5 Glättung und Filterung

| Größe | Verfahren | Ort |
|---|---|---|
| Fußgeschwindigkeit | Gleitender Median über 3 Frames (`deque(maxlen=3)`) | alle drei Klassen, `_compute_current_values()` |
| Standbeinwinkel | Median über alle Frames der Kick-Phase | `evaluate()`, `standing_knee_history` |
| Beckendrehung (Dwit) | Schrittweise Aufsummierung, Schritte über `MAX_STEP_DEG = 30` werden verworfen | `dwit_chagi.py`, Zeile 135 |
| Hüftausrichtung (Dwit) | Sprungfilter: Änderungen über `JUMP_FILTER_DEG = 60` werden verworfen, letzter gültiger Wert bleibt | `dwit_chagi.py`, Zeile 141 |

Die übrigen Winkel (Kniewinkel, Hüftbeugung, Oberschenkelneigung, Hüftausrichtung
bei Front- und Roundhouse Kick) werden **ungeglättet** verwendet.

Begründung der Sprungfilter: Eine Winkeländerung von 60° zwischen zwei Frames
entspricht bei 30 fps über 1800°/s und ist physiologisch unmöglich. Sie kann daher
nur aus einer Vertauschung linker und rechter Landmarken stammen.

---

## 3 Phasenerkennung

### 3.1 Zustandsmaschine

Jede Technik ist als eigene Klasse mit eigener Zustandsmaschine umgesetzt
(`kicks/ap_chagi.py`, `kicks/bandal_chagi.py`, `kicks/dwit_chagi.py`).
Der Dispatcher `KickAnalyzer` (`kick_analyzer.py`) wählt anhand des vom Nutzer
gesetzten `kick_type` die passende Klasse aus `KICK_CLASSES`.

| Technik | Phasenfolge |
|---|---|
| Front Kick / Roundhouse Kick | idle → chamber → kick → rechamber → idle |
| Spinning Back Kick | idle → **rotation** → chamber → kick → rechamber → idle |

Jeder Phasenwechsel wird in `phases_log` mit Framenummer protokolliert
(`_transition_to()`), zusätzlich setzt der Wechsel in die Kick-Phase das Flag
`kick_detected`.

### 3.2 Bestimmung der Kickseite

`KickAnalyzer._select_side()`: Solange die Phase `idle` ist, wird die Höhendifferenz
der beiden Knie geprüft. Übersteigt sie `SIDE_LIFT_THRESHOLD = 0,05 m`, gilt das
höhere Knie als Kickbein. **Die Seite wird danach nicht mehr zurückgesetzt**, das
System ist daher auf einen Kick pro Video ausgelegt.

### 3.3 Übergangsbedingungen im Einzelnen

**Front Kick** (`ap_chagi.py`)

| Übergang | Bedingung |
|---|---|
| idle → chamber | Kniewinkel < 100° **und** Oberschenkelneigung > 45° |
| chamber → kick | Kniewinkel > 150° **und** Oberschenkelneigung > 45° |
| kick → rechamber | Oberschenkelneigung > 45° **und** (max. Kniewinkel − aktueller) ≥ 80° |
| → idle (Abbruch) | Kniewinkel > 150° **und** Oberschenkelneigung < 50° |

**Roundhouse Kick** (`bandal_chagi.py`) — identische Struktur, abweichende Werte:
Chamber-Schwelle 130°, Kick-Schwelle 150°, Idle-Schwelle 160°.

**Spinning Back Kick** (`dwit_chagi.py`)

| Übergang | Bedingung |
|---|---|
| idle → rotation | aufsummierte Beckendrehung > 45° |
| idle → chamber (Rückfall) | Kniewinkel < 90° **und** Hüftbeugung < 150° |
| rotation → chamber | Kniewinkel < 90° |
| chamber → kick | horizontaler Knöchelabstand > 0,45 Beinlängen **und** Fußhöhe > 0,30 Beinlängen |
| chamber → idle (Abbruch) | Fuß war oben (> 0,30) und ist jetzt unten (< 0,15) |
| kick → rechamber | Fußhöhe > 0,25 Beinlängen **und** (max. Kniewinkel − aktueller) ≥ 35° |
| → idle | Kniewinkel > 160° **und** Hüftbeugung > 160° |

### 3.4 Treffmoment

Ein expliziter Treffmoment wird **ausschließlich beim Spinning Back Kick** bestimmt.
In `_update_kick()` (`dwit_chagi.py`, Zeile 236 ff.) gilt jener Frame als Treffmoment,
in dem der Kniewinkel seinen bisherigen Höchstwert übertrifft; dort wird
`hip_alignment_at_impact` festgehalten.

Bei Front- und Roundhouse Kick gibt es **keinen** definierten Treffmoment. Alle
Kriterien arbeiten mit Extrem- oder Medianwerten über eine ganze Phase.

Diese Asymmetrie ist erklärungsbedürftig und sollte in der Arbeit begründet werden:
Der Treffmoment wurde beim Spinning Back Kick eingeführt, weil ein Maximum über die
gesamte Kick-Phase einen durchgedrehten Spinning Side Kick fälschlich als korrekt
bewertet hatte (der Körper durchläuft dabei kurzzeitig die Idealausrichtung).

### 3.5 Technikspezifische Sonderregeln

- Nur der Spinning Back Kick besitzt eine Rotationsphase und führt eine
  aufsummierte Beckendrehung (`total_rotation`) mit.
- Nur der Spinning Back Kick erkennt den Kickbeginn über die Fußbahn statt über den
  Kniewinkel. Grund: Beim Back Kick streckt sich das Knie kaum, ein Kniewinkel-
  Kriterium könnte einen schwachen Kick nicht vom Absetzen des Beins unterscheiden.
- Der Test „Bein ist oben" (`_leg_is_up()`) ist unterschiedlich realisiert: Front-
  und Roundhouse Kick über die Oberschenkelneigung, Spinning Back Kick über die
  normierte Fußhöhe (dort ist die Hüfte gestreckt und als Indikator unbrauchbar).

---

## 4 Bewertungsregeln

### 4.1 Bewertungsfunktion

`score_linear(value, fail_at, ideal_at)` in `kicks/geometry.py`:

```python
t = (value - fail_at) / (ideal_at - fail_at)
return max(0.0, min(1.0, t)) * 100
```

Der Messwert wird linear auf 0–100 % abgebildet und an beiden Enden begrenzt.
`fail_at` und `ideal_at` können in beliebiger Reihenfolge stehen, dadurch lassen
sich sowohl „kleiner ist besser" als auch „größer ist besser" abbilden.
Ein Kriterium gilt als bestanden ab 50 % (`passed = score >= 50`), was allein die
Auswahl der Rückmeldungstexte steuert.

### 4.2 Winkelberechnung

Alle Gelenkwinkel werden **dreidimensional** berechnet (`calc_angle(a, b, c)`,
`geometry.py`): Winkel am Scheitelpunkt `b` über das normierte Skalarprodukt der
Vektoren `b→a` und `b→c`, ausgewertet mit `arccos`.

| Größe | Punkte / Definition | Dimension |
|---|---|---|
| Kniewinkel | Hüfte – Knie – Knöchel | 3D |
| Hüftbeugung (Gelenkwinkel) | Schulter – Hüfte – Knie | 3D |
| Oberschenkelneigung | Winkel des Vektors Hüfte→Knie gegen die Vertikale (0, 1, 0) | 3D |
| Hüftausrichtung | Winkel zwischen Becken-Blickrichtung und Kickrichtung | **2D**, x-z-Ebene (Draufsicht) |

Die Becken-Blickrichtung (`pelvis_facing()`) entsteht als Senkrechte auf die
Verbindungslinie beider Hüften in der Draufsicht; die Kickrichtung
(`kick_direction()`) als Vektor Hüfte→Knöchel des Kickbeins, ebenfalls in der
Draufsicht.

**Wichtig für die Interpretation:** Die Hüftausrichtung ist ein *körperrelatives*
Maß. Sie beschreibt die Lage des Beins zum Becken, nicht die Orientierung im Raum.
Eine Drehung des gesamten Körpers verändert den Wert nicht. Genau darauf beruht die
Unterscheidung der drei Techniken:

| Bein relativ zum Becken | Wert | Technik |
|---|---|---|
| in Blickrichtung | ~0° | Front Kick |
| seitlich | ~90° | Side Kick |
| nach hinten | ~180° | Back Kick |

Bei Front- und Roundhouse Kick wird der Wert auf 0–90° gefaltet
(`min(raw, 180 - raw)`), um gegen Links/Rechts-Vertauschungen der Hüften unempfindlich
zu sein. Beim Spinning Back Kick wird der volle Bereich 0–180° benötigt und
stattdessen der Sprungfilter eingesetzt.

### 4.3 Kriterien und Schwellenwerte

**Front Kick — 7 Kriterien** (`ap_chagi.py`, `evaluate()`)

| # | Kriterium | Messgröße | fail_at | ideal_at |
|---|---|---|---|---|
| 1 | Beinstreckung | max. Kniewinkel in der Kick-Phase | 130° | 160° |
| 2 | Chamber Winkel | min. Kniewinkel in der Chamber-Phase | 100° | 60° |
| 3 | Hüftbeugung | max. Oberschenkelneigung in der Chamber-Phase | 50° | 90° |
| 4 | Hüftrotation | max. Hüftausrichtung (gefaltet) in der Kick-Phase | 90° | 30° |
| 5 | Kniehöhe | Absinken des Knies während des Kicks | 0,30 m | 0,15 m |
| 6 | Standbein | Median des Standbein-Kniewinkels in der Kick-Phase | 180° | 165° |
| 7 | Rechamber | Rückzug des Knies vom Streckungsmaximum | 10° | 60° |

**Roundhouse Kick — 7 Kriterien** (`bandal_chagi.py`)

| # | Kriterium | Messgröße | fail_at | ideal_at |
|---|---|---|---|---|
| 1 | Beinstreckung | max. Kniewinkel in der Kick-Phase | 130° | 160° |
| 2 | Hüftrotation | max. Hüftausrichtung (gefaltet) | 30° | 70° |
| 3 | Chamber Winkel | min. Kniewinkel in der Chamber-Phase | 130° | 90° |
| 4 | Hüftbeugung | max. Oberschenkelneigung | 50° | 90° |
| 5 | Kniehöhe | Absinken des Knies | 0,30 m | 0,15 m |
| 6 | Standbein | Median Standbein-Kniewinkel | 180° | 165° |
| 7 | Rechamber | Rückzug vom Streckungsmaximum | 10° | 60° |

**Spinning Back Kick — 5 Kriterien** (`dwit_chagi.py`)

| # | Kriterium | Messgröße | fail_at | ideal_at |
|---|---|---|---|---|
| 1 | Chamber Winkel | min. Kniewinkel in der Chamber-Phase | 90° | 55° |
| 2 | Beinstreckung | max. Kniewinkel in der Kick-Phase | 90° | 140° |
| 3 | Körperdrehung | Hüftausrichtung im Treffmoment | 110° | 170° |
| 4 | Standbein | Median Standbein-Kniewinkel | 180° | 165° |
| 5 | Rechamber | Rückzug vom Streckungsmaximum | 10° | 60° |

Beim Spinning Back Kick wählt zusätzlich die aufsummierte Nachdrehung
(`max_over_rotation`, Schwelle 60°) zwischen zwei Rückmeldungstexten für die
Körperdrehung — sie fließt jedoch **nicht** in die Punktzahl ein.

### 4.4 Abbruchbedingungen in der Bewertung

Alle drei Klassen brechen `evaluate()` vorzeitig ab, wenn keine Kick-Phase erkannt
wurde (`if not self.kick_detected: return results`). Dann werden nur die zuvor
angehängten Kriterien zurückgegeben.

Der Roundhouse Kick besitzt einen **zusätzlichen** Frühabbruch: Liegt die
Hüftrotation unter 30°, endet die Bewertung nach drei Kriterien
(`bandal_chagi.py`, Zeile 255).

> Beide Abbrüche verzerren den Gesamtwert, weil dieser als Mittelwert über die
> *zurückgegebenen* Kriterien gebildet wird. Bei drei Kriterien, von denen zwei
> bei 100 % liegen und eines bei 0 %, ergibt sich rechnerisch 67 %. Das ist ein
> bekanntes offenes Problem und gehört in die Diskussion.

### 4.5 Gesamtergebnis

`build_summary(criteria, velocity)` in `main.py`:

- **Gesamtwert**: ungewichteter arithmetischer Mittelwert aller Kriterienwerte, gerundet
- **Sterne**: `round(overall / 20)`, begrenzt auf 1 bis 5
- **Tipps**: Rückmeldungstexte aller Kriterien mit `passed == False`
- **Geschwindigkeit**: höchste geglättete Fußgeschwindigkeit in km/h

**Eine Gewichtung findet nicht statt.** Alle Kriterien gehen gleich stark ein.
Praktische Folge: Ein technisch sauber ausgeführter, aber falscher Kicktyp erreicht
beim Spinning Back Kick noch rund 78 %, weil nur ein Kriterium von fünf die
Technikart erfasst.

### 4.6 Geschwindigkeitsmessung

Die Fußgeschwindigkeit wird aus der Positionsänderung des Kickknöchels zwischen zwei
Frames berechnet (`dist * fps * 3.6`), über drei Frames medianglättet und als Maximum
über Chamber- und Kick-Phase geführt.

Da die Weltkoordinaten hüftzentriert sind, misst der Wert die Fußbewegung **relativ
zur Hüfte**, nicht absolut im Raum. Der Anteil, der aus der Verlagerung des ganzen
Körpers stammt, fehlt. Beim Front Kick ist der Fehler klein, beim Spinning Back Kick
deutlich größer. Der angezeigte Wert unterschätzt daher systematisch.

---

## 5 Rückmeldung an den Nutzer

### 5.1 Technikauswahl

Die Technik wird **vor** der Analyse vom Nutzer gewählt (Schaltflächen in
`PoseAnalysis.tsx`, Konstante `kick_types`). Eine automatische Erkennung findet
nicht statt. Ebenso wählt der Nutzer die Modellvariante (lite / full / heavy),
was für Endanwender ungewöhnlich ist und als Prototyp-Eigenschaft benannt werden sollte.

### 5.2 Ausgabe nach der Analyse

**Annotiertes Video** (`/download/{job_id}`, Autoplay und Endlosschleife):
- Skelett-Overlay aus den 33 Landmarken, Verbindungslinien grün
- Punkt- und Liniendicke skalieren mit der Videoauflösung (`make_pose_styles()`)
- Debug-Overlay mit aktueller Phase, Kniewinkel, Hüftbeugung, Kickseite,
  Oberschenkelneigung, Startframe, Kickdauer, Sichtbarkeit, Hüftausrichtung;
  beim Spinning Back Kick zusätzlich Treffer-Ausrichtung, Nachdrehung,
  Gesamtdrehung, Fußabstand und Fußhöhe (`kick_analyzer.py`)

**Ergebniskarte** (`src/components/ResultCard.tsx`):
- Gesamtwert als Zahl von 0 bis 100
- Fußgeschwindigkeit in km/h
- Verbale Einordnung: „Sehr gut!" ab 90, „Gut!" ab 75, „Solide" ab 50, sonst „Übungsbedarf"
- Sternebewertung von 1 bis 5
- Je Kriterium eine Zeile mit Bezeichnung, Balken und Prozentwert
- Ampelfarben: grün ab 90 %, gelb ab 75 %, rot darunter (`scoreColor()`)
- Tipps-Liste mit den Rückmeldungen aller nicht bestandenen Kriterien

**Einzelbilder oder Keyframes werden nicht ausgegeben.**

### 5.3 Weitere Funktionen

- Speichern im Verlauf (`POST /save`), Ablage als MP4 plus JSON in `saved/`
- Der Verlauf ist pro Gerät getrennt: Das Frontend erzeugt beim ersten Aufruf eine
  zufällige Kennung im `localStorage` (`getClientId()` in `src/config.ts`), die beim
  Speichern mitgeschrieben und beim Abruf als Filter verwendet wird. Das ist
  Personalisierung, keine Zugriffskontrolle.
- Gürtelabhängige Vokabelliste und Vokabelabfrage im Multiple-Choice-Format
  (`VocabularyPage.tsx`, `VocabQuiz.tsx`)

---

## 6 Aufnahmebedingungen

### 6.1 Kommunizierte Anforderungen

Sämtliche Anforderungen werden ausschließlich als Hinweistext vermittelt
(`src/components/FilmingTips.tsx`), der beim Öffnen der Analyseseite automatisch
erscheint und über eine Schaltfläche erneut aufrufbar ist:

- Aufnahme von der Seite oder schräg im 45°-Winkel
- Nur eine Person im Bild
- Ganzer Körper sichtbar, auch im höchsten Punkt des Kicks
- Abstand etwa 3 bis 4 Meter, Kamera möglichst ruhig
- Nur ein Kick pro Video
- Gute Beleuchtung, ruhiger Hintergrund
- Hochkant-Aufnahme

### 6.2 Technisch erzwungene Anforderungen

**Keine.** Das System prüft weder Auflösung noch Bildrate, Personenanzahl,
Perspektive oder Ausleuchtung. Konkret:

| Aspekt | Verhalten |
|---|---|
| Bildrate | Wird aus dem Video gelesen, Rückfall auf 30 fps |
| Auflösung | Wird ohne Prüfung auf die Hälfte reduziert |
| Mehrere Personen | Es wird stets die erste erkannte Person verwendet |
| Kleidung | Keine Anforderung, keine Prüfung |
| Perspektive | Keine Prüfung |

### 6.3 Verhalten bei ungeeigneten Videos

- **Datei nicht lesbar** → Auftragsstatus `error`, Meldung „Konnte Video nicht öffnen."
- **Keine Person erkannt** → `process_frame()` gibt den Frame unverändert zurück,
  es werden keine Messwerte erhoben
- **Kein Kick erkannt** → `kick_detected` bleibt `False`, `evaluate()` gibt nur die
  ersten Kriterien zurück, die dann bei 0 % liegen

> Ein ausdrücklicher Hinweis „Kick nicht erkannt, bitte Aufnahme prüfen" existiert
> nicht. Der Nutzer sieht stattdessen eine verkürzte Kriterienliste mit niedriger
> Punktzahl. Das ist ein offener Punkt und für die Diskussion der Nutzerbefragung
> relevant.

---

## 7 Entscheidungen und Grenzen

### 7.1 Bewusste Entwurfsentscheidungen

| Entscheidung | Alternative | Begründung |
|---|---|---|
| Regelbasierte Bewertung | Klassifikation über neuronales Netz | Nachvollziehbarkeit der Rückmeldung; kein annotierter Datensatz verfügbar |
| Pose Estimation serverseitig | Im Browser über `tasks-vision` | Einheitliche Rechenleistung unabhängig vom Endgerät; Zugriff auf die heavy-Variante |
| Technikauswahl durch den Nutzer | Automatische Technikerkennung | Vermeidet eine zusätzliche Fehlerquelle vor der eigentlichen Bewertung |
| Lineare Abbildung statt Bestanden/Nicht bestanden | Binäre Kriterien | Abgestufte Rückmeldung gilt als lernwirksamer (Knowledge of Performance) |
| Erkennung großzügig, Bewertung streng | Gemeinsame Schwellen | Sonst könnte „Bein nicht gestreckt" nie als Ergebnis erscheinen — der Kick würde gar nicht erkannt |
| Endausrichtung im Treffmoment | Menge der Drehung | Die Drehmenge hängt von der Ausgangsstellung ab, die Endausrichtung nicht |
| Rechamber als Umkehrmessung | Vergleich mit der Chamber-Position | Relativ zum eigenen Streckungsmaximum, dadurch personenunabhängig |
| Segmentwinkel für die Kniehöhe | Gelenkwinkel Schulter–Hüfte–Knie | Der Gelenkwinkel wird durch Zurücklehnen des Oberkörpers verfälscht (1° Neigung = 1° Fehler) |
| Fußbahn statt Kniewinkel (Back Kick) | Einheitliche Kniewinkel-Erkennung | Beim Back Kick streckt das Knie kaum; das Absetzen des Beins wäre nicht unterscheidbar |
| Verarbeitung bei halber Auflösung | Volle Auflösung | Verkleinerung glättet Bewegungsunschärfe, MediaPipe arbeitet dadurch stabiler |
| Durchgängig dunkle Gestaltung | Anpassung an die Systemeinstellung | Die Anwendung war auf Geräten im hellen Modus unlesbar (weiße Schrift auf weißem Grund) |

### 7.2 Bekannte Schwächen

**Messtechnisch**

1. **Zusammenbruch der Hüftausrichtung bei geringer Beckenbreite in der Draufsicht.**
   Wird die projizierte Verbindungslinie beider Hüften kürzer als etwa 8 cm, ist ihre
   Richtung nicht mehr bestimmbar und die Hüftausrichtung wird zu Rauschen. In
   Testrechnungen: bei 10 cm Trennung Streuung ±1,2°, bei 5 cm ±30°, bei 2 cm ±76°.
   Betroffen ist vor allem die seitliche Aufnahme, bei der eine Hüfte die andere verdeckt.

2. **Extremwert-Aggregation.** Chamber Winkel, Beinstreckung, Hüftrotation und
   Kniehöhe werden weiterhin über `min()` bzw. `max()` aus einem einzelnen Frame
   gewonnen. Ein Ausreißer bestimmt damit den Wert. Nur Standbein und
   Fußgeschwindigkeit wurden bereits auf Medianbildung umgestellt.

3. **Faltung der Hüftausrichtung beim Roundhouse Kick.** Durch `min(raw, 180 - raw)`
   wird eine stark eingedrehte Hüfte (Rohwert über 90°) auf einen niedrigen Wert
   abgebildet und ist von fehlender Rotation nicht unterscheidbar. Ein Rohwert von
   150° ergibt denselben Kriterienwert wie 30°.

4. **Sichtbarkeitswerte ohne Wirkung** (siehe 2.3).

5. **Uneinheitliche Normierung** (siehe 2.4).

6. **Geschwindigkeit hüftzentriert** (siehe 4.6).

**Konzeptionell**

7. Ein Kick pro Video, da die Kickseite nur einmal bestimmt wird.
8. Keine Gewichtung der Kriterien.
9. Frühabbrüche verzerren den Gesamtwert (siehe 4.4).
10. Sämtliche Schwellenwerte sind expertenbasiert gesetzt und nicht empirisch
    kalibriert. Für Front- und Roundhouse Kick liegen zudem keine publizierten
    Referenzwerte zur Transversalrotation vor.

### 7.3 Spezifische Probleme beim Spinning Back Kick

Dieser Punkt schließt unmittelbar an Wang et al. an, die den Roundhouse Kick wegen
der Körperrotation und der daraus folgenden Verdeckungen als schwächste ihrer drei
Techniken beschreiben. Bei einer Technik mit noch stärkerer Rotation treten
zusätzliche Effekte auf:

1. **Die aufsummierte Beckendrehung unterschätzt systematisch.** Sie beginnt erst
   zu zählen, wenn die Kickseite feststeht — also nach dem Anheben des Knies. Die
   Einleitung der Drehung fällt heraus. Zusätzlich verwirft der Sprungfilter
   Änderungen über 30° pro Frame, was bei schnellen Drehungen weitere Anteile entfernt.

2. **Die Hüftausrichtung misst nicht die Raumdrehung.** Sie ist körperrelativ
   (siehe 4.2). Eine Ausführung, bei der sich die Person weit dreht, aber seitlich
   trifft, wird korrekt als Side-Kick-Konfiguration erkannt; die Rückmeldung „zu
   wenig eingedreht" beschreibt jedoch die Becken-Bein-Beziehung, nicht die
   subjektiv erlebte Drehung. Das ist erklärungsbedürftig gegenüber dem Nutzer.

3. **Über-Rotation ist nur eingeschränkt erkennbar**, da die Hüftausrichtung bei
   180° sättigt. Der Rohwert kann nicht zwischen „noch nicht weit genug" und
   „bereits darüber hinaus" unterscheiden. Die aufsummierte Nachdrehung dient als
   Hilfsgröße, ist aber wegen Punkt 1 unzuverlässig und bestimmt daher nur noch den
   Rückmeldungstext, nicht die Punktzahl.

4. **Verdeckung.** Bei seitlicher Aufnahme und starker Rotation stehen die Hüften
   hintereinander. Genau in dieser Konstellation tritt der unter 7.2.1 beschriebene
   Zusammenbruch auf.

---

## 8 Offene Punkte

Diese Punkte sind analysiert, aber noch nicht umgesetzt:

- Faltung beim Roundhouse Kick durch Sprungfilter ersetzen, Kriterium auf den
  vollen Wertebereich umstellen
- Frühabbruch beim Roundhouse Kick entfernen
- Bei nicht erkanntem Kick eine ausdrückliche Meldung statt einer Punktzahl anzeigen
- Gültigkeitsprüfung der projizierten Beckenbreite einführen
- Verbleibende Extremwert-Aggregationen auf Perzentile umstellen
- Kniehöhe auf die Beinlänge normieren
- Zwei in der Technikliteratur genannte, bislang nicht erfasste Kriterien:
  Pivot des Standfußes beim Roundhouse Kick sowie die Linie Fuß–Knie–Hüfte–Schulter
  im Treffmoment beim Back Kick
- Empirische Kalibrierung der Schwellenwerte gegen ein Trainerurteil