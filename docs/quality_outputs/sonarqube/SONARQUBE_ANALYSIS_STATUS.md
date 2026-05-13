# SonarQube Analysis Output Status

Date: 13.05.2026

## Installation

- SonarQube Community Build ZIP: `C:\Users\metehan\Downloads\sonarqube-26.5.0.122743.zip`
- Extracted SonarQube home: `C:\tmp\sonarqube-26.5.0.122743`
- SonarScanner CLI: `C:\tmp\sonar-scanner\sonar-scanner-8.0.1.6346-windows-x64`
- Java used by SonarQube Server: `C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot\bin\java.exe`
- SonarQube URL: `http://localhost:9000`
- Project key: `ifds-image-forgery`

Note: The local machine has low free disk space on `C:`. For this local evaluation run, Elasticsearch disk threshold was disabled after startup so the embedded evaluation database could run.

## Test and Coverage

Command:

```bash
python -m pytest tests -q --cov=src --cov-report=xml --cov-report=term-missing
```

Result:

```text
58 passed
Coverage XML written to file coverage.xml
lines-valid: 1202
lines-covered: 1142
line-rate: 0.9501
pytest coverage: 95.01%
```

Coverage artifact:

```text
docs/quality_outputs/sonarqube/coverage.xml
```

## SonarScanner Run

Command:

```bash
sonar-scanner.bat -Dsonar.host.url=http://localhost:9000 -Dsonar.token=<local-token>
```

Result:

```text
EXECUTION SUCCESS
ANALYSIS SUCCESSFUL
Dashboard: http://localhost:9000/dashboard?id=ifds-image-forgery
```

## SonarQube Dashboard Metrics

```text
Quality Gate: OK / Passed
Coverage: 94.8%
Bugs: 0
Vulnerabilities: 0
Security Hotspots: 0
Code Smells: 14
Duplications: 3.0%
Lines of Code: 2536
Reliability Rating: A
Security Rating: A
Maintainability Rating: A
```

Dashboard screenshot:

```text
docs/quality_outputs/sonarqube/sonarqube_dashboard.png
```
