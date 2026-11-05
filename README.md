# Sisi-pro: AI x Cybersecurity Projects

Hands-on projects at the intersection of **AI engineering** and **cybersecurity**: using machine learning to detect threats, and studying where these systems fail.

## Projects

| # | Project | What it does | Stack |
|---|---|---|---|
| 01 | [Network Intrusion Detection](01-network-intrusion-detection/) | Flags malicious network connections and classifies the attack family (DoS, Probe, R2L, U2R) on the NSL-KDD benchmark. Analyzes why detection drops on never-before-seen attacks and how the alert threshold trades detection for false alarms. | Python, scikit-learn, pandas |
| 02 | [Prompt Injection Scanner](02-prompt-injection-scanner/) | Detects prompt injection attacks on LLM apps, in user input and hidden inside documents an AI agent reads. Combines de-obfuscation, signature rules and an ML classifier, and compares all three. | Python, scikit-learn, regex |

More projects coming soon.

## Focus areas

- Machine learning for threat detection
- Evaluating detectors honestly: novel attacks, class imbalance, false alarm cost
- Security of AI systems
