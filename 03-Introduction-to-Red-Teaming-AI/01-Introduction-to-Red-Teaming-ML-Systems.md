# Introduction to Red Teaming ML-based Systems

Red teaming ML-based systems is complex and requires understanding of their many components and algorithms. This module lays the groundwork for assessing the security of ML-based systems.

## What is Red Teaming?

A **penetration test** is a focused, time-bound security assessment in which skilled professionals follow a structured methodology to identify vulnerabilities. Penetration testing is done within a specific scope. For example, a web application penetration test involves actively discovering and testing whether vulnerabilities exist in a web app.

Red teaming, on the other hand, is a step beyond penetration testing. In fact, there are three kinds of security assessments:

- Red Team
- Penetration Testing
- Vulnerability Assessment

The following diagram displays these hierarchically in the form of a pyramid. At the very bottom of the pyramid, is a **Vulnerability Assessment**, where a brief description of it is included. At the middle of the pyramid is **Penetration Testing**, where again, a brief description of it is included. Finally, at the top of the pyramid is a **Red Team Assessment**, where, again, a brief description of it is included.

**Vulnerability Assessment**

A vulnerability assessment is an automated or manual process designed to identify and catalog potential weaknesses in a system, application, or network. It primarily focuses on discovery and prioritization without actively exploiting the findings. Tools like Nessus or OpenVAS are commonly used.

For more information about vulnerability assessments, see the Vulnerability Assessment module on HTB Academy.

**Red Team Assessment**

A red team assessment is an advanced adversarial simulation that mimics real-world attackers. Rather than following a predefined checklist or scope, a red team operates with flexibility to employ Tactics, Techniques, and Procedures (TTPs) similar to those used by actual threat actors. This includes not only technical attacks but also social engineering, phishing campaigns, and even physical intrusion (depending on the engagement scope).

Stealth and persistence are emphasized. Red teams try to remain undetected while exploring how far they can go. Their goal is to identify gaps in defensive coverage, test the effectiveness of the blue team, and reveal weaknesses in detection and response capabilities. A red team engagement can take weeks or even months, operating in a manner that reflects the behavior of advanced adversaries.

For more information about red team assessments, see the Introduction to Information Security module on HTB Academy.

## Red Teaming ML-based Systems

ML-based systems can carry unique vulnerabilities. They often rely on large datasets, make decisions based on statistical inference, and are built using complex architectures. Red teaming is often preferred over penetration testing for these systems.

One of the main reasons for choosing red teaming over penetration testing in an ML context is time. Some attack techniques are advanced and take longer to complete. Moreover, the interconnected nature of ML components, such as data pipelines, model APIs, and training environments, means that a thorough assessment may benefit from broader exploration. This can be challenging to scope in a conventional penetration test.
