# Governance for cuda-sample

## Scope

This document describes the governance model for the cuda-sample project,
including roles, responsibilities, and decision making processes.

## Roles and responsibilities

### Maintainers

Maintainers are the individuals responsible for the long-term health and direction
of the project. They have the final say on key decisions but are expected to
collaborate with the community and seek consensus when possible.

- **Responsibilities**:
    - Oversee the technical direction of the project.
    - Make final decisions on contributions and community matters.
    - Ensure code quality and maintainability.
    - Manage project releases and documentation.
    - Resolve conflicts and mediate disputes within the community.
    - Actively participate in community discussions.

### Contributors

Contributors are individuals who actively participate in the project by submitting
code, bug reports, documentation, or other contributions. Contributors may be
individuals, teams, or organizations.

- **Responsibilities**:
    - Follow the project’s guidelines for contributions.
    - Engage in discussions and provide feedback on project decisions.
    - Respect the project's code of conduct.
    - Strive for high-quality contributions that adhere to the project's goals.

### Community

The community consists of all individuals interested in the project, including end users,
contributors, and other stakeholders. The community is encouraged to contribute ideas,
report issues, and engage in discussions.

- **Responsibilities**:
    - Provide feedback on project decisions and contribute to discussions.
    - Participate in community events and initiatives.
    - Respect the project’s code of conduct.

## Decision Making

### Routine Decisions

Routine decisions — bug fixes, minor features, documentation, dependency updates — proceed
via a PR. At least one maintainer approval is required for technical acceptance; a maintainer
then merges after verifying project requirements are met.

### Core Decisions

Core decisions that involve significant changes to the project require a prior proposal before
implementation work begins. A change is significant if it involves:

- Architectural changes
- Major new features or subsystems
- Breaking changes to APIs or behavior
- Changes to the contribution model, release cadence, or this governance document

A proposal is a GitHub issue that explains what is changing, why, what alternatives were
considered, and the impact on existing contributors and users.

Core decisions are made by the maintainers. For contributors who want to propose and contribute
a major change, the process is as follows:

- A proposal for a major change will be submitted as an issue.
- The maintainers will discuss the proposal and seek feedback from the community.
- Once the discussion is concluded, maintainers will make the final decision.

## Changes to Governance

A change to this document follows the process for core decisions defined above. Maintainers make
the final decision on changes to the governance model.
