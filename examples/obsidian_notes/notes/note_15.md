# System Design Principles

## Fundamental Concepts
- [[Separation of Concerns]]: Divide system into distinct sections
- [[Single Responsibility]]: Each component has one job
- [[SOLID Principles]]: Guidelines for maintainable OO design
- [[Design Patterns]]: Reusable solutions to common problems

> "Good architecture makes the system easy to understand, easy to develop, easy to maintain, and easy to deploy." - Robert C. Martin

### Common Architectural Patterns
| Pattern | Use Case | Examples |
|---------|----------|----------|
| MVC | UI Applications | Django, Rails |
| Microservices | Complex Systems | Netflix, Uber |
| Event-Driven | Reactive Systems | Kafka-based apps |

```
┌───────────┐    ┌───────────┐    ┌───────────┐
│   Client  │───►│  Service  │───►│  Database │
└───────────┘    └───────────┘    └───────────┘
                       │
                       ▼
                ┌───────────┐
                │   Cache   │
                └───────────┘
```

## Trade-offs to Consider
- [[Performance vs Maintainability]]
- [[Simplicity vs Flexibility]]
- [[Monolithic vs Distributed]]

This connects to my work on [[note_09.md|Distributed Systems Architecture]] and lessons from [[note_19.md|System Failure Post-Mortem]].

The approach I found most valuable: start with clear requirements and constraints before jumping to specific technologies.

#architecture #software #engineering #design