# Distributed Systems Architecture

## Core Principles
- [[Scalability]]: System can handle growing workloads
- [[Availability]]: System remains operational despite failures
- [[Partition Tolerance]]: System continues functioning during network partitions
- [[Consistency]]: All nodes see the same data at the same time

> CAP Theorem states you can only guarantee two of the three: Consistency, Availability, and Partition Tolerance

### Common Patterns
| Pattern | Purpose | Examples |
|---------|---------|----------|
| Microservices | Modularity | Netflix, Amazon |
| Event Sourcing | State tracking | CQRS systems |
| Circuit Breaker | Failure isolation | Hystrix |

```java
// Circuit Breaker pattern pseudo-code
class CircuitBreaker {
    enum State { CLOSED, OPEN, HALF_OPEN }
    private State state = State.CLOSED;
    
    Result call(Function service) {
        if (state == State.OPEN) {
            return fallback();
        }
        
        try {
            Result result = service.execute();
            recordSuccess();
            return result;
        } catch (Exception e) {
            recordFailure();
            return fallback();
        }
    }
}
```

Related to [[note_15.md|System Design Principles]] and my project on [[note_18.md|Cloud Migration Strategy]].

#software #distributedsystems #architecture #engineering