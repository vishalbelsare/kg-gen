# System Failure Post-Mortem

## Incident Overview
- **Date**: October 5, 2023
- **Duration**: 3 hours 42 minutes
- **Impact**: 78% of users experienced service degradation
- **Root Cause**: Database connection pool exhaustion

> "The most severe bugs are not those that crashed your systems; they're the ones that silently corrupted your data." - Unknown

### Timeline
| Time | Event |
|------|-------|
| 14:23 | First error alerts triggered |
| 14:45 | On-call engineer acknowledged |
| 15:10 | Initial diagnosis (incorrect) |
| 16:30 | Root cause identified |
| 18:05 | Service restored |

## Technical Details
```sql
-- The problematic query
SELECT * FROM transactions 
WHERE created_at BETWEEN ? AND ? 
ORDER BY amount DESC;
-- Missing index on created_at column
```

This relates to our earlier discussions on [[note_15.md|System Design Principles]] about performance considerations and the importance of proper [[database indexing]].

### Action Items
1. Add monitoring for [[connection pool]] utilization
2. Implement [[circuit breaker pattern]] in database access layer
3. Create [[runbook]] for similar incidents
4. Schedule [[postmortem review]] with all engineering teams

The incident highlights the need for better [[observability]] and aligns with our [[note_12.md|Quarterly Planning]] focus on infrastructure resilience.

#incidents #reliability #engineering #databases