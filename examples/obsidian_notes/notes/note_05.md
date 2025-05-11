# Meeting Notes: ML Engineering Team - 2023-10-12

## Attendees
- Sarah (Team Lead)
- [[Michael]] (ML Architect)
- [[Priya]] (Data Scientist)
- Me (ML Engineer)

## Agenda
- Model performance metrics
- Production deployment pipeline
- Feature engineering improvements

> Key takeaway: Need to balance model complexity with inference speed

### Action Items
- Investigate [[feature drift]] in production data
- Implement [[A/B testing]] framework for model variants
- Optimize [[batch inference]] processing for large datasets

The team agreed that our current [[accuracy metrics]] don't capture the business impact well enough. We need to develop custom metrics aligned with business outcomes.

```python
# Proposed evaluation function
def business_aligned_metric(y_true, y_pred, business_weights):
    # Custom implementation here
    pass
```

Next meeting scheduled for [[note_12.md|Quarterly Planning]].

#meetings #machinelearning #teamwork #MLOps