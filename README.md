# Delegate 

## Methodology
Test llm and slm only on Math datasets.

Then test the delegate framework to see if they work

the idea is using ReAct framework, llm reasoning, breakdown problem and delegate small task to SLM.

## Result

### LLM
📈 Summary:
   Accuracy: 78.50% (157/200)
   Avg Latency: 1.27s
   Avg Tokens: 94 → 323

### Delegate
📈 Summary:
   Accuracy: 73.00% (146/200)
   Avg Latency: 17.58s total
     ├─ LLM: 2.95s
     └─ SLM: 14.63s
   Tool Calls: 2.9 per problem
   Avg Tokens: 1929 → 355

### SLM
📈 Summary:
   Accuracy: 82.00% (164/200)
   Avg Latency: 9.66s
   Avg Tokens: 91 → 296"

### Conclusion
Fail miserably.
The LLM doesn't understand the SLM well and the delegate tasks are too hard or too obvious.
