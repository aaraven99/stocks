# Architecture

```text
provider adapters -> timestamp validation -> features -> regime + strategies
                                                       -> rank -> risk gate
                                                             -> evidence agents
                                                                  -> report / SQLite paper ledger
```

All signals are calculated from completed bars. A signal observed after session *t* may first
trade at session *t+1* open. The LLM layer receives structured evidence after ranking and is
incapable of changing a numeric decision.

