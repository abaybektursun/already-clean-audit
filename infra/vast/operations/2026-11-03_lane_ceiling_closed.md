# Vast lane ceiling, closed for the REUSE program, 2026-11-03

State of record: `infra/vast/budget.json` program ceiling $42.00 (raised 36 → 42 on 2026-11-02 under the user's full-control grant); spent $33.29 before the SPEC4 rental and $2.40 on it (instance 58846849, 3.166 GPU-hours); the user's vast credit read $27.00 after that rental. The ceiling tracks the user's credit and rises only on the user's top-up.

Decision: no further vast run is needed by the REUSE program. Every run after SPEC4 (SPEC5 to SPEC12 and the two ALWAYSON specs) ran on the AWS lane under the standing directive, every queue row of `studies/reuse/SPEC4_birth.md` is closed with a run outcome, and no successor item is queued. The ceiling is therefore not raised and no top-up is requested. A future program that chooses the vast lane by price starts from this state: $42.00 ceiling, $35.69 spent, $27.00 credit; the first step is the owner's top-up, then `budget.json`.

Lifecycle: no vast instance is running (the SPEC4 rental was destroyed by the lane on 2026-11-02); the lane's watchdog is idle.
