# Mac Studio research queue

[Open the live queue on the controlling Mac](http://127.0.0.1:8769/). Each run shows its training progress when available. Expand a finished experiment to see its decision and scores. **Move up next** promotes a study as soon as it qualifies. **Hold after current run** lets the current experiment finish and persists across restarts. **Resume dispatch** releases the hold.

The Studio checks the queue immediately after each exit. It runs one experiment at a time and chooses the highest-priority eligible entry. A failed scientific threshold closes only the follow-ups that require that outcome. Independent work remains eligible. A process failure gets one retry. The queue can drain if there is no justified experiment or a research dependency fails. The recurring research review checks the backlog every ten minutes while available, replenishes it with targeted comparisons, and reviews scientific decisions separately from dispatch.

The Studio worker runs under the logged-in user's launchd session. Each live experiment holds a macOS idle-sleep assertion tied to its process. The display can sleep, and the assertion expires when the process exits. Closing the browser or the controlling SSH session leaves dispatch running. The web interface on the controlling Mac uses SSH to read or edit the queue and listens only on loopback. Training and model checkpoints stay on the Studio.

## Change what runs

Commit the protocol and runnable script first. Add its full source commit, repository, unique job ID and arguments to [queue.json](queue.json). An optional `after` maps prior jobs to the precise decision string required, or `any` for any successful completion. List dependencies before dependents. Use a new ID for revised experiments to preserve earlier receipts. Run the installer from this checkout.

```sh
python3 studio_queue/install.py
python3 studio_queue/install_dashboard.py
```

The first installer uploads missing immutable source snapshots and atomically hands the new manifest to the running Studio worker. The second starts the local browser service. Dashboard priority survives manifest updates. To return to manifest order run `front` with no names. The same controls work through SSH, for example

```sh
ssh genflick-m5ultra '~/research/routed-expert-splice/.venv/bin/python ~/research/architecture-queue/worker.py status'
ssh genflick-m5ultra '~/research/routed-expert-splice/.venv/bin/python ~/research/architecture-queue/worker.py next register-addressing'
```

Each job writes in `~/research/architecture-queue/runs/<id>/` on the Studio. Its receipt records the source, attempt, timestamps and process outcome. Its result records the scientific decision. The append-only `events.jsonl` records handoffs. Model runs that implement checkpointing resume from their last complete checkpoint. A live hold defers new work. User-directed priority changes apply at the next handoff.
