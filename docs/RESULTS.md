# Results

AndroidWorld `2655e27`, family `android_world` (116 tasks), seed 30, one instance per task, `loop_rules=default`
(repeat limit 3), 1 s pause after each action. Android emulator (Pixel 6, API 33, 1080×2400) on an Apple-silicon Mac.
Decision model `typesafe/jev-1.14` and text model `inception/mercury-2.5` in both runs. Each configuration was run once.

| | `aw_full_hybrid_v2` (vision) | `aw_full_mobile114_v1` (text-only) |
|---|---|---|
| Agent | `mobile_jev_hybrid_aw` | `mobile_jev_aw` |
| Jev input | screenshot + element options | element list + screen text |
| Date | 2026-10-29 | 2026-10-29 |
| Success | **41 / 116 (35.3%)** | 39 / 116 (33.6%) |
| Steps | 1,216 | 1,497 |
| Mean seconds per step | 2.33 | 2.29 |
| Wall time | 59 min | 68 min |
| Cost: Jev / text / total | $0.292 / $0.009 / $0.300 | $0.257 / $0.012 / $0.269 |
| Jev input tokens | 6.94 M | 6.12 M |
| Ended by: done | 45 | 43 |
| answer | 12 | 12 |
| stuck (loop rule) | 21 | 24 |
| budget_exhausted | 16 | 26 |
| blocked | 11 | 8 |
| loading_timeout | 8 | 3 |
| errored | 3 | 0 |

- Vision vs text-only: 10 tasks newly solved, 8 lost. Solved only with the screenshot: AudioRecorderRecordAudioWithFileName,
  MarkorDeleteAllNotes, MarkorDeleteNote, NotesMeetingAttendeeCount, NotesRecipeIngredientCount,
  RecipeDeleteMultipleRecipes, SimpleCalendarLocationOfEvent, SimpleSmsResend, SimpleSmsSendClipboardContent,
  TasksIncompleteTasksOnDate. Lost: BrowserMaze, ClockStopWatchPausedVerify, RetroCreatePlaylist,
  SaveCopyOfReceiptTaskEval, SimpleCalendarDeleteOneEvent, SimpleCalendarEventsInNextWeek,
  SportsTrackerLongestDistanceActivity, SportsTrackerTotalDurationForCategoryThisWeek.
- With 8–10 tasks flipping between configurations, a 2-task difference is within run-to-run variation; the clearer
  effect is fewer steps (1,497 → 1,216) and fewer budget-exhausted tasks (26 → 16).
- The 3 errored tasks (TasksDueNextWeek, TasksDueOnDate, TasksHighPriorityTasks) hit an adb server restart while a
  second emulator was running another suite; all three failed in the text-only run as well.
- An earlier vision run with coordinate-only history (`history=coords`) reached 34/116: without the element labels
  in the history ("Tap OK."), Jev rarely chose DONE after finishing (36 done-terminated tasks vs 45).
- Per-task outcomes: `results/<run>/tasks.jsonl` (`is_successful`, `episode_length`). Run settings:
  `results/<run>/config.json`.
